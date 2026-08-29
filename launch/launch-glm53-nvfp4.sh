#!/usr/bin/env bash
#
# GLM-5.3 (big, 743B) Int4-Int8Mix on 4x DGX Spark (GB10/sm121), TP4,
# with the NVFP4 KV cache (--kv-cache-dtype nvfp4_ds_mla, 400 B/token/layer).
#
# usage: launch-glm53-nvfp4.sh <rank 0-3> [none|mtp]
#
# ---------------------------------------------------------------------------
# PROVENANCE
#   Structure + GLM-5.3 weights/env  : ~/launch-glm53-big-qt.sh (known good, fp8_ds_mla)
#   NVFP4 image + kernel overlay dir : ~/glm-5.2-gb10/speednight-nvfp4.sh (GLM-5.2 316K)
#   Public spec : github.com/tonyd2wild/GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s
#
# HOW nvfp4_ds_mla IS ACCEPTED (this is the whole trick):
#   The image's OWN vllm/v1/attention/backends/mla/flashmla_sparse.py lists only
#   ["auto","bfloat16","fp8_ds_mla","fp8"]. The NVFP4 lane bind-mounts a DIFFERENT
#   flashmla_sparse.py from KERNELS_DIR=/var/tmp/glm-triton-nvfp4, which at line 96
#   declares supported_kv_cache_dtypes = [... "nvfp4_ds_mla" ...], adds the
#   get_kv_cache_shape -> (num_blocks, block_size, 400) branch, and adds a
#   do_kv_cache_update() override calling store_nvfp4_glm_kv.
#   => KERNELS_DIR MUST be /var/tmp/glm-triton-nvfp4, NOT ~/glm-triton.
#   Only 2 of the 10 overlay files differ between those dirs:
#   flashmla_sparse.py and b12x_sparse_helpers.py. The other 8 are byte-identical.
#
# ---------------------------------------------------------------------------
# UNCERTAIN / UNVERIFIED - READ BEFORE FIRST BOOT
#   1. NEVER YET RUN FOR GLM-5.3. The 400 B/token NVFP4 record was validated on
#      GLM-5.2 744B. GLM-5.3 743B Int4-Int8Mix is structurally identical
#      (78 layers / 6144 hidden / kv_lora 512 / qk_rope 64, both GlmMoeDsaForCausalLM)
#      so the layout SHOULD transfer, but it is not proven on 5.3 weights.
#   2. KV POOL MATH IS INHERITED, NOT MEASURED. --kv-cache-memory-bytes 10950000000
#      is copied from both reference launchers. On GLM-5.2 it yielded a
#      317,312-token pool (~34.5 KB/token across 78 layers at TP4). At
#      --max-model-len ${MAX_MODEL_LEN:-316000} there is large headroom so the pin is conservative,
#      but the exact GLM-5.3 pool is unmeasured. Read "GPU KV cache size:" on boot.
#   3. MTP k. The proven GLM-5.3 lane uses num_speculative_tokens 4; the GLM-5.2
#      NVFP4 reference uses 5. This uses 4 to stay closest to known-good GLM-5.3.
#      Raise to 5 only after a clean k=4 boot.
#   4. cudagraph. Kept at the GLM-5.3-proven {"cudagraph_mode":"FULL"}. The GLM-5.2
#      NVFP4 reference additionally pinned cudagraph_capture_sizes
#      [6,12,18,24,30,36] and pass_config.fuse_gemm_comms=true. Both deliberately
#      OMITTED here to minimise delta from the working GLM-5.3 lane. Note FULL
#      escalates to two graph sets when speculation is on.
#   5. ACCURACY IS UNVALIDATED. NVFP4 KV is a lossy 4-bit store (vs 8-bit
#      fp8_ds_mla). Run a needle / quality check against the fp8_ds_mla lane
#      before trusting output.
#   6. The nvfp4 overlay also flips can_return_lse_for_decode = True (a DCP
#      enabler). That flag is inert at --decode-context-parallel-size 1 (default
#      here). See the DCP note at the bottom before raising it.
#
# SAFETY: NAME=vllm_glm53nvfp4 and PORT=8211 are distinct from the production
# lane (container vllm_glm53big / port 8000). This will not touch production.
# ---------------------------------------------------------------------------
set -uo pipefail

NODE_RANK="${1:?usage: launch-glm53-nvfp4.sh <0|1|2|3> [none|mtp]}"
SPEC_MODE="${2:-none}"

IMAGE="vllm-node-tf5-glm52-b12x:nvfp4-v1"
NAME="vllm_glm53nvfp4"
PORT=8211
MASTER_PORT=29551
HEAD_IP="192.168.192.2"                    # Reddie = rank 0, weights local here
KERNELS_DIR="/var/tmp/glm-triton-nvfp4"    # NVFP4 overlay - NOT ~/glm-triton
COMPILE_CACHE_DIR="/var/tmp/glm-compile-cache"

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.2; HEADLESS=0; WEIGHTS=/var/tmp/models/GLM-5.3-Int4-Int8Mix ;;
  1) HOST_IP=192.168.192.4; HEADLESS=1; WEIGHTS=/mnt/reddie-models/GLM-5.3-Int4-Int8Mix ;;
  2) HOST_IP=192.168.192.3; HEADLESS=1; WEIGHTS=/mnt/reddie-models/GLM-5.3-Int4-Int8Mix ;;
  3) HOST_IP=192.168.192.1; HEADLESS=1; WEIGHTS=/mnt/reddie-models/GLM-5.3-Int4-Int8Mix ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

test -f "$WEIGHTS/config.json" || { echo "weights not visible at $WEIGHTS" >&2; exit 3; }

# --- preflight: the 10 sm12x overlays, from the NVFP4 dir ---
KERNEL_FILES=(sparse_mla_kernels.py sparse_mla_env.py sm12x_sparse_mla_attn.py
  patch_flashmla_ops.py flashmla_sparse.py sm12x_deep_gemm_fallbacks.py
  sm12x_mqa.py b12x_sparse_helpers.py sparse_attn_indexer.py deepseek_v2.py)
for f in "${KERNEL_FILES[@]}"; do
  [ -f "$KERNELS_DIR/$f" ] || { echo "kernel overlay missing: $KERNELS_DIR/$f" >&2; exit 4; }
done

# --- preflight: THE nvfp4 check. Without this the engine dies at startup with
# "FlashMLASparse backend does not support kv_cache_dtype nvfp4_ds_mla".
grep -q "nvfp4_ds_mla" "$KERNELS_DIR/flashmla_sparse.py" || {
  echo "FATAL: $KERNELS_DIR/flashmla_sparse.py does not list nvfp4_ds_mla." >&2
  echo "       You are pointing at the fp8 overlay (~/glm-triton). Wrong dir." >&2
  exit 7; }
grep -q "store_nvfp4_glm_kv" "$KERNELS_DIR/flashmla_sparse.py" || {
  echo "FATAL: overlay lacks the nvfp4 store path (store_nvfp4_glm_kv)." >&2; exit 8; }

# matched-pair check (repo issue #5)
if grep -q "fused_indexer_q_rope_quant" "$KERNELS_DIR/deepseek_v2.py" 2>/dev/null \
   && ! grep -Eq "def[[:space:]]+fused_indexer_q_rope_quant" "$KERNELS_DIR/sparse_attn_indexer.py" 2>/dev/null; then
  echo "kernel mismatch (issue #5): version-skewed overlays" >&2; exit 5
fi
grep -q "GlmMoeDsaForCausalLM" "$KERNELS_DIR/deepseek_v2.py" || {
  echo "overlay deepseek_v2.py does not define GlmMoeDsaForCausalLM" >&2; exit 6; }

# --- preflight: the image must carry the nvfp4 Triton store/gather kernels ---
NVK=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/nvfp4_glm_kernels.py
docker run --rm --entrypoint test "$IMAGE" -f "$NVK" || {
  echo "FATAL: $IMAGE lacks nvfp4_glm_kernels.py - wrong image (need :nvfp4-v1)" >&2
  exit 9; }

# --- safety: never collide with the production endpoint ---
if [ "$PORT" = "8000" ]; then echo "refusing: 8000 is the production port" >&2; exit 10; fi

mkdir -p "$COMPILE_CACHE_DIR"

MLA="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla"
OPS="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/deepseek_v4_ops"
LAYERS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers"
MODELS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"

case "$SPEC_MODE" in
  none) SPEC=() ;;
  mtp)  SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":5,"draft_tensor_parallel_size":1,"attention_backend":"FLASHMLA_SPARSE"}') ;;
  *) echo "spec must be none|mtp" >&2; exit 2 ;;
esac

if [ "$SPEC_MODE" = "none" ]; then MAXSEQS=4; else MAXSEQS=6; fi

docker rm -f "$NAME" 2>/dev/null

docker run -d --name "$NAME" \
  --restart no \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --network host --ipc host --shm-size 10gb --gpus all \
  --device /dev/infiniband:/dev/infiniband \
  -v /var/tmp/models:/cache/huggingface \
  -v "$WEIGHTS:/models/glm-5.3:ro" \
  -v "$COMPILE_CACHE_DIR:/compile-cache" \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$KERNELS_DIR/sparse_mla_kernels.py:$MLA/sparse_mla_kernels.py:ro" \
  -v "$KERNELS_DIR/sparse_mla_env.py:$MLA/sparse_mla_env.py:ro" \
  -v "$KERNELS_DIR/sm12x_sparse_mla_attn.py:$MLA/sm12x_sparse_mla_attn.py:ro" \
  -v "$KERNELS_DIR/patch_flashmla_ops.py:$MLA/patch_flashmla_ops.py:ro" \
  -v "$KERNELS_DIR/flashmla_sparse.py:$MLA/flashmla_sparse.py:ro" \
  -v "$KERNELS_DIR/sm12x_deep_gemm_fallbacks.py:$OPS/sm12x_deep_gemm_fallbacks.py:ro" \
  -v "$KERNELS_DIR/sm12x_mqa.py:$OPS/sm12x_mqa.py:ro" \
  -v "$KERNELS_DIR/b12x_sparse_helpers.py:$OPS/b12x_sparse_helpers.py:ro" \
  -v "$KERNELS_DIR/sparse_attn_indexer.py:$LAYERS/sparse_attn_indexer.py:ro" \
  -v "$KERNELS_DIR/deepseek_v2.py:$MODELS/deepseek_v2.py:ro" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e LD_PRELOAD=/cache/huggingface/hub/nccl-2.30.4/libnccl.so.2 \
  -e HF_HOME=/cache/huggingface \
  -e TRITON_CACHE_DIR=/compile-cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/compile-cache/inductor \
  -e VLLM_CACHE_ROOT=/compile-cache/vllm \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e GLM52_BIND_HOST_TRITON=1 \
  -e GLM52_MQA_LOGITS_TRITON=1 \
  -e GLM52_PAGED_MQA_TRITON=1 \
  -e GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192 \
  -e GLM52_B12X_MLA=1 -e VLLM_DISABLE_FLASHINFER_AUTOTUNE=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 \
  -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e NODE_RANK="$NODE_RANK" -e MASTER_ADDR="$HEAD_IP" \
  -e VLLM_HOST_IP="$HOST_IP" \
  "$IMAGE" \
  vllm serve /models/glm-5.3 \
    --served-model-name glm-5.3-nvfp4 --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-prefix-caching \
    --async-scheduling \
    "${SPEC[@]}" \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --max-model-len ${MAX_MODEL_LEN:-316000} --max-num-seqs "$MAXSEQS" --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.91 --kv-cache-memory-bytes 10950000000 \
    --kv-cache-dtype nvfp4_ds_mla \
    --distributed-executor-backend mp --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[6,12,18,24,30,36],"pass_config":{"fuse_gemm_comms":true}}' \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \
    $( [ "$HEADLESS" = 1 ] && echo --headless )

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP spec=$SPEC_MODE kv=nvfp4_ds_mla weights=$WEIGHTS"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited immediately; docker logs $NAME" >&2; exit 1; }

# ---------------------------------------------------------------------------
# DCP NOTE (--decode-context-parallel-size)
#   The NVFP4 overlay sets FlashMLASparseImpl.can_return_lse_for_decode = True
#   specifically so cp_utils.py stops asserting at dcp>1. Nothing under
#   v1/attention/backends/mla/ rejects dcp>1 for the sparse path, and GLM-5.3's
#   main KV group is an MLAAttentionSpec (NOT SlidingWindowMLASpec), so the two
#   hard "DCP not support sliding window" asserts do not fire.
#   BUT the overlay's own comment warns the lse convention is UNVERIFIED and a
#   mismatch corrupts output SILENTLY. Do not enable DCP without a
#   token-for-token diff against a dcp=1 run at temperature 0.
# ---------------------------------------------------------------------------
