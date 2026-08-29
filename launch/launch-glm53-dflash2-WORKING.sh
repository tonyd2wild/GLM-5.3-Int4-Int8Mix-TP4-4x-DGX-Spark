#!/usr/bin/env bash
#
# GLM-5.3 (big, 743B) Int4-Int8Mix on 4x DGX Spark (GB10/sm121), TP4.
#
# Derived from tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s launch.sh.
# GLM-5.3 and GLM-5.2 are structurally IDENTICAL (verified: 56 config keys each,
# only moe_router_dtype vs name_or_path differ; 78 layers / 6144 hidden /
# kv_lora 512 / qk_rope 64 / index_topk 2048 / 256 experts all match), and both
# resolve to GlmMoeDsaForCausalLM. So the proven GLM-5.2 recipe transfers with
# only the weights path and served name changed.
#
# Deltas vs the GLM-5.2 original, and WHY:
#   * gpu-memory-utilization stays 0.91 as in the original. It only works while
#     page cache is held down: on GB10 unified memory, host page cache eats
#     CUDA-visible free memory 1:1 (measured: 37 GiB cached -> CUDA free
#     74.25/121.69, max gmu 0.61). cache_flusher.sh MUST be running on every
#     node during load, or 0.91 trips the startup admission check. This is what
#     buys the 200,064-token pool instead of ~74k.
#   * weights: local on the head (Reddie), NFS at /mnt/reddie-models on workers
#     (nodes 3/4 have ~150G free and cannot hold a 378 GB local copy).
#   * SPEC_MODE arg: stage 1 runs BARE (no speculative decoding) so that a
#     coherence failure has exactly one meaning -- the quant is wrong.
#
# usage: launch-glm53-big-qt.sh <rank 0-3> [none|mtp|dflash]
set -uo pipefail

NODE_RANK="${1:?usage: launch-glm53-big-qt.sh <0|1|2|3> [none|mtp|dflash]}"
SPEC_MODE="${2:-none}"

IMAGE="vllm-glm52-b12x:dflash2-port2"
NAME="vllm_glm53big"
PORT=8000
MASTER_PORT=29541
HEAD_IP="192.168.192.2"          # Reddie = rank 0, weights are local here
KERNELS_DIR="$HOME/glm-triton"

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.2; HEADLESS=0; WEIGHTS=/var/tmp/models/GLM-5.3-Int4-Int8Mix ;;
  1) HOST_IP=192.168.192.4; HEADLESS=1; WEIGHTS=/mnt/reddie-models/GLM-5.3-Int4-Int8Mix ;;
  2) HOST_IP=192.168.192.3; HEADLESS=1; WEIGHTS=/mnt/reddie-models/GLM-5.3-Int4-Int8Mix ;;
  3) HOST_IP=192.168.192.1; HEADLESS=1; WEIGHTS=/mnt/reddie-models/GLM-5.3-Int4-Int8Mix ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

test -f "$WEIGHTS/config.json" || { echo "weights not visible at $WEIGHTS" >&2; exit 3; }

# --- kernel preflight (repo issue #5): the 10 sm12x overlays must be present
# and deepseek_v2.py <-> sparse_attn_indexer.py must be a MATCHED pair.
KERNEL_FILES=(sparse_mla_kernels.py sparse_mla_env.py sm12x_sparse_mla_attn.py
  patch_flashmla_ops.py flashmla_sparse.py sm12x_deep_gemm_fallbacks.py
  sm12x_mqa.py b12x_sparse_helpers.py sparse_attn_indexer.py deepseek_v2.py)
for f in "${KERNEL_FILES[@]}"; do
  [ -f "$KERNELS_DIR/$f" ] || { echo "kernel overlay missing: $KERNELS_DIR/$f" >&2; exit 4; }
done
if grep -q "fused_indexer_q_rope_quant" "$KERNELS_DIR/deepseek_v2.py" 2>/dev/null \
   && ! grep -Eq "def[[:space:]]+fused_indexer_q_rope_quant" "$KERNELS_DIR/sparse_attn_indexer.py" 2>/dev/null; then
  echo "kernel mismatch (issue #5): version-skewed overlays" >&2; exit 5
fi
grep -q "GlmMoeDsaForCausalLM" "$KERNELS_DIR/deepseek_v2.py" || {
  echo "overlay deepseek_v2.py does not define GlmMoeDsaForCausalLM" >&2; exit 6; }

MLA="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla"
OPS="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/deepseek_v4_ops"
LAYERS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers"
MODELS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"

case "$SPEC_MODE" in
  none)   SPEC=() ;;
  mtp)    SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_tensor_parallel_size":1,"attention_backend":"FLASHMLA_SPARSE"}') ;;
  dflash) SPEC=(--speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7,"draft_tensor_parallel_size":1}') ;;
  *) echo "spec must be none|mtp|dflash" >&2; exit 2 ;;
esac

# stage 1 (bare) keeps concurrency low; the MTP-overhang patch is only needed at
# --max-num-seqs >= 3 WITH MTP enabled.
if [ "$SPEC_MODE" = "none" ]; then MAXSEQS=4; else MAXSEQS=6; fi

DRAFT_DIR="$(dirname "$WEIGHTS")/GLM-5.3-DFlash2-draft"
test -f "$DRAFT_DIR/config.json" || { echo "drafter missing at $DRAFT_DIR" >&2; exit 7; }

docker rm -f "$NAME" 2>/dev/null

docker run -d --name "$NAME" \
  --restart no \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --network host --ipc host --shm-size 10gb --gpus all \
  --device /dev/infiniband:/dev/infiniband \
  -v /var/tmp/models:/cache/huggingface \
  -v "$WEIGHTS:/models/glm-5.3:ro" \
  -v "$DRAFT_DIR:/models/dflash2-draft:ro" \
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
  -e TRITON_CACHE_DIR=/cache/huggingface/.tritoncache \
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
    --served-model-name glm-5.3 --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-prefix-caching \
    --async-scheduling \
    "${SPEC[@]}" \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --max-model-len 80000 --max-num-seqs "$MAXSEQS" --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.91 --kv-cache-memory-bytes 10950000000 \
    --kv-cache-dtype fp8_ds_mla --kv-cache-dtype-skip-layers sliding_window \
    --distributed-executor-backend mp --compilation-config '{"cudagraph_mode":"FULL"}' \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \
    $( [ "$HEADLESS" = 1 ] && echo --headless )

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP spec=$SPEC_MODE weights=$WEIGHTS"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited immediately; docker logs $NAME" >&2; exit 1; }
