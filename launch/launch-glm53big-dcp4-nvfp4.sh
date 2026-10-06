#!/usr/bin/env bash
#
# GLM-5.3 (743B) Int4-Int8Mix on 4x DGX Spark: TP4 + DCP4 + NVFP4 KV + DFlash2 k=7, on :8000.
# Lane 2: ajclark's DCP overlay set merged with the NVFP4 KV port (~/glm-dcp-nv, see dcp/glm-dcp-nv/MERGE-NOTES.md).
# DCP_SIZE=1 MAXLEN=131072 gives the same stack without DCP: the reference for the token-for-token gate.
#
# usage: launch-glm53big-dcp4-nvfp4.sh <rank 0-3>      (start ranks 1, 2, 3, then 0)
#
# CREDITS
#   DCP + DFlash2 placement patch set, the indexer bounds fix (issue #6) and the sm12x_mqa
#   de-specialisation fix: ajclark, github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark
#   (Apache-2.0), files staged unmodified from its stage/glm-dcp and stage/glm-triton (SHA256SUMS
#   verified). This launcher is the "legacy" branch of its launch-glm53big-dcp.sh with our fleet's
#   values substituted. Quant, sm12x kernels and DFlash2 port: this repo. DFlash2 model: PR #52816
#   (vLLM); drafter incoai/GLM-5.3-DFlash2.
#
# Deltas from ajclark's launcher, all fleet-specific:
#   - switched 200G fabric: our NCCL block (rocep1s0f0, GID 3, 4 channels, NCCL 2.30.4 preload)
#     instead of the switchless-ring block; rank IPs 192.168.192.2/.4/.3/.1
#   - ranks 1-3 read weights and drafter over NFS from Reddie (/mnt/reddie-models)
#   - DCP 4 (theirs now defaults to 2), 262,144 context, 7 GB KV per rank: their issue #4 point
#   - NVMe KV tier off for this step (KVTIER), NCCL hot-plug off, no temperature override
#   - :8000 serves glm-5.3-flash and glm-5.3 (same endpoint as the Flash lane it replaces)
#   - reasoning_effort low by default (fleet convention); clients can override per request
set -uo pipefail

NODE_RANK="${1:?usage: launch-glm53big-dcp4-nvfp4.sh <0|1|2|3>}"

IMAGE="${DCP_IMAGE:-vllm-glm52-b12x:nvfp4-dflash2-p2}"
NAME="vllm_glm53big"
PORT=8000
MASTER_PORT=29541
FABRIC_IF=enp1s0f0np0
HEAD_IP="192.168.192.2"
KERNELS_DIR="${KERNELS_DIR:-$HOME/glm-triton-aj}"
DCP_DIR="${DCP_DIR:-$HOME/glm-dcp-nv}"
COMPILE_CACHE_DIR="/var/tmp/glm-compile-cache"

DCP_SIZE="${DCP_SIZE:-4}"
MAXLEN="${MAXLEN:-262144}"
DFLASH_K=7
MAXBATCHED="${MAXBATCHED:-2048}"
MAXSEQS="${MAXSEQS:-6}"
KVBYTES="${KVBYTES:-7000000000}"
EFFORT="${EFFORT:-low}"

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.2; HEADLESS=0; MROOT=/var/tmp/models ;;
  1) HOST_IP=192.168.192.4; HEADLESS=1; MROOT=/mnt/reddie-models ;;
  2) HOST_IP=192.168.192.3; HEADLESS=1; MROOT=/mnt/reddie-models ;;
  3) HOST_IP=192.168.192.1; HEADLESS=1; MROOT=/mnt/reddie-models ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac
WEIGHTS=$MROOT/GLM-5.3-Int4-Int8Mix
DRAFT_DIR=$MROOT/GLM-5.3-DFlash2-draft

test -f "$WEIGHTS/config.json" || { echo "weights not visible at $WEIGHTS" >&2; exit 3; }
grep -q DFlash2DraftModel "$DRAFT_DIR/config.json" 2>/dev/null || { echo "DFlash2 drafter missing at $DRAFT_DIR" >&2; exit 3; }
ip -o -4 addr show "$FABRIC_IF" | grep -q "$HOST_IP" || { echo "$FABRIC_IF does not hold $HOST_IP -- wrong rank?" >&2; exit 3; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE missing on this node" >&2; exit 3; }

KERNEL_FILES=(sparse_mla_kernels.py sparse_mla_env.py sm12x_sparse_mla_attn.py patch_flashmla_ops.py
  sm12x_deep_gemm_fallbacks.py sm12x_mqa.py deepseek_v2.py)
for f in "${KERNEL_FILES[@]}"; do [ -f "$KERNELS_DIR/$f" ] || { echo "kernel overlay missing: $KERNELS_DIR/$f" >&2; exit 4; }; done
DCP_FILES=(flashmla_sparse.py sparse_attn_indexer.py sparse_utils.py indexer.py mla_attention.py
  kv_cache_interface.py kv_cache_utils.py kv_cache_coordinator.py block_table.py gpu_input_batch.py
  gpu_model_runner.py cp_utils.py flash_attn.py scheduler.py b12x_sparse_helpers.py adaptive.py
  model_runner.py cudagraph_utils.py v2_block_table.py v2_async_utils.py v1_outputs.py
  confidence_trace.py v2_rejection_sampler_utils.py v2_rejection_sampler.py v2_sample_states.py)
for f in "${DCP_FILES[@]}"; do [ -f "$DCP_DIR/$f" ] || { echo "DCP overlay missing: $DCP_DIR/$f" >&2; exit 4; }; done
grep -q "triton_filter_and_convert_dcp_index" "$DCP_DIR/sparse_utils.py" || { echo "sparse_utils.py is not the DCP version" >&2; exit 5; }
grep -q "_merge_dcp_topk_global" "$DCP_DIR/sparse_attn_indexer.py" || { echo "sparse_attn_indexer.py is not the DCP version" >&2; exit 5; }
grep -q "DCP overlay: hybrid-aware" "$DCP_DIR/scheduler.py" || { echo "scheduler.py is not the DCP version" >&2; exit 5; }
grep -q "cp_world_size_for_kv_cache_spec" "$DCP_DIR/kv_cache_interface.py" || { echo "kv_cache_interface.py is not the DCP version" >&2; exit 5; }
grep -q "GlmMoeDsaForCausalLM" "$KERNELS_DIR/deepseek_v2.py" || { echo "deepseek_v2.py lacks GlmMoeDsaForCausalLM" >&2; exit 6; }
grep -q "nvfp4_ds_mla" "$DCP_DIR/flashmla_sparse.py" && grep -q "store_nvfp4_glm_kv" "$DCP_DIR/flashmla_sparse.py" || { echo "$DCP_DIR/flashmla_sparse.py is not the DCP+NVFP4 merge" >&2; exit 8; }
docker run --rm --entrypoint test "$IMAGE" -f /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/nvfp4_glm_kernels.py || { echo "$IMAGE lacks nvfp4_glm_kernels.py" >&2; exit 9; }

mkdir -p "$COMPILE_CACHE_DIR"
VLLM="/usr/local/lib/python3.12/dist-packages/vllm"
MLA="$VLLM/v1/attention/backends/mla"
OPS="$VLLM/v1/attention/ops/deepseek_v4_ops"
LAYERS="$VLLM/model_executor/layers"
MODELS="$VLLM/model_executor/models"

docker rm -f "$NAME" 2>/dev/null
docker run -d --name "$NAME" --restart no \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  --network host --ipc host --shm-size 10gb --gpus all \
  --device /dev/infiniband:/dev/infiniband \
  -v /var/tmp/models:/cache/huggingface \
  -v "$WEIGHTS:/models/glm-5.3:ro" \
  -v "$DRAFT_DIR:/models/dflash2-draft:ro" \
  -v "$COMPILE_CACHE_DIR:/compile-cache" \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$KERNELS_DIR/sparse_mla_kernels.py:$MLA/sparse_mla_kernels.py:ro" \
  -v "$KERNELS_DIR/sparse_mla_env.py:$MLA/sparse_mla_env.py:ro" \
  -v "$KERNELS_DIR/sm12x_sparse_mla_attn.py:$MLA/sm12x_sparse_mla_attn.py:ro" \
  -v "$KERNELS_DIR/patch_flashmla_ops.py:$MLA/patch_flashmla_ops.py:ro" \
  -v "$KERNELS_DIR/sm12x_deep_gemm_fallbacks.py:$MLA/sm12x_deep_gemm_fallbacks.py:ro" \
  -v "$KERNELS_DIR/sm12x_mqa.py:$OPS/sm12x_mqa.py:ro" \
  -v "$KERNELS_DIR/deepseek_v2.py:$MODELS/deepseek_v2.py:ro" \
  -v "$DCP_DIR/b12x_sparse_helpers.py:$OPS/b12x_sparse_helpers.py:ro" \
  -v "$DCP_DIR/flashmla_sparse.py:$MLA/flashmla_sparse.py:ro" \
  -v "$DCP_DIR/sparse_utils.py:$MLA/sparse_utils.py:ro" \
  -v "$DCP_DIR/indexer.py:$MLA/indexer.py:ro" \
  -v "$DCP_DIR/sparse_attn_indexer.py:$LAYERS/sparse_attn_indexer.py:ro" \
  -v "$DCP_DIR/mla_attention.py:$LAYERS/attention/mla_attention.py:ro" \
  -v "$DCP_DIR/kv_cache_interface.py:$VLLM/v1/kv_cache_interface.py:ro" \
  -v "$DCP_DIR/kv_cache_utils.py:$VLLM/v1/core/kv_cache_utils.py:ro" \
  -v "$DCP_DIR/kv_cache_coordinator.py:$VLLM/v1/core/kv_cache_coordinator.py:ro" \
  -v "$DCP_DIR/block_table.py:$VLLM/v1/worker/block_table.py:ro" \
  -v "$DCP_DIR/gpu_input_batch.py:$VLLM/v1/worker/gpu_input_batch.py:ro" \
  -v "$DCP_DIR/gpu_model_runner.py:$VLLM/v1/worker/gpu_model_runner.py:ro" \
  -v "$DCP_DIR/cp_utils.py:$VLLM/v1/worker/cp_utils.py:ro" \
  -v "$DCP_DIR/flash_attn.py:$VLLM/v1/attention/backends/flash_attn.py:ro" \
  -v "$DCP_DIR/scheduler.py:$VLLM/v1/core/sched/scheduler.py:ro" \
  -v "$DCP_DIR/adaptive.py:$VLLM/v1/spec_decode/adaptive.py:ro" \
  -v "$DCP_DIR/model_runner.py:$VLLM/v1/worker/gpu/model_runner.py:ro" \
  -v "$DCP_DIR/v2_block_table.py:$VLLM/v1/worker/gpu/block_table.py:ro" \
  -v "$DCP_DIR/v2_async_utils.py:$VLLM/v1/worker/gpu/async_utils.py:ro" \
  -v "$DCP_DIR/v1_outputs.py:$VLLM/v1/outputs.py:ro" \
  -v "$DCP_DIR/confidence_trace.py:$VLLM/v1/spec_decode/confidence_trace.py:ro" \
  -v "$DCP_DIR/cudagraph_utils.py:$VLLM/v1/worker/gpu/cudagraph_utils.py:ro" \
  -v "$DCP_DIR/v2_rejection_sampler_utils.py:$VLLM/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:ro" \
  -v "$DCP_DIR/v2_rejection_sampler.py:$VLLM/v1/worker/gpu/spec_decode/rejection_sampler.py:ro" \
  -v "$DCP_DIR/v2_sample_states.py:$VLLM/v1/worker/gpu/sample/states.py:ro" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_DO_NOT_TRACK=1 -e DO_NOT_TRACK=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e LD_PRELOAD=/cache/huggingface/hub/nccl-2.30.4/libnccl.so.2 \
  -e HF_HOME=/cache/huggingface \
  -e TRITON_CACHE_DIR=/compile-cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/compile-cache/inductor \
  -e VLLM_CACHE_ROOT=/compile-cache/vllm \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e VLLM_DEBUG_WORKSPACE=1 \
  -e GLM_DCP_Q_PREGATHER=0 -e GLM_DCP_COMPACT=1 -e GLM_DCP_LSE_FOLD=0 -e GLM_DCP_RS_HEADMAJOR=0 \
  -e GLM_SPEC_POLICY=off -e GLM_SPEC_VERIFY_CAP=7 -e GLM_SPEC_CONFIDENCE_TRACE=0 -e GLM_SPEC_LOSSY=0 \
  -e GLM52_BIND_HOST_TRITON=1 -e GLM52_MQA_LOGITS_TRITON=1 -e GLM52_PAGED_MQA_TRITON=1 \
  -e GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192 \
  -e GLM52_B12X_MLA=1 -e VLLM_DISABLE_FLASHINFER_AUTOTUNE=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 \
  -e NCCL_SOCKET_IFNAME=$FABRIC_IF -e GLOO_SOCKET_IFNAME=$FABRIC_IF \
  -e NCCL_IB_GID_INDEX=3 -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 \
  -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e NODE_RANK="$NODE_RANK" -e MASTER_ADDR="$HEAD_IP" -e VLLM_HOST_IP="$HOST_IP" \
  "$IMAGE" \
  vllm serve /models/glm-5.3 \
    --served-model-name glm-5.3-flash glm-5.3 --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --default-chat-template-kwargs "{\"reasoning_effort\": \"$EFFORT\"}" \
    --enable-prefix-caching --async-scheduling \
    --speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":'"$DFLASH_K"',"draft_tensor_parallel_size":1}' \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend ag_rs \
    --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --max-num-batched-tokens "$MAXBATCHED" \
    --long-prefill-token-threshold 2048 \
    --gpu-memory-utilization 0.91 --kv-cache-memory-bytes "$KVBYTES" \
    --kv-cache-dtype nvfp4_ds_mla --kv-cache-dtype-skip-layers sliding_window \
    --distributed-executor-backend mp --compilation-config '{"cudagraph_mode":"FULL"}' \
    --nnodes 4 --node-rank "$NODE_RANK" --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \
    $( [ "$HEADLESS" = 1 ] && echo --headless )

echo "launched $NAME rank=$NODE_RANK tp4 dcp$DCP_SIZE nvfp4 dflash k=$DFLASH_K maxlen=$MAXLEN kv=$KVBYTES seqs=$MAXSEQS"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited immediately; docker logs $NAME" >&2; exit 1; }
