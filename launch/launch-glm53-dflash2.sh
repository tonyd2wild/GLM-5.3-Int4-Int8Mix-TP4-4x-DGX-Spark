#!/usr/bin/env bash
set -uo pipefail
#
# GLM-5.3 (743B) Int4-Int8Mix + DFlash2 drafter, TP4 on 4x DGX Spark (GB10/sm121).
#
# IMAGE: keys-vllm-glm53:b12x-dflash2-v1  (vllm 0.1.dev20051+g487ecf187)
#   This is the ONLY image on the fleet with all four required capabilities:
#     - DFlash2DraftModel registered (registry.py:632 -> qwen3_dflash2.py)
#     - GlmMoeDsaForCausalLM       (registry.py:118 -> deepseek_v2.py:1931)
#     - GB10 kernels via the b12x package + b12x_mla_sparse.py
#       (supports_compute_capability gates on capability.major == 12)
#     - decode-context-parallel support (for a later DCP4 stage)
#   Built from ghcr.io/drowzeys/keys-vllm-glm53-flash-nvfp4-ablit:b12x-cu130 with
#   our DFlash2 patches baked in (patch_registry_and_select / glm_aux_capture /
#   glm5_drafter_group).
#
# DELIBERATELY NOT DONE HERE, and why:
#   * NO ~/glm-triton overlay bind-mounts. Those target the older dev17863 tree.
#     This image's deepseek_v2.py is NEWER (GlmMoeDsa at line 1931) and its
#     sparse_attn_indexer.py already defines fused_indexer_q_rope_quant natively.
#     Mounting the old set would clobber newer code, and the usual preflight
#     guard would PASS (the overlay set is self-consistent) then fail at runtime.
#   * NO GLM52_* env vars. Dead on this build; replaced by the VLLM_B12X_* family.
#   * NO deepseek_v4_ops mounts -- that path does not exist here, docker would
#     silently create an empty dir.
#
# SPEC CONFIG: method is "dflash", NOT "dflash2". v1 vs v2 is dispatched by the
# DRAFTER's architectures field (DFlash2DraftModel). num_speculative_tokens 7 =
# drafter block_size 8 minus the bonus token.
#
# CONTEXT: 80K. Our measured KV is 54.7 KB/token at fp8_ds_mla. If vLLM issue
# #41559 bites (DFlash needs non-causal attention; some backends reject fp8 KV
# there) we get forced to bf16 = ~109 KB/token. A 9 GB pin covers 80K EITHER WAY,
# so the boot cannot fail on KV sizing whichever dtype is accepted.
#
# usage: launch-glm53-dflash2.sh <rank 0-3> [fp8|bf16]

NODE_RANK="${1:?usage: launch-glm53-dflash2.sh <0|1|2|3> [fp8|bf16]}"
KVMODE="${2:-fp8}"

IMAGE="keys-vllm-glm53:b12x-dflash2-v1"
NAME="vllm_glm53_dflash2"
PORT=8000
MASTER_PORT=29551
HEAD_IP="192.168.192.2"

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.2; HEADLESS=0; MDIR=/var/tmp/models ;;
  1) HOST_IP=192.168.192.4; HEADLESS=1; MDIR=/mnt/reddie-models ;;
  2) HOST_IP=192.168.192.3; HEADLESS=1; MDIR=/mnt/reddie-models ;;
  3) HOST_IP=192.168.192.1; HEADLESS=1; MDIR=/mnt/reddie-models ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

WEIGHTS="$MDIR/GLM-5.3-Int4-Int8Mix"
DRAFT="$MDIR/GLM-5.3-DFlash2-draft"

test -f "$WEIGHTS/config.json"       || { echo "weights not visible at $WEIGHTS" >&2; exit 3; }
test -f "$DRAFT/config.json"         || { echo "drafter not visible at $DRAFT" >&2; exit 4; }
grep -q DFlash2DraftModel "$DRAFT/config.json" || { echo "drafter is not DFlash2" >&2; exit 5; }

case "$KVMODE" in
  fp8)  KV_ARGS=(--kv-cache-dtype fp8_ds_mla) ;;
  bf16) KV_ARGS=() ;;   # let it default to auto/bf16
  *) echo "kv mode must be fp8|bf16" >&2; exit 2 ;;
esac

docker rm -f "$NAME" 2>/dev/null

docker run -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 10gb --gpus all \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$WEIGHTS:/models/glm-5.3:ro" \
  -v "$DRAFT:/models/dflash2-draft:ro" \
  -v /var/tmp/glm53-dflash2-cache:/cache \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/cache/tritoncache \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 \
  -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  "$IMAGE" \
    /models/glm-5.3 \
    --served-model-name glm-5.3 \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --quantization compressed-tensors \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7}' \
    --max-model-len 80000 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.91 \
    --kv-cache-memory-bytes 9000000000 \
    "${KV_ARGS[@]}" \
    --enable-prefix-caching \
    --async-scheduling \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --distributed-executor-backend mp \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \
    $( [ "$HEADLESS" = 1 ] && echo --headless )

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP kv=$KVMODE weights=$WEIGHTS"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited immediately; docker logs $NAME" >&2; exit 1; }
