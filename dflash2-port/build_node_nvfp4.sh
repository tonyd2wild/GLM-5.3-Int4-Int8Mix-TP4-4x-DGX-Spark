#!/usr/bin/env bash
# Build vllm-glm52-b12x:nvfp4-dflash2 on this node.
#   base image  : vllm-node-tf5-glm52-b12x:nvfp4-v1   (NVFP4 KV-cache plumbing)
#   donor image : keys-vllm-glm53:b12x-dflash2-v1     (harvest dflash2 only)
# Variant of build_node.sh -- identical except BASE/TAG and a separate
# Dockerfile.nvfp4 so the original build_node.sh Dockerfile is left untouched.
set -euo pipefail

CTX="$HOME/dflash2-port"
V=/usr/local/lib/python3.12/dist-packages/vllm
BASE=vllm-node-tf5-glm52-b12x:nvfp4-v1
DONOR=keys-vllm-glm53:b12x-dflash2-v1
TAG=vllm-glm52-b12x:nvfp4-dflash2

echo "== $(hostname): checking source images =="
docker image inspect "$BASE"  >/dev/null
docker image inspect "$DONOR" >/dev/null

echo "== harvesting DFlash2 files from the donor =="
rm -rf "$CTX/donor"
mkdir -p "$CTX/donor/dflash2"
cid=$(docker create "$DONOR")
docker cp "$cid:$V/model_executor/models/qwen3_dflash2.py" "$CTX/donor/qwen3_dflash2.py"
docker cp "$cid:$V/v1/worker/gpu/spec_decode/dflash2/__init__.py" "$CTX/donor/dflash2/__init__.py"
docker cp "$cid:$V/v1/worker/gpu/spec_decode/dflash2/speculator.py" "$CTX/donor/dflash2/speculator.py"
docker rm "$cid" >/dev/null
md5sum "$CTX/donor/qwen3_dflash2.py" "$CTX/donor/dflash2/"*.py

cat > "$CTX/Dockerfile.nvfp4" <<DOCKERFILE
FROM ${BASE}

# DFlash2 (vLLM PR #52816) ported onto the NVFP4-KV GLM-5.3 743B b12x image.
COPY donor/qwen3_dflash2.py \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py
COPY donor/dflash2/ \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash2/
COPY patches/ /opt/dflash2-port/

RUN python3 /opt/dflash2-port/patch_base_dflash2.py \
 && python3 /opt/dflash2-port/patch_base_kv_dsa.py

LABEL dflash2.port="PR#52816 onto 0.23.1rc1.dev190+gab6660699 (b12x nvfp4-v1)"
DOCKERFILE

echo "== docker build $TAG =="
rm -rf "$CTX/patches/__pycache__"
docker build --network=none -f "$CTX/Dockerfile.nvfp4" -t "$TAG" "$CTX"

echo "== built =="
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep '^vllm-glm52-b12x:nvfp4-dflash2'
