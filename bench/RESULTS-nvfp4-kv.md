# NVFP4 KV cache — GLM-5.3 743B Int4-Int8Mix, TP4 on 4x DGX Spark

Measured 2026-08-29. **End-to-end** tok/s (wall clock, request send → full
response received), not the engine's internal decode rate.

```
image      vllm-node-tf5-glm52-b12x:nvfp4-v1
overlay    /var/tmp/glm-triton-nvfp4          (NOT ~/glm-triton -- see below)
kv dtype   nvfp4_ds_mla                       (400 B/token vs fp8_ds_mla's 656)
ctx        300,000        KV pool  317,278 tokens
spec       MTP k=5,  cudagraph FULL [6,12,18,24,30,36], gmu 0.91, kv pin 10.95 GB
endpoint   :8211          served model name  glm-5.3-nvfp4
```

## Results

| lane | count100 | C1 | C2 | C3 | C4 | C5 | C6 | KV pool | ctx |
|---|---|---|---|---|---|---|---|---|---|
| **NVFP4 + MTP-5** | 25.37 | 17.78 | 26.29 | 34.07 | 40.34 | 46.20 | **53.31** | **317,278** | **300K** |
| fp8 + DFlash2 (k=7) | **53.32** | 18.70 | 27.34 | 35.37 | 40.96 | 45.96 | 49.88 | 179,479 | 80K |
| fp8 + MTP-4 | 26.91 | 19.62 | 26.17 | 32.54 | 45.97 | 52.07 | 51.93 | 200,064 | 200K |

count-to-100: 100/100 lines correct, **93.7% MTP acceptance**, 7.88 s wall.
Sweep acceptance 29.5–33.1% (higher than fp8+DFlash2's ~23%).

**Trade:** NVFP4 buys **+77% KV pool** (317,278 vs 179,479) and **3.75x context**
(300K vs 80K) for ~6% single-stream throughput versus fp8+MTP-4, and it posts the
best C6 aggregate of any lane measured. Structured-output single-stream still
belongs to fp8+DFlash2 (53.32), which is a drafter effect, not a KV effect.

## How `nvfp4_ds_mla` is enabled — the non-obvious part

There are **two** kernel overlay directories on these nodes, and only one enables NVFP4:

| | `/var/tmp/glm-triton` (== `~/glm-triton`) | `/var/tmp/glm-triton-nvfp4` |
|---|---|---|
| lane | fp8_ds_mla | **nvfp4_ds_mla** |
| files | 10 | 10 |

Only **2 of 10 files differ**: `flashmla_sparse.py` and `b12x_sparse_helpers.py`.
The other 8 are byte-identical. The decisive line is in the overlay's
`flashmla_sparse.py`:

```python
supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
    "auto", "bfloat16", "fp8_ds_mla",
    "nvfp4_ds_mla",          # <-- present ONLY in the -nvfp4 overlay
    "fp8",                   # alias for fp8_ds_mla
]
```

The image's own copy stops at `"fp8_ds_mla", "fp8"`. Same overlay also supplies
`get_kv_cache_shape → (num_blocks, block_size, 400)`, a `do_kv_cache_update()`
calling `store_nvfp4_glm_kv`, and the `gather_dequant_nvfp4_glm` path.
`patch_flashmla_ops.py` is identical in both — it is **not** the mechanism.

Pointing the launcher at `~/glm-triton` while asking for `nvfp4_ds_mla` fails
backend selection. The overlay directory is the whole switch.

## Two traps that cost boots

1. **Unpinned `cudagraph_mode: FULL` breaks the chat path.** With
   `{"cudagraph_mode":"FULL"}` and no `cudagraph_capture_sizes`, the engine died
   under the first concurrent chat request with
   `EngineCore ... KeyError: 'chatcmpl-<id>'` — request-tracking corruption, not
   an OOM. Pinning `[6,12,18,24,30,36]` (the proven GLM-5.2 set) fixed it.
   Raw `/v1/completions` was unaffected, which made it look like a chat-template
   bug at first. It is not.
2. **NVFP4 cannot coexist with another lane.** Both take `--gpus all` at
   gmu 0.91. A stale `vllm_glm53big` holding 113 GB produced
   `NCCL error: unhandled cuda error` 26 s into boot. Verify
   `nvidia-smi --query-compute-apps=pid` is EMPTY on all four nodes before launching.

Reference lane (GLM-5.2 744B): `tonyd2wild/GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s`,
which reported 317,312 tokens from the same 10.95 GB pin — GLM-5.3 lands at
317,278, within 34 tokens.
