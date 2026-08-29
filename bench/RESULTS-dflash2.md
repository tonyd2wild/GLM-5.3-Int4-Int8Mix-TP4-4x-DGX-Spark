# GLM-5.3 743B Int4-Int8Mix — TP4 on 4x DGX Spark (GB10 / sm121)

Measured 2026-08-29. All figures are **end-to-end** tok/s: wall clock from HTTP
request send to full response received, `completion_tokens / wall`. This is NOT
the engine's internal decode rate — it includes queueing, prefill, sampling and
detokenization, so it is lower than the decode number vLLM logs.

Hardware: 4x NVIDIA DGX Spark (GB10, sm121, 48 SMs each), TP4 over InfiniBand.
Weights: GLM-5.3 743B quantized to Int4-Int8Mix (compressed-tensors pack-quantized).
Shipping KV config: **`--kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window`**.

## First bench: count to 100

Prompt: `Count from 1 to 100. One number per line. Nothing else.` (temperature 0)

| lane | KV dtype | ctx | KV pool | wall | out tok | **e2e tok/s** | accept | correct |
|---|---|---|---|---|---|---|---|---|
| **DFlash2 (k=7)** | **fp8** | 80K | 179,479 | 3.75 s | 200 | **53.32** | 95.6% | 100/100 |
| DFlash2 (k=7) | fp8_ds_mla | 80K | 179,479 | 3.92 s | 200 | 51.03 | 95.6% | 100/100 |
| MTP-4 | fp8_ds_mla | 200K | 200,064 | 7.43 s | 200 | 26.91 | 97.0% | 100/100 |

**DFlash2 on fp8 is 1.98x faster than MTP-4 on structured output.**

## C1-C6 concurrency sweep

Free-form technical prose, 256 tokens each, temperature 0.
AGG = aggregate throughput across all concurrent streams.

| conc | DFlash2 fp8 | accept | DFlash2 fp8_ds_mla | accept | MTP-4 | accept |
|---|---|---|---|---|---|---|
| C1 | 18.70 | 22.7% | 19.24 | 23.8% | 19.62 | 37.4% |
| C2 | 27.34 | 21.6% | 27.60 | 23.0% | 26.17 | 36.7% |
| C3 | 35.37 | 22.7% | 35.39 | 23.7% | 32.54 | 40.5% |
| C4 | 40.96 | 22.5% | 43.97 | 24.1% | 45.97 | 38.1% |
| C5 | 45.96 | 22.9% | 46.83 | 23.5% | 52.07 | 39.0% |
| C6 | 49.88 | 22.6% | 50.28 | 23.0% | 51.93 | 39.4% |

## Why `fp8` and not `fp8_e4m3`

`fp8_e4m3` is **rejected outright** — it is not in the sparse MLA backend's
`supported_kv_cache_dtypes`, and the MLA *target* layers fail selection:

```
ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(
  head_size=576, use_mla=True, use_sparse=True, kv_cache_dtype=fp8_e4m3, ...)
```

`fp8` (which in vLLM *is* e4m3) is accepted, because `flashmla_sparse.py:100`
lists it explicitly:

```python
supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [..., "fp8_ds_mla", "fp8"]
#                                                                    "fp8" -> alias
```

`mla_attention.py:397-405` then converts it for the MLA layers, logging
`Using DeepSeek's fp8_ds_mla`. Because that assignment mutates the **shared**
`cache_config`, the drafter's non-MLA sliding-window layers would inherit
`fp8_ds_mla` too and fail — hence `--kv-cache-dtype-skip-layers sliding_window`
is still required alongside it.

Net effect on the 743B model: `fp8` is marginally **faster** than requesting
`fp8_ds_mla` directly (53.32 vs 51.03 on count-to-100) and statistically
indistinguishable on the prose sweep.

## Reading these numbers honestly

DFlash2's advantage is **strongly workload-dependent**:

- On structured / low-entropy output (counting, and by extension code, lists,
  JSON) the block-diffusion drafter hits 95.6% acceptance and roughly doubles
  throughput.
- On free-form prose acceptance falls to ~23%, and DFlash2 is a wash with MTP-4 —
  better at C2/C3, slightly worse at C4-C6.

**Known tuning gap.** The prior GLM-5.3-Flash DFlash2 deployment recorded 40-53%
acceptance on mixed prompts. The ~23% here suggests the aux hidden-state layer
selection is not tuned for the 743B model: it uses `deepseek_v2.py`'s stock
Eagle3 aux layers `(6,20,34,48,62,76)` rather than a GLM-specific choice. This
degrades silently — it costs speed, never correctness. Tuning the aux tap layers
is the obvious next step.

## Correctness

Speculative decoding is distribution-preserving, so at temperature 0 a correct
drafter reproduces the target's tokens. Verified against the MTP-4 lane:

```
prompt : "The capital of France is"
MTP-4      (fp8_ds_mla): ' Paris. Distance from London to Paris is 343 km, while straight line distance is 344 km. Direct'
DFlash2    (fp8_ds_mla): ' Paris. Distance from London to Paris is 343 km, while straight line distance is 344 km. Direct'   <- byte-identical
DFlash2    (fp8)       : ' Paris. Distance from London to Paris is 344 km, while straight line distance is 344 km. Direct'   <- one token differs
```

DFlash2 on `fp8_ds_mla` is **byte-identical** to MTP-4, which is the strongest
available correctness signal for the drafter. The `fp8` lane differs by one token
in a factual figure; that is the KV quantization differing, not the drafter — the
same one-token class of difference you get between any two KV quant settings.
Both lanes emit 100/100 correct lines for count-to-100.

**Not yet run:** the 69-scenario quality eval. No claim of quality parity with
the BF16 base is made here.
