# GLM-5.3 743B Int4-Int8Mix — TP4 on 4x DGX Spark (GB10 / sm121)

Measured 2026-08-29. All figures are **end-to-end** tok/s: wall clock from HTTP
request send to full response received, `completion_tokens / wall`. This is NOT
the engine's internal decode rate — it includes queueing, prefill, sampling and
detokenization, so it is lower than the decode number vLLM logs.

Hardware: 4x NVIDIA DGX Spark (GB10, sm121, 48 SMs each), TP4 over InfiniBand.
Weights: GLM-5.3 743B quantized to Int4-Int8Mix (compressed-tensors pack-quantized).
KV cache: `fp8_ds_mla`, sliding-window layers exempted (`--kv-cache-dtype-skip-layers`).

## First bench: count to 100

Prompt: `Count from 1 to 100. One number per line. Nothing else.` (temperature 0)

| lane | ctx | KV pool | wall | out tok | **e2e tok/s** | acceptance | correct |
|---|---|---|---|---|---|---|---|
| **DFlash2 (k=7)** | 80K | 179,479 | 3.92 s | 200 | **51.03** | 95.6% | 100/100 |
| MTP-4 | 200K | 200,064 | 7.43 s | 200 | 26.91 | 97.0% | 100/100 |

**DFlash2 is 1.90x faster than MTP-4 on structured output.**

## C1-C6 concurrency sweep

Prompt: a free-form technical explanation, 256 tokens each, temperature 0.
AGG = aggregate throughput across all concurrent streams.

| concurrency | DFlash2 AGG | DFlash2 accept | MTP-4 AGG | MTP-4 accept |
|---|---|---|---|---|
| C1 | 19.24 | 23.8% | 19.62 | 37.4% |
| C2 | 27.60 | 23.0% | 26.17 | 36.7% |
| C3 | 35.39 | 23.7% | 32.54 | 40.5% |
| C4 | 43.97 | 24.1% | 45.97 | 38.1% |
| C5 | 46.83 | 23.5% | 52.07 | 39.0% |
| C6 | 50.28 | 23.0% | 51.93 | 39.4% |

## Reading these numbers honestly

DFlash2's advantage is **strongly workload-dependent**:

- On structured / low-entropy output (counting, and by extension code, lists,
  JSON) the block-diffusion drafter hits 95.6% acceptance and nearly doubles
  throughput.
- On free-form prose acceptance falls to ~23%, and DFlash2 is roughly a wash
  with MTP-4 — better at C2/C3, slightly worse at C5/C6.

**Known tuning gap.** The prior GLM-5.3-Flash DFlash2 deployment recorded 40-53%
acceptance on mixed prompts. The 23% here suggests the aux hidden-state layer
selection is not tuned for the 743B model: it uses `deepseek_v2.py`'s stock
Eagle3 aux layers rather than a GLM-specific choice. This degrades silently —
it costs speed, never correctness. Tuning the aux tap layers is the obvious next
step and should recover a chunk of the prose case.

## Correctness

Speculative decoding is distribution-preserving, so at temperature 0 a correct
drafter must reproduce the target's tokens exactly. It does — byte-identical:

```
prompt : "The capital of France is"
MTP-4  : ' Paris. Distance from London to Paris is 343 km, while straight line distance is 344 km. Direct'
DFlash2: ' Paris. Distance from London to Paris is 343 km, while straight line distance is 344 km. Direct'
```

Both lanes emit 100/100 correct lines for count-to-100.

**Not yet run:** the 69-scenario quality eval. No claim of quality parity with
the BF16 base is made here.
