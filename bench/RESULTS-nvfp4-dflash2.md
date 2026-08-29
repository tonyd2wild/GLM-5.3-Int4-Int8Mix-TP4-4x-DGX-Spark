# NVFP4 KV + DFlash2 — GLM-5.3 743B Int4-Int8Mix, TP4 on 4x DGX Spark

Measured 2026-08-29. **End-to-end** tok/s (wall clock, request send → full
response received), not the engine's internal decode rate.

This is the best-of-both lane: NVFP4's 400 B/token KV record **and** the DFlash2
block-diffusion drafter, which previously could not be combined because the
NVFP4 image shipped without DFlash2 support.

```
image      vllm-glm52-b12x:nvfp4-dflash2-p2
overlay    /var/tmp/glm-triton-nvfp4
kv         nvfp4_ds_mla --kv-cache-dtype-skip-layers sliding_window
spec       dflash k=7
ctx        270,000        KV pool  293,447 tokens
endpoint   :8211          served model name  glm-5.3-nvfp4
```

## Results

| lane | count100 | C1 | C2 | C3 | C4 | C5 | C6 | KV pool | ctx |
|---|---|---|---|---|---|---|---|---|---|
| **NVFP4 + DFlash2** | **51.03** | 17.81 | 28.16 | 34.86 | 38.72 | 43.11 | 50.98 | **293,447** | **270K** |
| NVFP4 + MTP-5 | 25.37 | 17.78 | 26.29 | 34.07 | 40.34 | 46.20 | 53.31 | 317,278 | 300K |
| fp8 + DFlash2 (k=7) | 53.32 | 18.70 | 27.34 | 35.37 | 40.96 | 45.96 | 49.88 | 179,479 | 80K |
| fp8 + MTP-4 | 26.91 | 19.62 | 26.17 | 32.54 | 45.97 | 52.07 | 51.93 | 200,064 | 200K |

count-to-100: **100/100 lines correct, 95.6% acceptance**, 3.92 s wall.

**The headline:** NVFP4 + DFlash2 keeps essentially all of fp8+DFlash2's
structured-output speed (51.03 vs 53.32, −4%) while carrying a **63% larger KV
pool** (293,447 vs 179,479) and **3.4x the context** (270K vs 80K). Sweep
acceptance ~21–25%, in line with the other DFlash2 lanes.

## How it was built

`vllm-node-tf5-glm52-b12x:nvfp4-v1` and `vllm-node-tf5-glm52-b12x:probe-modded`
are the **same vLLM commit** (`0.23.1rc1.dev190+gab6660699`), differing only in
the NVFP4 plumbing. So the existing DFlash2 port applies unchanged — all nine
files the patches touch are byte-identical between the two bases, zero overlap
with the NVFP4 surface. Re-pointing the port's `BASE=` gave a clean build:
25 anchored edits, 0 anchor failures, 13/13 verification checks pass, and every
NVFP4 file byte-identical to the base afterwards.

Then bake the SWA-under-MLA assert fix on top (`:nvfp4-dflash2-p2`) — the
`Attention.get_kv_cache_spec` assert that rejects the drafter's sliding-window
layers under an MLA target. See [`patches/patch_swa_under_mla.sh`](../patches/patch_swa_under_mla.sh).

## Two constraints worth knowing

**1. DCP is impossible in this combination.** The drafter's `SlidingWindowSpec`
layers trip a hard vLLM assert:

```
kv_cache_interface.py:528
  assert decode_context_parallel_size == 1, "DCP not support sliding window."
```

No image or KV dtype changes this. DCP requires MTP, which is exactly why the
reference 655K lane used MTP k=3. So the three-way NVFP4 + DFlash2 + DCP4 is not
reachable; pick DFlash2 (speed) or DCP (pool), not both.

**2. Context must leave room for the drafter group.** At 300K the boot fails with
`11.06 GiB KV cache is needed ... larger than available (10.2 GiB)` — the drafter
group costs roughly 8% of the pool versus the MTP lane. 270K fits with headroom.
