## ⭐ New (2026-10-06): two DCP4 lanes, up to a 689,772-token KV pool with DFlash2

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="runs/2026-10-06-dcp4-dflash2/charts/suite-dark.svg">
  <img alt="Single-stream decode by prompt type, NVFP4 lane vs fp8 lane: <!--ALT:suite-->" src="runs/2026-10-06-dcp4-dflash2/charts/suite-light.svg" width="880">
</picture>

<!--TABLE:single-->

Two new lanes, both TP4 + **decode context parallelism (DCP4)** + DFlash2 k=7 on the same four Sparks. DCP
shards the target model's KV cache across the four ranks instead of keeping a copy on each, so the same
memory holds four times the context. The DCP + DFlash2 patch set is **[Allan Clark (@ajclark)](https://github.com/ajclark)'s**
([issue #4](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/4),
[his repo](https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark), vendored in
[`dcp/`](dcp/) under his Apache-2.0 license). Lane 2 combines it with this repo's NVFP4 KV port, which nobody
had run together before.

| lane | KV cache | context | KV pool (vLLM) | KV pool (measured) | KV per rank | launcher |
|---|---|---|---|---|---|---|
| **Lane 2: DCP4 + NVFP4 + DFlash2** (serving default) | `nvfp4_ds_mla`, 400 B/token | 262,144 | **689,772** (2.63×) | <!--L2:measured--> | 7 GB | [`launch-glm53big-dcp4-nvfp4.sh`](launch/launch-glm53big-dcp4-nvfp4.sh) |
| Lane 2 at 512K | `nvfp4_ds_mla` | <!--L2X:ctx--> | <!--L2X:pool--> | <!--L2X:measured--> | <!--L2X:kv--> | same, `MAXLEN=<!--L2X:maxlen--> KVBYTES=<!--L2X:kvbytes-->` |
| Lane 1: DCP4 + fp8 + DFlash2 | `fp8_ds_mla`, 656 B/token | 262,144 | 462,308 (1.76×) | ~433,900 | 7 GB | [`launch-glm53big-dcp4.sh`](launch/launch-glm53big-dcp4.sh) |
| previous best: NVFP4 + DFlash2, no DCP | `nvfp4_ds_mla` | 270,000 | 293,447 | not measured | 10.95 GB | [`launch-glm53-nvfp4-dflash2.sh`](launch/launch-glm53-nvfp4-dflash2.sh) |

**What it costs: prefill.** DCP4 prefill runs at roughly 340-415 tok/s at every prompt length we tried (5K to
490K tokens). On the same 15,662-token prompt the DCP1 reference answered in 29 s and DCP4 in 45 s, and a cold
100K-token prompt takes about four and a half minutes. Decode is barely touched on this switched fabric: count-to-100 end to end is <!--L2:c100e2e--> tok/s against
53.3 for the old DCP1 fp8 lane (about -5%; the old figure is from this repo's earlier harness), where ajclark
measured about -22% on his switchless ring. Prefix caching hides most of the prefill cost for agents that resend
the same context.

### Concurrency, prefill and long context

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="runs/2026-10-06-dcp4-dflash2/charts/sweep-dark.svg">
  <img alt="Aggregate throughput C1-C6 with mixed real prompts, peak of 3 rounds: <!--ALT:sweep-->" src="runs/2026-10-06-dcp4-dflash2/charts/sweep-light.svg" width="880">
</picture>

<!--TABLE:sweep-->

<!--TABLE:longctx-->

### Is the NVFP4 + DCP merge correct?

Two of ajclark's DCP files had to be merged by hand with the NVFP4 port (`flashmla_sparse.py`, 5 conflicts;
`b12x_sparse_helpers.py`, 2), two merged cleanly (`mla_attention.py`, `kv_cache_interface.py`), and one DCP
guard that whitelisted only `fp8_ds_mla` was extended to `nvfp4_ds_mla` for the reason its own comment gives
(inline scales, kernel-side dequantization, bf16 query). Notes: [`dcp/glm-dcp-nv/MERGE-NOTES.md`](dcp/glm-dcp-nv/MERGE-NOTES.md).

The test was planned as a temperature-0 token-for-token match against the same stack at DCP1. **That test
cannot pass on this stack for any lane:** the DCP1 reference run twice on the same prompts matched itself on
only 3 of 8 (the MoE kernels use atomic adds, `VLLM_MARLIN_USE_ATOMIC_ADD=1`, so summation order differs run to
run, and speculative decoding changes the batch shapes). So the gate scores the same token sequences on both
lanes (teacher forcing, prefill only, no sampling or drafting) and compares every per-token log-probability:

| | completion tokens: mean / p99 \|Δ logprob\| | prompt tokens: mean / p99 | long 20K prompt, completion mean |
|---|---|---|---|
| DCP1 reference vs itself (noise floor) | 0.0220 / 0.487 | 0.2530 / 1.127 | 0.0065 |
| **DCP4 + NVFP4 vs reference** | **0.0232 / 0.551** | **0.2564 / 1.138** | **0.0030** |

DCP4 sits inside the reference's own run-to-run noise, and on the longest prompt, where the cross-rank merge
does the most work, closer to the reference than the reference is to itself. Greedy outputs: 4/8 identical
(the reference vs itself: 3/8), including both long-context prompts (300 tokens identical at 15,662 tokens of
context); every divergence is at a near-tie. Both buried codes were retrieved and count-to-100 was 100/100.
Scorer: [`bench/tf_score.py`](runs/2026-10-06-dcp4-dflash2/bench/tf_score.py).

### Findings worth knowing

- **vLLM's KV-pool figure is close here, unlike on GLM-5.3-Flash.** One 204,901-token request held 47.2% of the
  fp8 lane's blocks: measured capacity ~433,900 tokens against the reported 462,308 (6.5% optimistic).
  <!--L2:kvfinding--> On the Flash model the same check found 2× ([Flash repo PR #12](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/pull/12)).
  The `cache_config_info` metric reports a third number (312,035 on lane 1); use the boot log or measure.
- **Not deterministic at temperature 0**, even without DCP (above). Compare lanes with teacher forcing, not text.
- **This vLLM streams thinking as `reasoning`, not `reasoning_content`.** A harness that only watches
  `reasoning_content` measures time-to-first-*answer* and divides thinking tokens by the answer window:
  math read 50 tok/s instead of 36. The harness here counts both and reports time to first token and time to
  answer separately.
- **Filler text tokenizes 3.5× denser on the 743B tokenizer** than on Flash's (`w123`-style words); size
  synthetic prompts with `/tokenize`, not word counts.
- **`--max-num-batched-tokens 2048` under DCP** (ajclark's validated value; the DCP-gathered fp32 accumulator
  is 4× per token). 4096 is untested and is the first thing to try for faster prefill.

### Method

Speed-night harness ([`bench/`](runs/2026-10-06-dcp4-dflash2/bench/)), temperature 0, `reasoning_effort: low`,
median of 3, GPUs verified first (85.7-90.1 TFLOPS bf16 on all four, `gputest.sh`). The endpoint served live
agents during the runs, so every measurement waited for an idle server, sampled `running + waiting` while it
ran, and was retried when other requests appeared; dirty cells were re-run (`rerun_dirty.py`). Raw JSON, logs
and per-lane summaries: [`runs/2026-10-06-dcp4-dflash2/`](runs/2026-10-06-dcp4-dflash2/).

### Run it

```bash
# every node: the DCP overlay set and the patched paged-MQA kernel (see dcp/README.md)
cp -r dcp/glm-dcp ~/glm-dcp && cp -r dcp/glm-dcp-nv ~/glm-dcp-nv
cp -a ~/glm-triton ~/glm-triton-aj && cp dcp/glm-triton-aj/sm12x_mqa.py ~/glm-triton-aj/
# ranks 1, 2, 3 first, then 0 (rank 0 serves :8000)
./launch/launch-glm53big-dcp4-nvfp4.sh <rank>                                  # lane 2, 262K
MAXLEN=<!--L2X:maxlen--> KVBYTES=<!--L2X:kvbytes--> ./launch/launch-glm53big-dcp4-nvfp4.sh <rank>   # lane 2 at 512K
./launch/launch-glm53big-dcp4.sh <rank>                                        # lane 1 (fp8 image)
```

Lane 2 needs `vllm-glm52-b12x:nvfp4-dflash2-p2` ([`dflash2-port/build_node_nvfp4.sh`](dflash2-port/build_node_nvfp4.sh)),
lane 1 `vllm-glm52-b12x:dflash2-port2`. Run the cache flusher during boot, as for every lane here.

---
