# GLM-5.3-Int4-Int8Mix — TP4 on 4× DGX Spark

**The first Int4-Int8Mix quantization of big GLM-5.3 (743B), quantized and served on four
NVIDIA DGX Spark (GB10 / sm121 / aarch64).**

No quantization of the large GLM-5.3 existed anywhere when this was made. This repo is the
recipe: how the quant was produced, how it is verified, and how it is served at TP4 across
four 121 GB unified-memory boxes.

> ### 📦 Weights
> **[`2wild4tv/GLM-5.3-Int4-Int8Mix`](https://huggingface.co/2wild4tv/GLM-5.3-Int4-Int8Mix)** — 377.4 GiB, 282 shards.
>
> The weights are **not** in this repository (GitHub cannot hold 378 GB). This repo holds the
> quantization script, the verification gates, and the serving recipe.

> **Status: stage 1 complete.** The quant is finished, structurally verified, and proven to
> generate coherent output at TP4. Speculative decoding and CUDA graphs are being layered on
> now; this README grows one stage at a time. Numbers below are labelled with exactly what
> was and was not enabled when they were measured.

---

## What it is

| | |
|---|---|
| Base | [`zai-org/GLM-5.3-BF16`](https://huggingface.co/zai-org/GLM-5.3-BF16) — 1507 GB, 282 shards, genuine BF16 |
| Architecture | `GlmMoeDsaForCausalLM` (`model_type: glm_moe_dsa`) — served by vLLM's `deepseek_v2` path |
| Size | 743B total / ~40B active MoE |
| Layers | 78 + MTP head at layer 78; `first_k_dense_replace: 3` → 3 dense + 75 sparse |
| Dims | hidden 6144, `kv_lora_rank` 512, `qk_nope` 192, `qk_rope` 64, `index_topk` 2048 |
| Experts | 256 routed + 1 shared, 8 active per token |
| Quantized output | **377.4 GiB**, 282 shards, `compressed-tensors` / `pack-quantized` |
| Per-rank at TP4 | **95.53 GiB** (98.07 GiB with MTP loaded) |

**Why a BF16 base and not the fp8 repo:** the main `zai-org/GLM-5.3` repo already ships fp8
(`quant_method: fp8`, e4m3, block 128×128). The QuantTrio recipe deliberately keeps the MoE
router, the DSA indexer, the LM head and layer 0 at **full precision** — from a BF16 base
those stay true BF16; from an fp8 base they would be fp8, degrading exactly the layers that
matter most for accuracy.

---

## The quantization recipe

Config groups and ignore list are taken **verbatim** from
[`QuantTrio/GLM-5.2-Int4-Int8Mix`](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix).

That transfer is not an assumption — GLM-5.2 and GLM-5.3 were diffed and are structurally
identical: **56 top-level config keys each**, differing only in `moe_router_dtype` (5.3) vs
`name_or_path` (5.2). Layers, hidden size, `kv_lora_rank`, `qk_rope_head_dim`, `index_topk`,
expert count and `first_k_dense_replace` all match, and both resolve to
`GlmMoeDsaForCausalLM`.

Data-free RTN — **no calibration dataset**, static, symmetric, weight-only, group size 128.

| group | targets | bits | strategy | modules matched |
|---|---|---|---|---|
| `w4a16_experts` | layers 3–77 `mlp.experts.N.{gate,up,down}_proj` | 4 | group / 128 | **57,600** |
| `w8a16_linears` | layers 1–77 attention projections + dense & shared-expert MLP | 8 | group / 128 | **616** |
| `w8a16_mtp_channel` | layer 78 (MTP) attention + MLP + experts | 8 | **channel** (gs −1) | **776** |

Counts reconcile exactly — 75 sparse layers × 256 experts × 3 projections = 57,600.

### Kept at full precision

This list is what protects accuracy. Do not shorten it.

- `model.layers.0.*` — the entire first layer
- every `mlp.gate` — the **MoE router**; quantizing it destroys expert routing
- `self_attn.indexer` and `self_attn.indexers_proj` — the DSA sparse-attention selector
- MTP `eh_proj`, `enorm`, `hnorm`, `shared_head.norm`, `shared_head.head`
- `lm_head`

`kv_cache_scheme: None` — KV precision is a serve-time choice (`--kv-cache-dtype`), not baked
into the weights.

---

## How it was quantized — shard streaming, not `oneshot`

`llmcompressor.oneshot` with accelerate disk-offload is **not usable here**: offloading a
1507 GB model needs ~1.4 TB of scratch, and with the BF16 base (1507 GB) and the output
(378 GB) both on the fleet, no node has that free.

[`quant/glm53_quant_stream.py`](quant/glm53_quant_stream.py) instead streams the checkpoint:
read one BF16 shard → quantize its targeted tensors → pack → write one output shard. Peak RAM
is roughly one shard in plus one shard out (~10 GiB), and the 1:1 shard mapping makes the job
**resumable** — a completed output shard is skipped on restart.

It uses `compressed-tensors`' own `calculate_qparams`, `quantize` and `pack_to_int32` rather
than reimplementing the bit packing, which removes an entire class of silent-corruption bugs.

One detail worth copying: the scale is **rounded to bf16 before quantizing**, so the scale
persisted to disk is exactly the one quantization assumed. Compute in fp32, round, then
quantize with the rounded value — otherwise dequantization at serve time uses a slightly
different scale than the quantizer did.

**Runtime: 28.2 minutes** for 377.4 GiB on a single DGX Spark, CPU-only.

### Verify before you quantize, and again after

Two gates, both fail-closed:

- [`quant/dryrun_layermap.py`](quant/dryrun_layermap.py) — compiles the group regexes against
  the **actual** tensor names in the base index and prints per-group match counts before a
  single tensor is touched. A regex that silently matches nothing produces a quant that looks
  fine and is wrong.
  **Match module names, not tensor names.** The group patterns are anchored (`...o_proj$`)
  and describe *modules*, while `weight_map` keys are *tensors* (`...o_proj.weight`). Compare
  them directly and every group reports zero matches.
- [`quant/verify_quant.py`](quant/verify_quant.py) — after the run: tensor accounting, per-group
  counts, sacred-module checks, shard completeness, dtype spot-checks, config validation.

Results on this quant:

```
source tensors        59,585
quantized modules     58,992
expected out tensors  177,569
actual   out tensors  177,569        exact
group counts          57,600 / 616 / 776    matches the pre-quant dry-run exactly
sacred modules        eh_proj, mlp.gate, layer 0, lm_head -> plain BF16, 0 packed leaks
indexer               0 quantized
shards                282, none missing, declared == on-disk 377.4 GiB
```

On-disk layout matches QuantTrio byte-for-byte (checked against their published shards by
range-reading the safetensors headers):

| tensor | QuantTrio | this quant |
|---|---|---|
| expert `down_proj.weight_packed` | I32 `[6144, 256]` | I32 `[6144, 256]` |
| expert `down_proj.weight_scale` | BF16 `[6144, 16]` | BF16 `[6144, 16]` |
| MTP `down_proj.weight_scale` | BF16 `[6144, 1]` | BF16 `[6144, 1]` |

Round-trip error: int8 group/128 ≈ **0.70%** relative, int4 group/128 ≈ **12%** (normal for 16
levels); 256 / 16 distinct levels — full range used.

---

## Serving

[`launch/launch-glm53-tp4.sh`](launch/launch-glm53-tp4.sh) — one `docker run` per node, vLLM's
**native** multi-node (`--nnodes/--node-rank/--master-addr`), **no Ray**. Workers start
headless, head last.

It is derived from
[`tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s`](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s),
whose 10 sm12x Triton kernel overlays and modded image are **required** on GB10 — stock vLLM
kernels fault on sm121. Follow that repo for the image build and kernel staging; this repo
changes only the weights path and the served name.

```bash
./launch/launch-glm53-tp4.sh <rank 0-3> [none|mtp|dflash]
```

### GB10 unified memory — the thing that will bite you

CPU and GPU share one 121 GB pool, so **host page cache consumes CUDA-visible memory 1:1**.
Measured on this fleet: with 37 GiB of page cache, CUDA reported only **74.25 / 121.69 GiB**
free — a hard ceiling of `gpu-memory-utilization 0.61`. After dropping caches, 105–113 GiB.

Consequences, all learned the hard way:

1. **Run [`launch/cache_flusher.sh`](launch/cache_flusher.sh) on every node** for the whole
   boot. It drops caches unconditionally every 60 s. A *conditional* flusher (only when cache
   exceeds a threshold) can sit there and never fire while still leaving you short.
2. **Pin `--kv-cache-memory-bytes`.** Sizing KV off "currently free" memory means the same
   command boots or OOMs depending on what the page cache happened to look like at profiling
   time. Pinning it makes boots deterministic.
   Caveat: pinning it also makes vLLM **skip memory profiling entirely** — it then never
   validates that what remains is enough for activations, NCCL buffers and graphs.
3. **Set `vm.swappiness` to 10.** At the default 0–1 the box will wedge with 16 GB of swap
   completely untouched.

### Steady state is meant to be tight

MemAvailable ~0.7 GB with ~3.2 GB of swap parked is the **designed** operating point for this
config class, not a warning sign.

---

## Results

### Stage 1 — bare (no speculative decoding, no CUDA graphs)

```
TP4, vLLM v0.23.1rc1.dev190+gab6660699, --kv-cache-dtype fp8_ds_mla
--gpu-memory-utilization 0.91  --kv-cache-memory-bytes 10950000000
--max-model-len 200000  --max-num-seqs 4  --compilation-config '{"cudagraph_mode":"NONE"}'

weights 95.53 GiB/rank      GPU KV cache: 202,944 tokens @ 200K context
```

**These are a floor, not a headline number.** Thinking was ON (no reasoning parser), and both
speculative decoding and CUDA graphs were disabled.

| concurrency | agg tok/s | per-stream | mean latency s |
|---|---|---|---|
| 1 | 4.68 | 4.68 | 54.67 |
| 2 | 9.22 | 4.61 | 55.29 |
| 3 | 13.95 | 4.65 | 54.94 |
| 4 | **17.68** | 4.42 | 57.79 |
| 5 | 11.53 | 2.31 | 68.75 |
| 6 | 14.89 | 2.48 | 68.20 |

Near-linear scaling through c4 (per-stream holds 4.4–4.7). The c5/c6 regression is
`--max-num-seqs 4` queueing the extra requests — a config artifact, not a model limit.

**Coherence: PASS.** The bat-and-ball problem — whose trap answer is $0.10 — was answered
correctly with working:

> **$0.05** — Ball = x, bat = x + $1.00, so x + (x + 1.00) = 1.10 → 2x = 0.10 → x = 0.05

A model whose router gate or DSA indexer had been quantized cannot do that. This is the test
the bare stage exists for: with no drafter in the loop, a coherence failure has exactly one
meaning.

### Stage 2 — MTP k=4 + CUDA graphs

Same everything, plus in-checkpoint MTP (layer 78) and graphs re-enabled:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":4,
                       "draft_tensor_parallel_size":1,"attention_backend":"FLASHMLA_SPARSE"}'
--max-num-seqs 6  --compilation-config '{"cudagraph_mode":"FULL"}'
--reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice

weights 98.07 GiB/rank      GPU KV cache: 200,064 tokens @ 200K context
```

Both figures land **exactly** on the GLM-5.2 reference's published numbers (98.07 GiB/node,
200,064 tokens), which is the clearest confirmation that the recipe transferred correctly.
The +2.54 GiB over the bare stage is the MTP head.

Thinking still ON for this measurement, so it is directly comparable to stage 1.

| concurrency | stage 1 (bare) | **stage 2 (MTP + graphs)** | gain |
|---|---|---|---|
| 1 | 4.68 | **8.34** | 1.78× |
| 2 | 9.22 | **18.50** | 2.01× |
| 3 | 13.95 | **24.01** | 1.72× |
| 4 | 17.68 | **30.41** | 1.72× |
| 5 | 11.53 | **34.13** | 2.96× |
| 6 | 14.89 | **36.90** | 2.48× |

**36.90 tok/s aggregate at c6.** For reference, the GLM-5.2 QuantTrio recipe on this same
four-node hardware reports 32.5 mean / 36 peak.

Two caveats that make this an understatement rather than a headline:

- **Thinking was ON**, so a large share of those tokens are reasoning traces rather than
  answer text.
- **One of the four GPUs was running at 29% clock** (see below). In TP4 every rank waits on
  the slowest, so this number was set with a quarter of the cluster hobbled.

#### `cudagraph_mode: FULL` still escalates — MTP does not prevent it

Worth recording because it contradicts a reasonable guess. `FULL` escalates to
`FULL_AND_PIECEWISE` regardless of speculative decoding:

```
CUDAGraphMode.FULL is not supported with FlashMLASparseBackend
(support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_AND_PIECEWISE
```

It is a property of the attention backend, not of batch uniformity. The bare stage died
here; this stage survived the same escalation. The difference was **not** MTP — it was
`vm.swappiness=10` (giving the 16 GB of swap a chance to act as a cushion; at swappiness 1
the box wedged with swap 100% untouched) and an unconditional page-cache flusher. NVRM still
logged `NV_ERR_NO_MEMORY` twice during capture on this run; it recovered rather than taking
the node down.

#### One node running at a third clock

Under identical load, sampled across all four ranks:

| node | SM clock | power | util |
|---|---|---|---|
| node1 | 2476 MHz | 26.3 W | 96% |
| **node2 (head)** | **721 MHz** | **8.9 W** | **95%** |
| node3 | 2496 MHz | 24.6 W | 96% |
| node4 | 2522 MHz | 24.6 W | 96% |

The head is fully utilized and doing the work — at 29% of the clock and 35% of the power of
its siblings. It is **not** a workload artifact: at complete idle (0% util) the other three
sit at ~2410 MHz while the head sits at 890 MHz. All four report identical `P0`, identical
2418 MHz applications clock, persistence enabled, and `clocks_throttle_reasons.active = 0x0`.
`nvidia-smi -lgc 2418,3003` is **accepted and silently ignored** — the clock does not move.

The one thing distinguishing that node: it hard-crashed and rebooted earlier in the session
during the NVRM OOM described below. A degraded post-crash power state is the leading
hypothesis; a clean reboot is the only remaining lever, since no software control has any
effect.

If you are reproducing this, check per-node SM clocks under load before trusting any
throughput number. One quiet rank at a third clock gates the entire tensor-parallel group.

---

## Notes for anyone reproducing this

**Stage your bring-up.** Serve bare first, then add speculative decoding, then swap in a
different drafter. If the quant and the drafter go in together and the output is garbage, you
cannot tell which one is wrong, and vLLM's own diagnostic ("garbage means the router or
indexer got quantized") stops being valid the moment a drafter is in the loop.

**But know that bare mode is not automatically the safe one.** On this stack, bare + CUDA
graphs is *worse* than MTP + CUDA graphs: `FlashMLASparseBackend` only advertises
`UNIFORM_BATCH` support, so `cudagraph_mode: FULL` silently escalates to `FULL_AND_PIECEWISE`
— **two** graph sets instead of one. That extra allocation is what took a node down here.
Speculative decoding produces uniform batches and keeps it to a single set. Watch for this
line:

```
CUDAGraphMode.FULL is not supported with FlashMLASparseBackend
(support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_AND_PIECEWISE
```

**`enable_thinking` does not exist in GLM-5.3's stock chat template.** Passing
`chat_template_kwargs: {"enable_thinking": false}` per request, or
`--default-chat-template-kwargs`, silently does nothing — there is no such variable to set.
The template ends with an unconditional open-thinking tag:

```jinja
{%- if add_generation_prompt -%}
    <|assistant|>{{- '<think>' -}}
{%- endif -%}
```

(`clear_thinking`, which the template *does* define, only controls whether prior reasoning
stays in history.) [`patches/patch_chat_template_thinking.py`](patches/patch_chat_template_thinking.py)
adds the variable, emitting a pre-closed `<think></think>` when thinking is off. Then
`--default-chat-template-kwargs '{"enable_thinking": false}'` works — single-quote the JSON
or argparse rejects it.

Note `--reasoning-parser glm45` is a *different* thing: it routes reasoning into a separate
`reasoning_content` field so it stops polluting `content`, but the model still spends tokens
thinking. Use both.

**When a GB10 box OOMs, the process table will lie to you.** NVRM holds its memory outside
all kernel memory accounting — in the failure here, ~114 GiB was unaccounted while the vLLM
worker's RSS read 3.66 GiB. Because `oom_score` is computed from RSS, the OOM killer ranked a
3.66 GiB Python process below its own audio daemons and reaped those instead. Read
`/proc/buddyinfo` (zero free blocks of order ≥ 8 is the real signal) rather than trusting
`free`.

---

## Credits

- **[QuantTrio](https://huggingface.co/QuantTrio)** — the Int4-Int8Mix recipe. The
  `config_groups` and `ignore` list here are theirs, taken verbatim from
  `QuantTrio/GLM-5.2-Int4-Int8Mix`.
- **[zai-org](https://huggingface.co/zai-org)** — GLM-5.3 and the BF16 release.
- **[`tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s`](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s)**
  — the GB10 serving recipe, sm12x kernel overlays and modded image this launcher derives
  from, which in turn derives from
  [CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10).
- **vLLM** — `compressed-tensors`, and the packing routines this quantizer calls rather than
  reimplements.

## License

Apache-2.0 for the code in this repository. The model weights follow the licenses of the
upstream base model and the referenced recipes.
