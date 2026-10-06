# DFlash2 port — how the working image was built

## The problem this solves

Two docker images were in play, and each had exactly half of what we needed:

| | `keys-vllm-glm53:b12x-dflash2-v1` | `vllm-node-tf5-glm52-b12x:probe-modded` |
|---|---|---|
| vLLM | `0.1.dev20051+g487ecf187` | `0.23.1rc1.dev190+gab6660699` |
| DFlash2 support | **yes** | no |
| Correct output on the 743B model | **NO — digit soup** | **yes** |

The DFlash2 image was built and proven for GLM-5.3-**Flash**, not the 743B model.
On the big model it emits prompt-independent garbage (`'1.0.0.5.0.0...'` for every
prompt, including `"The capital of France is"`).

**Root cause**, established by control experiment: the garbage persists with the
drafter *completely removed*, so it is not a DFlash2 problem at all. That image's
`B12X_MLA_SPARSE` backend has none of the sm12x Triton kernel overlays from
`~/glm-triton` (`sm12x_mqa.py`, `b12x_sparse_helpers.py`, `sparse_mla_kernels.py`,
...) nor the `GLM52_*_TRITON` env switches that the working lane sets. Since GLM
sparse MLA routes 100% of prompt tokens through `forward_mqa` on this hardware,
the paged-MQA kernel *is* the prompt attention — and unpatched it is wrong on
GB10's 48-SM parts.

Attempting to fix that image failed across five boots (forcing MQA, forcing MHA,
removing the FlashInfer sampler, disabling autotune, mounting the kpool indexer
patch). The auto backend is not an escape either: without the glm-triton kernels
it does not spin up at all (96% GPU utilization at 10 W — a spin, not compute).

**So the fix is the other direction: port DFlash2 into the image whose kernels
are proven.** That turned out to be much smaller than expected, because the big
model's `deepseek_v2.py` in the good image *already* carries `SupportsEagle3` and
aux-hidden-state capture — so the GLM aux-capture patch is not needed at all.

## What the port does

Base: `vllm-node-tf5-glm52-b12x:probe-modded` → tag `vllm-glm52-b12x:dflash2-port`

Copied verbatim from the donor image:
- `model_executor/models/qwen3_dflash2.py`
- `v1/worker/gpu/spec_decode/dflash2/{__init__,speculator}.py`

The donor's `dflash/` v1 package is deliberately **not** copied — its
`create_forward_fn` returns `fwd` where the base returns `(fwd, attn_state)`.

Then `patch_base_dflash2.py` applies the PR #52816 wiring, re-anchored for the
older base (the base's DFlash v1 is ~7 weeks older and lacked much of what the
donor's patch assumed present):
1. `registry.py` — `DFlash2DraftModel` entry
2. `spec_decode/__init__.py` — route DFlash2 drafts to the DFlash2 speculator
3. `qwen3_dflash.py` — 10 sub-edits; notably the base's `DFlashQwen3Attention`
   had **no sliding-window support at all**, and its decoder layer took no
   `layer_idx`. Both are load-bearing: the real drafter declares 6x
   `sliding_attention` with `sliding_window: 2048`.
4. `speculator.py` — `draft_logits_spec` hook
5. `config/vllm.py` — force the V2 model runner for a DFlash2 draft
6. `logits_processor.py` — `get_top_k_tokens`

Plus two edits not in the donor script: a local `tl_rand32` (the base's
`gumbel.py` has only `tl_rand64`, and is shared with the production MTP path so
it must not be touched), and routing `get_dflash_causal` through
`dflash_has_any_non_causal` so the drafter's top-level `is_causal: false` is honoured.

`patch_base_kv_dsa.py` ports `_get_kv_cache_groups_dsa_drafter`. Without it the
`{78 MLA + 78 DSA-indexer + 6 SlidingWindowSpec}` mix misses every fast path and
falls into `unify_kv_cache_spec_page_size`, which raises on the indexer's
132 B/token page. Design: target group **first** (id stays 0), drafter SWA layers
in their own group appended **last**.

> **Trap:** never set `page_size_padded` on the drafter group. Padding routes the
> runner into a strided view and FlashInfer's int kernel block sizes split the
> manager block x36, each charged a full page stride — 13.59 GB demanded from a
> 377 MB tensor. The patch strips it and asserts `None`.

## Two further fixes found at boot time

- **`patches/patch_swa_under_mla.sh`** — `Attention.get_kv_cache_spec` asserts
  `not vllm_config.model_config.use_mla` whenever a layer has a sliding window.
  That is a *model-level* flag gating a *per-layer* decision; the method lives on
  the non-MLA `Attention` class, so a sliding-window layer reaching it is
  genuinely non-MLA — it is the drafter's. The assert predates draft models
  registering SWA layers under an MLA target. Baked into tag `:dflash2-port2`.
- **`--kv-cache-dtype-skip-layers sliding_window`** — `fp8_ds_mla` is an MLA-only
  packed layout; without the exemption every non-MLA backend refuses the
  drafter's layers with `No valid attention backend found ... use_mla=False`.

## Build and verify

**Files and layout.** The build expects `~/dflash2-port/patches/` to hold the four scripts in this
directory (`patch_base_dflash2.py`, `patch_base_kv_dsa.py`, `verify_dflash2.py`, and
**`dsa_block.py`**). `dsa_block.py` was missing from the first publication of this repo, so
`patch_base_kv_dsa.py` could not run for anyone else; it is restored here from the original
2026-08-29 build tree (identical on all four build nodes, sha256 `22ed5b16fbe5c53d...`). Thanks to
[@omanuke](https://github.com/omanuke) for reporting it and to [@ajclark](https://github.com/ajclark)
for publishing a reconstruction that reproduced the same KV geometry (179,479-token pool) in the
meantime ([issue #1](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/1)).

```bash
mkdir -p ~/dflash2-port/patches && cp dflash2-port/*.py ~/dflash2-port/patches/
bash dflash2-port/build_node.sh            # fp8 lane image  -> vllm-glm52-b12x:dflash2-port
bash dflash2-port/build_node_nvfp4.sh      # NVFP4 lane image -> vllm-glm52-b12x:nvfp4-dflash2
docker run --rm -v $HOME/dflash2-port/patches:/vp:ro \
  --entrypoint python3 vllm-glm52-b12x:dflash2-port /vp/verify_dflash2.py
```
13 checks, including resolving the real on-disk drafter `config.json` through the
registry and confirming the KV grouping yields 2 groups with the drafter last
and unpadded.

## Known gap — do not enable probabilistic drafting

`DFlash2Speculator`'s kernels use `sample_idx_mapping < 0` as the padded-row
sentinel; this base pads with **0**, the donor with **-1**. On a padded CUDA-graph
batch, padded rows would be attributed to request 0. Not triggered by default
(`draft_sample_method` defaults to `greedy`, and the affected path only runs when
`probabilistic`). Fixing it properly also requires `is_valid_req` masks in
`gumbel_block_argmax`, which is shared with the production MTP-4 sampling path —
so it was left alone deliberately.

## Drafter weights

[`incoai/GLM-5.3-DFlash2`](https://huggingface.co/incoai/GLM-5.3-DFlash2): the 743B model's
DFlash2 drafter (4.58 GiB, `DFlash2DraftModel`, hidden 6144, six `sliding_attention` layers,
`target_layer_ids [5, 19, 33, 47, 61, 75]`). Its `config.json` is byte-identical to the one these
lanes run (sha256 `f59e1da17d41d24a...`). Not `incoai/GLM-5.3-Flash-DFlash2`, which is the
320B Flash model's drafter. Launchers expect it at `<models>/GLM-5.3-DFlash2-draft`.

## Without the donor image

`keys-vllm-glm53:b12x-dflash2-v1` is not published. The three DFlash2 files it supplies are
upstream since vLLM PR #52816. [@omanuke](https://github.com/omanuke) found the revision whose
anchors match this port: take them at vLLM **`660a446fc93ebb780144585b3fdc34ed96fa755c`**, not the
merge commit (`2b4f36d0` refactored `speculator.py` to import `gumbel_noised_argmax`, so the
`tl_rand32` anchor no longer matches):

```
vllm/model_executor/models/qwen3_dflash2.py
vllm/v1/worker/gpu/spec_decode/dflash2/__init__.py
vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py
```

On that route `verify_dflash2.py` reports 12/13: check 6 imports both `tl_rand32` and
`gumbel_noised_argmax`, and no single upstream revision of `speculator.py` has both. That check is
stricter than the build, not a build failure.
