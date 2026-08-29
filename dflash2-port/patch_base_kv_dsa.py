#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DSA-DRAFTER-GROUP for the target base (vLLM 0.23.1rc1.dev190+gab6660699).

Port of ~/patches/kv_cache_utils.py's `_get_kv_cache_groups_dsa_drafter` /
`_dsa_drafter_tensor_layout` (written against the donor tree,
0.1.dev20051+g487ecf187) onto the target base's kv_cache_utils.py.

Why it is needed here too: the DFlash2 drafter registers plain
`SlidingWindowSpec` layers. In the target base's `get_kv_cache_groups`, a
GLM-5.3 spec dict of {MLAAttentionSpec target + DSA-indexer MLA spec +
6 plain SlidingWindowSpec drafter layers} matches NONE of the fast paths --
`is_kv_cache_spec_uniform` (mixed types), `UniformTypeKVCacheSpecs.from_specs`
(mixed types), `group_and_unify_kv_cache_specs` (needs SlidingWindowMLASpec) --
and falls through to `unify_kv_cache_spec_page_size` +
`_get_kv_cache_groups_uniform_page_size`, which raises on the indexer's
132 B/token page.

Adaptations vs the donor-tree original:
  * `vllm_config.max_in_flight_tokens` does not exist in this tree; the
    drafter's admission bound is taken from
    `scheduler_config.max_num_batched_tokens`, which is exactly what this
    tree's own `SlidingWindowSpec.max_memory_usage_bytes` passes.
  * `KpoolTailSpec` does not exist in this tree; the `type(x) is
    SlidingWindowSpec` exact tests already exclude every subclass, so no
    import or extra guard is needed.
  * This tree has no `_use_packed_kv_cache_config` / GLM-5-Next branch; the
    DSA branches are ordered ahead of this tree's DeepseekV4 equivalents
    (`_get_kv_cache_config_deepseek_v4` and the `all(...
    UniformTypeKVCacheSpecs)` accounting branch), which is the same relative
    ordering the original required and for the same reason.

CRITICAL invariant carried over unchanged: the drafter group NEVER carries
`page_size_padded` (a padded spec routes the runner into a strided view whose
memory demand explodes), and the target group is the very object the
drafterless dispatch would build, kept FIRST so its group id stays 0.
"""

import ast
import os
import sys

VLLM_ROOT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/usr/local/lib/python3.12/dist-packages/vllm"
)
BLOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsa_block.py")

REL = "v1/core/kv_cache_utils.py"
path = os.path.join(VLLM_ROOT, REL)

with open(BLOCK_PATH, encoding="utf-8") as f:
    DSA_BLOCK = f.read().rstrip("\n")

with open(path, encoding="utf-8") as f:
    src = f.read()


def apply(marker: str, old: str, new: str) -> None:
    global src
    if marker in src:
        print(f"[skip] {REL}: already applied ({marker[:56]!r})")
        return
    assert old in src, f"ANCHOR NOT FOUND in {REL}:\n{old[:300]!r}"
    assert src.count(old) == 1, (
        f"ANCHOR NOT UNIQUE ({src.count(old)}x) in {REL}:\n{old[:300]!r}"
    )
    src = src.replace(old, new)
    print(f"[edit] {REL}: applied ({marker[:56]!r})")


# --------------------------------------------------------------------------
# A. Insert the DSA-DRAFTER-GROUP block. Placed just above
#    get_kv_cache_config_from_groups: every helper it calls
#    (is_kv_cache_spec_uniform, _get_kv_cache_groups_uniform_spec,
#    _get_kv_cache_groups_uniform_type) is defined above this point.
# --------------------------------------------------------------------------
_GKCFG = (
    "def get_kv_cache_config_from_groups(\n"
    "    vllm_config: VllmConfig,\n"
    "    kv_cache_groups: list[KVCacheGroupSpec],\n"
    "    available_memory: int,\n"
    ") -> KVCacheConfig:\n"
)
apply("def _get_kv_cache_groups_dsa_drafter(", _GKCFG, DSA_BLOCK + "\n\n\n" + _GKCFG)

# --------------------------------------------------------------------------
# B. _pool_bytes_per_block: the effective-capacity divisor must agree with the
#    tensor emission below, or num_gpu_blocks_override accounting drifts.
# --------------------------------------------------------------------------
apply(
    "DSA-DRAFTER-GROUP: mirrors the divisor",
    "    if all(\n"
    "        isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) for g in kv_cache_groups\n"
    "    ):\n"
    "        # buckets = {page_size: [[layer_names], [layer_names], ...]}\n"
    "        buckets = _bucket_layers_by_page_size(kv_cache_groups)\n"
    "        return sum(ps * len(slots) for ps, slots in buckets.items())\n",
    "    if (dsa := _dsa_drafter_tensor_layout(kv_cache_groups)) is not None:\n"
    "        # DSA-DRAFTER-GROUP: mirrors the divisor used by\n"
    "        # get_kv_cache_config_from_groups -- every target layer's own page plus\n"
    "        # one standalone page per DFlash2 drafter layer. Must precede the\n"
    "        # all-UniformTypeKVCacheSpecs bucket branch, which matches this group\n"
    "        # list too and would re-bucket it into the DeepseekV4 slot layout.\n"
    "        return dsa[4]\n"
    "    if all(\n"
    "        isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) for g in kv_cache_groups\n"
    "    ):\n"
    "        # buckets = {page_size: [[layer_names], [layer_names], ...]}\n"
    "        buckets = _bucket_layers_by_page_size(kv_cache_groups)\n"
    "        return sum(ps * len(slots) for ps, slots in buckets.items())\n",
)

# --------------------------------------------------------------------------
# C. get_kv_cache_config_from_groups: tensor emission.
# --------------------------------------------------------------------------
apply(
    "DSA-DRAFTER-GROUP: big GLM-5.3 target group",
    "    elif all(\n"
    "        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)\n"
    "        for group in kv_cache_groups\n"
    "    ):\n"
    "        # DeepseekV4: UniformTypeKVCacheSpecs but multiple groups.\n"
    "        # Delegate to the DeepseekV4-specific allocator.\n",
    "    elif (dsa := _dsa_drafter_tensor_layout(kv_cache_groups)) is not None:\n"
    "        # DSA-DRAFTER-GROUP: big GLM-5.3 target group (per-layer pages, exactly\n"
    "        # as in the single-group drafterless case above) plus compact\n"
    "        # STANDALONE per-layer tensors for the DFlash2 drafter. Must precede\n"
    "        # the DeepseekV4 branch, whose test is True for ANY all-\n"
    "        # UniformTypeKVCacheSpecs group list -- including this one -- and would\n"
    "        # re-lay-out every tensor into the shared slot form, breaking both the\n"
    "        # MLA per-layer sizing and the drafter's contiguous view.\n"
    "        target_names, target_page_by_name, draft_names, draft_page, per_block = dsa\n"
    "        num_blocks = available_memory // per_block\n"
    "        num_blocks = may_override_num_blocks(vllm_config, num_blocks)\n"
    "        kv_cache_tensors = [\n"
    "            # Unchanged from the drafterless layout: one tensor per target\n"
    "            # layer, sized by that layer's own page. No padding, no sharing.\n"
    "            KVCacheTensor(\n"
    "                size=target_page_by_name[name] * num_blocks, shared_by=[name]\n"
    "            )\n"
    "            for name in target_names\n"
    "        ] + [\n"
    "            # DSA-DRAFTER-GROUP (standalone): compact per-layer drafter\n"
    "            # tensors, indexed by the shared pool block ids. Contiguous\n"
    "            # reshape -- valid under any kernel-block split, unlike the\n"
    "            # page_size_padded strided view, which is why the drafter specs\n"
    "            # reaching here are guaranteed unpadded.\n"
    "            KVCacheTensor(size=draft_page * num_blocks, shared_by=[name])\n"
    "            for name in draft_names\n"
    "        ]\n"
    "    elif all(\n"
    "        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)\n"
    "        for group in kv_cache_groups\n"
    "    ):\n"
    "        # DeepseekV4: UniformTypeKVCacheSpecs but multiple groups.\n"
    "        # Delegate to the DeepseekV4-specific allocator.\n",
)

# --------------------------------------------------------------------------
# D. get_kv_cache_groups: dispatch.
# --------------------------------------------------------------------------
apply(
    "DSA-DRAFTER-GROUP: big GLM-5.3 (MLA + DSA indexer, no mamba) with the",
    "        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)\n"
    "        _annotate_eagle_groups_deepseek_v4(vllm_config, kv_cache_spec, kv_cache_groups)\n"
    "        return kv_cache_groups\n"
    "\n"
    "    # Pull HiddenStateCacheSpec layers out before the general multi-group\n",
    "        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)\n"
    "        _annotate_eagle_groups_deepseek_v4(vllm_config, kv_cache_spec, kv_cache_groups)\n"
    "        return kv_cache_groups\n"
    "    elif dsa_groups := _get_kv_cache_groups_dsa_drafter(vllm_config, kv_cache_spec):\n"
    "        # DSA-DRAFTER-GROUP: big GLM-5.3 (MLA + DSA indexer, no mamba) with the\n"
    "        # DFlash2 drafter attached. Must precede the generic fallthrough below:\n"
    "        # unify_kv_cache_spec_page_size would try to pad every layer to the\n"
    "        # drafter's page and raise\n"
    "        #   'page size is not divisible by the maximum page size and cannot be\n"
    "        #    padded'\n"
    "        # on the indexer, whose 132 B/token neither divides nor pads into it.\n"
    "        return dsa_groups\n"
    "\n"
    "    # Pull HiddenStateCacheSpec layers out before the general multi-group\n",
)

# --------------------------------------------------------------------------
# E. _max_memory_usage_bytes_from_groups: the boot-time pool-demand gate.
# --------------------------------------------------------------------------
apply(
    "DSA-DRAFTER-GROUP. MUST precede the DeepseekV4 branch",
    "    elif all(\n"
    "        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)\n"
    "        for group in kv_cache_groups\n"
    "    ):\n"
    "        # Special case (only DeepseekV4 for now): all groups are\n",
    "    elif (dsa := _dsa_drafter_tensor_layout(kv_cache_groups)) is not None:\n"
    "        # DSA-DRAFTER-GROUP. MUST precede the DeepseekV4 branch below: that\n"
    "        # branch also matches 'all groups are UniformTypeKVCacheSpecs', but it\n"
    "        # assumes a shared padded layer-tuple layout and multiplies by\n"
    "        # num_layer_tuples -- nonsense for a target+drafter pair whose groups\n"
    "        # share no tuple structure at all.\n"
    "        #\n"
    "        # Every block id drawn from the shared pool -- target- or drafter-owned\n"
    "        # -- costs the FULL per-block byte sum, because the drafter's\n"
    "        # standalone tensors are all `num_blocks` long. Keep this in lock-step\n"
    "        # with the demand expression in _get_kv_cache_groups_dsa_drafter.\n"
    "        _, _, _, _, per_block = dsa\n"
    "        blocks_needed = _dsa_group_max_pages(\n"
    "            kv_cache_groups[0].kv_cache_spec, vllm_config\n"
    "        )\n"
    "        # The drafter's demand is window-bounded (SlidingWindowSpec.\n"
    "        # max_admission_blocks_per_request), not context-bounded.\n"
    "        blocks_needed += _dsa_group_max_pages(\n"
    "            kv_cache_groups[1].kv_cache_spec, vllm_config\n"
    "        )\n"
    "        return blocks_needed * per_block\n"
    "    elif all(\n"
    "        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)\n"
    "        for group in kv_cache_groups\n"
    "    ):\n"
    "        # Special case (only DeepseekV4 for now): all groups are\n",
)

ast.parse(src, filename=path)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print(f"[ok]   {REL}: written, ast.parse clean")
print("\nDSA-DRAFTER-GROUP patch applied. VLLM_ROOT =", VLLM_ROOT)
