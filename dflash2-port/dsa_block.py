# ------------------------------------------------------------------------
# DSA-DRAFTER-GROUP: big GLM-5.3 (MLA + DSA indexer, no mamba) + DFlash2.
# ------------------------------------------------------------------------

# Floor for the drafter's manager block size (see _dsa_draft_block_candidates).
# 32 rather than 16 because ``resolve_kv_cache_block_sizes`` sets
# ``hash_block_size = gcd(group block sizes)``: dropping the drafter to 16 would
# quadruple prefix-cache block hashing for the whole model to buy a fraction of
# a percent of pool. Override with VLLM_DSA_DRAFT_MIN_BLOCK (set it to the
# target block size to disable the shrink entirely).
_DSA_DRAFT_MIN_BLOCK_DEFAULT = 32


def _dsa_draft_min_block() -> int:
    raw = os.environ.get("VLLM_DSA_DRAFT_MIN_BLOCK")
    if not raw:
        return _DSA_DRAFT_MIN_BLOCK_DEFAULT
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Ignoring malformed VLLM_DSA_DRAFT_MIN_BLOCK=%r; using %d.",
            raw,
            _DSA_DRAFT_MIN_BLOCK_DEFAULT,
        )
        return _DSA_DRAFT_MIN_BLOCK_DEFAULT


def _dsa_drafter_decline(reason: str, *args) -> None:
    """Say why this branch is NOT claiming a model that looked like its case.

    Called only once we know the spec dict holds BOTH a plain
    ``SlidingWindowSpec`` drafter and at least one exact ``MLAAttentionSpec``
    target, so this never fires for ordinary hybrid models. It is a WARNING
    because the only thing that follows a wrong decline is the stock
    ``unify_kv_cache_spec_page_size`` NotImplementedError, which says nothing
    about why the specialised path stood down.
    """
    logger.warning("DSA-DRAFTER-GROUP: declining -- " + reason, *args)


def _dsa_group_max_pages(spec: KVCacheSpec, vllm_config: VllmConfig) -> int:
    """Block-id demand of one group, for either group-spec shape.

    ``UniformTypeKVCacheSpecs`` already exposes this as
    ``max_memory_usage_pages``; a merged plain spec (the
    ``is_kv_cache_spec_uniform`` shape) has to be reduced by hand.
    """
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.max_memory_usage_pages(vllm_config)
    return cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)


def _dsa_drafterless_target_groups(
    vllm_config: VllmConfig,
    target_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Exactly what ``get_kv_cache_groups`` returns for the DRAFTERLESS model.

    This REPLAYS the dispatch prefix of ``get_kv_cache_groups`` -- in the same
    order -- on the target-only spec dict, so the target group this branch emits
    is byte-for-byte the drafterless group by construction, on whatever backend,
    KV dtype and block size the run happens to use, rather than by re-deriving a
    layout that only matches one geometry.

    Only the two single-group shapes are replayed. The DeepseekV4
    (``group_and_unify_kv_cache_specs``) and GLM-5-Next branches are
    deliberately NOT replayed: both are multi-group layouts with their own
    tensor-emission and accounting paths, and both are matched ahead of this
    branch in ``get_kv_cache_groups`` anyway, so a target set that would hit
    them cannot reach here with a plain SWA drafter attached.

    Returns the group list (never mutating ``target_specs``), or None.
    """
    probe = dict(target_specs)
    if is_kv_cache_spec_uniform(probe):
        # Every target layer identical -> one merged plain spec.
        return _get_kv_cache_groups_uniform_spec(probe)
    uniform = UniformTypeKVCacheSpecs.from_specs(probe)
    if uniform is not None:
        # Same type, different hidden sizes (MLA 656 B/token + DSA indexer
        # 132 B/token) -> one group holding the per-layer specs.
        return _get_kv_cache_groups_uniform_type(uniform)
    return None


def _dsa_draft_block_candidates(target_block: int, native_block: int) -> list[int]:
    """Legal manager block sizes for the standalone drafter group.

    Constraints, in order of severity:
      * must DIVIDE the target block, so ``resolve_kv_cache_block_sizes``'
        scheduler LCM stays at ``target_block`` instead of exploding, and the
        prefix-cache GCD stays a clean divisor;
      * must be a multiple of 16, the smallest kernel block every SWA backend
        in this build advertises (FlashAttention / Triton report
        ``MultipleOf(16)``), so ``select_common_block_size`` always finds a
        clean virtual split of the manager block;
      * must be >= ``_dsa_draft_min_block()`` (see that constant).
    ``native_block`` is always included as a fallback so the candidate list is
    never empty and we can always fall back to "leave the drafter alone".
    """
    floor = _dsa_draft_min_block()
    cands = {
        d
        for d in range(16, target_block + 1, 16)
        if target_block % d == 0 and d >= floor
    }
    cands.add(native_block)
    return sorted(cands)


def _get_kv_cache_groups_dsa_drafter(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Groups for a DSA model (MLA + indexer, no mamba) plus a DFlash2 drafter.

    Returns ``[target_group, drafter_group]`` -- the target group FIRST so its
    group id stays 0 and the proven per-layer-page layout is bit-for-bit what
    the drafterless model produces -- or None if this is not that model.
    """
    # EXACT type test, not isinstance: KpoolTailSpec and SlidingWindowMLASpec
    # both subclass SlidingWindowSpec and must NOT be swept into the drafter
    # group (the kpool tail belongs to the glm5-next path; SlidingWindowMLASpec
    # belongs to DeepseekV4's group_and_unify path).
    draft_specs: dict[str, KVCacheSpec] = {
        k: v for k, v in kv_cache_spec.items() if type(v) is SlidingWindowSpec
    }
    if not draft_specs:
        return None
    target_specs: dict[str, KVCacheSpec] = {
        k: v for k, v in kv_cache_spec.items() if type(v) is not SlidingWindowSpec
    }
    if not target_specs:
        return None
    # Below this line the model has BOTH a plain SWA drafter and at least one
    # exact MLA target, so it is our case and every further rejection is worth
    # logging. Above it we stay silent: ordinary hybrid models land here too.
    if not any(type(s) is MLAAttentionSpec for s in target_specs.values()):
        return None

    # Every non-drafter layer must be MLA-flavoured. HiddenStateCacheSpec is
    # excluded by exact type: it subclasses MLAAttentionSpec but is pulled out
    # and re-aligned by the generic path below. KpoolTailSpec (glm5-next) and
    # MambaSpec are not MLAAttentionSpec at all, so they land here as failures
    # rather than being silently absorbed. isinstance rather than `type is`
    # only so a future MLA subclass with the same page semantics still works;
    # every exclusion v1 made is still made.
    foreign = {
        name: type(s).__name__
        for name, s in target_specs.items()
        if not isinstance(s, MLAAttentionSpec) or type(s) is HiddenStateCacheSpec
    }
    if foreign:
        _dsa_drafter_decline(
            "%d of %d non-drafter layers are not plain MLA specs (e.g. %s); "
            "this model belongs on another grouping path.",
            len(foreign),
            len(target_specs),
            sorted(foreign.items())[:3],
        )
        return None

    # THE safety argument of the patch: claim this model only if removing the
    # drafter reproduces a SINGLE-group layout, and then reuse that exact group
    # object. See _dsa_drafterless_target_groups.
    target_groups = _dsa_drafterless_target_groups(vllm_config, target_specs)
    if target_groups is None or len(target_groups) != 1:
        _dsa_drafter_decline(
            "the DRAFTERLESS model would not be one KV cache group (%s); this "
            "branch only reproduces a single-group target layout. "
            "target: %d layers, block sizes %s, pages %s, spec types %s.",
            "no single-group shape matched"
            if target_groups is None
            else "%d groups" % len(target_groups),
            len(target_specs),
            sorted({s.block_size for s in target_specs.values()}),
            sorted({s.page_size_bytes for s in target_specs.values()}),
            sorted({type(s).__name__ for s in target_specs.values()}),
        )
        return None
    target_group = target_groups[0]
    target_group_spec = target_group.kv_cache_spec

    # NEVER page_size_padded on a drafter spec, and strip any INHERITED padding
    # BEFORE any geometry math. See module docstring: a padded spec routes the
    # runner into a strided view that overruns once the backend virtually splits
    # the manager block into kernel blocks. On the nvfp4 lane
    # (--kv-cache-dtype-skip-layers) Attention.get_kv_cache_spec stamps every
    # skipped bf16 SWA layer with cache_config.skip_page_size_padded, so this is
    # not hypothetical.
    if any(s.page_size_padded is not None for s in draft_specs.values()):
        draft_specs = {
            name: replace(s, page_size_padded=None) for name, s in draft_specs.items()
        }
    any_draft = next(iter(draft_specs.values()))
    if any(spec != any_draft for spec in draft_specs.values()):
        # Decline rather than assert: a boot-time AssertionError inside the
        # grouping code is strictly worse than falling through to the generic
        # path, which at least names the layer it could not place.
        _dsa_drafter_decline(
            "the %d DFlash2 drafter layers do not share one SlidingWindowSpec "
            "(windows %s, blocks %s, pages %s, dtypes %s).",
            len(draft_specs),
            sorted({s.sliding_window for s in draft_specs.values()}),
            sorted({s.block_size for s in draft_specs.values()}),
            sorted({s.page_size_bytes for s in draft_specs.values()}),
            sorted({str(s.dtype) for s in draft_specs.values()}),
        )
        return None
    assert any_draft.page_size_padded is None

    target_block = target_group_spec.block_size
    native_block = any_draft.block_size
    draft_bytes_per_token = any_draft.page_size_bytes // native_block

    # EXACT PAGE FIT IS IMPOSSIBLE for every MLA record this family ships and is
    # deliberately NOT attempted: 576 = 2**6*9, 656 = 2**4*41 (fp8_ds_mla),
    # 432 = 2**4*27 and 368 = 2**4*23 (nvfp4_ds_mla) all carry an odd factor,
    # while any drafter page is 2 * heads * head_size * sizeof(dtype) -- a power
    # of two at every TP degree and dtype. Check it explicitly anyway so a
    # future record size that DOES allow the cheaper slot-shared layout is
    # visible in the log rather than silently paying for standalone tensors.
    # This is a WARNING, never an assert: an exact fit becoming possible is a
    # better situation, and must not block boot.
    target_pages = {s.page_size_bytes for s in target_specs.values()}
    for page in sorted(target_pages):
        if page % draft_bytes_per_token == 0 and (
            page // draft_bytes_per_token
        ) % 16 == 0:
            logger.warning(
                "DSA-DRAFTER-GROUP: drafter (%d B/token) exactly tiles target "
                "page %d; a slot-shared layout would be cheaper but this branch "
                "only implements standalone drafter tensors.",
                draft_bytes_per_token,
                page,
            )

    # STANDALONE sizing. The drafter's tensors are its own, but they are indexed
    # by SHARED pool block ids, so the block size trades two costs:
    #   * a SMALL block adds few bytes to each pool block but needs many block
    #     ids to cover the sliding window (and a wider block table);
    #   * a LARGE block needs few ids but adds its page to EVERY block in the
    #     pool, including the thousands the 1M-context target group needs.
    # Minimise peak pool demand -- exactly the quantity boot is gated on in
    # _check_enough_kv_cache_memory. This expression MIRRORS the DSA branch of
    # _max_memory_usage_bytes_from_groups; keep the two in lock-step.
    per_block_target = sum(s.page_size_bytes for s in target_specs.values())
    base_blocks = _dsa_group_max_pages(target_group_spec, vllm_config)
    best: tuple[int, int] | None = None
    for cand in _dsa_draft_block_candidates(target_block, native_block):
        probe = replace(any_draft, block_size=cand)
        demand = (
            base_blocks
            + probe.max_admission_blocks_per_request(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.model_config.max_model_len,
            )
        ) * (per_block_target + len(draft_specs) * probe.page_size_bytes)
        # Tie-break toward the LARGER block: coarser prefix-cache hashing and a
        # narrower block table for the same bytes.
        if best is None or demand < best[0] or (demand == best[0] and cand > best[1]):
            best = (demand, cand)
    assert best is not None
    draft_block = best[1]

    new_draft_specs: dict[str, KVCacheSpec] = {
        name: replace(s, block_size=draft_block, page_size_padded=None)
        for name, s in draft_specs.items()
    }
    draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)
    if draft_uniform is None:
        _dsa_drafter_decline(
            "the %d drafter layers are not a uniform type at block %d.",
            len(new_draft_specs),
            draft_block,
        )
        return None

    draft_page = next(iter(new_draft_specs.values())).page_size_bytes
    logger.info(
        "DSA-DRAFTER-GROUP: %d target layers (%d B/block) + %d DFlash2 drafter "
        "layers as a standalone group (block %d, %d B/block each, +%.2f%% pool); "
        "exact page fit impossible (target pages %s vs %d B/token drafter).",
        len(target_specs),
        per_block_target,
        len(new_draft_specs),
        draft_block,
        draft_page,
        100.0 * len(new_draft_specs) * draft_page / per_block_target,
        sorted(target_pages),
        draft_bytes_per_token,
    )

    # Drafter group LAST so the target group keeps id 0. ``target_group`` is the
    # very object the drafterless dispatch built, handed through untouched.
    return [
        target_group,
        KVCacheGroupSpec(list(new_draft_specs), draft_uniform),
    ]


def _dsa_drafter_tensor_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[list[str], dict[str, int], list[str], int, int] | None:
    """Detect the ``_get_kv_cache_groups_dsa_drafter`` layout from the
    (possibly PP-projected) groups, so tensor emission and every accounting
    path derive their numbers from one place and can never disagree.

    Returns:
      - (target_names, target_page_by_name, draft_names, draft_page, per_block)
      - None if these groups are not that layout.
    """
    if len(kv_cache_groups) != 2:
        return None
    target_group, draft_group = kv_cache_groups
    if not isinstance(draft_group.kv_cache_spec, UniformTypeKVCacheSpecs):
        return None

    draft_inner = cast(
        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
    ).kv_cache_specs
    if not draft_inner:
        return None
    # EXACT type: excludes KpoolTailSpec (glm5-next) and SlidingWindowMLASpec
    # (DeepseekV4's group_and_unify path), so this can never shadow either.
    if not all(type(s) is SlidingWindowSpec for s in draft_inner.values()):
        return None
    # A padded drafter spec is the strided-view overrun; reject outright rather
    # than emitting a layout the runner will misread.
    if any(s.page_size_padded is not None for s in draft_inner.values()):
        return None
    draft_pages = {s.page_size_bytes for s in draft_inner.values()}
    if len(draft_pages) != 1:
        return None
    draft_page = draft_pages.pop()

    # The target group takes either shape _dsa_drafterless_target_groups can
    # produce: a UniformTypeKVCacheSpecs holding per-layer specs (MLA + DSA
    # indexer, different pages), or one merged plain spec shared by every layer.
    target_names = list(target_group.layer_names)
    if not target_names:
        return None
    target_spec = target_group.kv_cache_spec
    if isinstance(target_spec, UniformTypeKVCacheSpecs):
        target_inner = target_spec.kv_cache_specs
        if set(target_inner) != set(target_names):
            return None
        member_specs = list(target_inner.values())
        target_page_by_name = {n: target_inner[n].page_size_bytes for n in target_names}
    else:
        member_specs = [target_spec]
        target_page_by_name = {n: target_spec.page_size_bytes for n in target_names}
    if not member_specs:
        return None
    if not all(
        isinstance(s, MLAAttentionSpec) and type(s) is not HiddenStateCacheSpec
        for s in member_specs
    ):
        return None

    draft_names = list(draft_group.layer_names)
    if set(draft_names) != set(draft_inner):
        return None
    # Per-block pool cost: every target layer contributes its OWN page (the
    # per-layer layout is unchanged from the drafterless model, padding
    # included, since page_size_bytes honours page_size_padded on both sides)
    # and every drafter layer contributes its standalone page.
    per_block = sum(target_page_by_name.values()) + len(draft_names) * draft_page
    return target_names, target_page_by_name, draft_names, draft_page, per_block


