#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""B12X-PORT of vLLM PR #52816 (DFlash2) onto
vllm-node-tf5-glm52-b12x:probe-modded  (vLLM 0.23.1rc1.dev190+gab6660699).

This is the ADAPTED form of ~/dflash2-overlay/patch_registry_and_select.py,
whose anchors were written for the donor tree (0.1.dev20051+g487ecf187).
Every anchor below was re-derived from the TARGET BASE's actual source.

Run INSIDE the image at build time, AFTER copying the new files in:
    $VLLM/model_executor/models/qwen3_dflash2.py
    $VLLM/v1/worker/gpu/spec_decode/dflash2/{__init__,speculator}.py

Every edit is anchored (asserted present exactly once), idempotent (skipped
when its marker is already in the file) and ast.parse-gated before writing.
"""

import ast
import os
import sys

VLLM_ROOT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/usr/local/lib/python3.12/dist-packages/vllm"
)


def patch_file(rel_path: str, edits: list[tuple[str, str, str]]) -> None:
    path = os.path.join(VLLM_ROOT, rel_path)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    changed = False
    for marker, old, new in edits:
        if marker in src:
            print(f"[skip] {rel_path}: already applied ({marker[:56]!r})")
            continue
        assert old in src, f"ANCHOR NOT FOUND in {rel_path}:\n{old[:300]!r}"
        assert src.count(old) == 1, (
            f"ANCHOR NOT UNIQUE ({src.count(old)}x) in {rel_path}:\n{old[:300]!r}"
        )
        src = src.replace(old, new)
        changed = True
        print(f"[edit] {rel_path}: applied ({marker[:56]!r})")
    ast.parse(src, filename=path)
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"[ok]   {rel_path}: written, ast.parse clean")
    else:
        print(f"[ok]   {rel_path}: no changes needed, ast.parse clean")


# --------------------------------------------------------------------------
# 0. dflash2/speculator.py: the donor file imports tl_rand32 from gumbel.py.
#    The target base's gumbel.py has tl_rand64 but NOT tl_rand32 -- its fp32
#    branch inlines exactly `tl.rand` + `tl.maximum(u, _TL_RAND_MIN)`, which
#    IS tl_rand32(includes_zero=False). Define it locally rather than editing
#    gumbel.py, which is on the production MTP sampling path.
# --------------------------------------------------------------------------
patch_file(
    "v1/worker/gpu/spec_decode/dflash2/speculator.py",
    [
        (
            "B12X-PORT tl_rand32",
            "from vllm.v1.worker.gpu.sample.gumbel import tl_rand32, tl_rand64\n",
            "# B12X-PORT tl_rand32: the target base's gumbel.py predates the\n"
            "# tl_rand32 factor-out; its fp32 branch inlines the identical draw\n"
            "# (tl.rand then clamp to _TL_RAND_MIN). Defining it here keeps the\n"
            "# shared gumbel.py -- used by the production MTP path -- untouched,\n"
            "# and the draft walk and the verification draw identical noise.\n"
            "from vllm.v1.worker.gpu.sample.gumbel import _TL_RAND_MIN, tl_rand64\n"
            "\n"
            "\n"
            "@triton.jit\n"
            "def tl_rand32(seed, offset, includes_zero: tl.constexpr):\n"
            "    u = tl.rand(seed, offset)\n"
            "    if not includes_zero:\n"
            "        u = tl.maximum(u, _TL_RAND_MIN)\n"
            "    return u\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 1. registry.py: DFlash2DraftModel entry.  (anchor identical to the donor)
# --------------------------------------------------------------------------
patch_file(
    "model_executor/models/registry.py",
    [
        (
            '"DFlash2DraftModel"',
            '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n',
            '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
            "    # B12X-PORT PR#52816\n"
            '    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),\n',
        ),
    ],
)

# --------------------------------------------------------------------------
# 2. v1/worker/gpu/spec_decode/__init__.py: speculator selection.
#    (anchor identical to the donor)
# --------------------------------------------------------------------------
patch_file(
    "v1/worker/gpu/spec_decode/__init__.py",
    [
        (
            "DFlash2Speculator",
            '    if speculative_config.method == "dflash":\n'
            "        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (\n",
            '    if speculative_config.method == "dflash":\n'
            "        # B12X-PORT PR#52816: route DFlash2 drafts (declared by\n"
            "        # architecture, or by a dflash_config carrying a candidate\n"
            "        # selector) to the DFlash2 speculator. On the plain DFlash path\n"
            "        # such a checkpoint would silently draft as DFlash1.\n"
            "        _draft_cfg = speculative_config.draft_model_config\n"
            '        _dflash_cfg = getattr(_draft_cfg.hf_config, "dflash_config", None) or {}\n'
            '        if "DFlash2DraftModel" in (_draft_cfg.architectures or []) or (\n'
            '            "selector_rank" in _dflash_cfg\n'
            "        ):\n"
            "            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (\n"
            "                DFlash2Speculator,\n"
            "            )\n"
            "\n"
            "            return DFlash2Speculator(vllm_config, device)\n"
            "        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 3. model_executor/models/qwen3_dflash.py
#
#    The target base's DFlash v1 is ~7 weeks older than the donor's: it has no
#    per-layer sliding-window / causal resolution, no `layer_idx` on the
#    decoder layer, and no decoder_layer_cls / model_cls subclass hooks. The
#    donor's edit 3 assumed all of the former already present. The GLM-5.3
#    DFlash2 drafter checkpoint declares layer_types = 6 x sliding_attention,
#    sliding_window = 2048 and top-level is_causal = false, so all of it is
#    load-bearing and is ported here alongside the two subclass hooks.
# --------------------------------------------------------------------------
_ATTN_HELPERS = '''logger = init_logger(__name__)


# ---------------------------------------------------------------- B12X-PORT
# PR#52816 (+ the per-layer attention resolution the merged DFlash v1 grew
# alongside it). The target base predates both.
_SLIDING_ATTENTION = "sliding_attention"


def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:
    """Resolve explicit causality before falling back to legacy layer defaults.

    Honors a top-level ``is_causal`` and a ``dflash_config``-level ``is_causal``
    (the GLM-5.3 DFlash2 drafter ships the former) before the legacy
    ``dflash_config.causal`` key.
    """
    dflash_cfg = getattr(config, "dflash_config", None) or {}
    is_causal = getattr(config, "is_causal", None)
    if is_causal is None:
        is_causal = dflash_cfg.get("is_causal")
    if is_causal is not None:
        return bool(is_causal)
    override = dflash_cfg.get("causal")
    if override is not None:
        return bool(override)
    layer_types = getattr(config, "layer_types", None)
    return bool(layer_types) and layer_types[layer_idx] == _SLIDING_ATTENTION


def dflash_has_any_non_causal(config: Qwen3Config) -> bool:
    """Whether the draft needs a non-causal-capable attention backend."""
    return not all(
        _dflash_layer_causal(config, i) for i in range(config.num_hidden_layers)
    )


def _resolve_layer_attention(
    config: Qwen3Config, layer_idx: int
) -> tuple[int | None, bool]:
    """Resolve ``(sliding_window, causal)`` for one DFlash draft layer.

    ``layer_types[i] == "sliding_attention"`` (or ``dflash_config.use_swa``,
    which forces SWA on every layer) selects sliding-window attention; the
    window comes from ``dflash_config.swa_window_size`` or the top-level
    ``sliding_window``. Causality is resolved by ``_dflash_layer_causal``.
    """
    dflash_config = getattr(config, "dflash_config", None) or {}
    layer_types = getattr(config, "layer_types", None)
    use_swa = dflash_config.get("use_swa", False)

    any_sliding = False
    if layer_types is not None:
        num_sliding = sum(lt == _SLIDING_ATTENTION for lt in layer_types)
        any_sliding = num_sliding > 0
        # Mixed sliding/full attention needs multiple KV groups (V2 runner only).
        if (
            0 < num_sliding < len(layer_types)
            and not get_current_vllm_config().use_v2_model_runner
        ):
            raise NotImplementedError(
                "DFlash drafters with mixed sliding/full attention require "
                "the V2 model runner; relaunch with "
                "VLLM_USE_V2_MODEL_RUNNER=1."
            )

    # ``use_swa`` forces SWA on every layer, even an all-full ``layer_types``.
    if layer_types is None or (use_swa and not any_sliding):
        is_sliding = use_swa
    else:
        is_sliding = layer_types[layer_idx] == _SLIDING_ATTENTION

    sliding_window = None
    if is_sliding:
        sliding_window = dflash_config.get(
            "swa_window_size", getattr(config, "sliding_window", None)
        )
        if sliding_window is None:
            raise ValueError(
                "DFlash sliding attention requires a window size configured in "
                "dflash_config.swa_window_size or the top-level sliding_window."
            )

    return sliding_window, _dflash_layer_causal(config, layer_idx)


# -------------------------------------------------------------- /B12X-PORT


class DFlashQwen3Attention(nn.Module):
'''

patch_file(
    "model_executor/models/qwen3_dflash.py",
    [
        # 3a. per-layer attention resolution helpers
        (
            "def _resolve_layer_attention(",
            "logger = init_logger(__name__)\n\n\nclass DFlashQwen3Attention(nn.Module):\n",
            _ATTN_HELPERS,
        ),
        # 3b. DFlashQwen3Attention: accept sliding_window / causal
        (
            "sliding_window: int | None = None,\n        causal: bool = False,",
            "        rms_norm_eps: float = 1e-06,\n"
            "        attention_bias: bool = False,\n"
            "        cache_config: CacheConfig | None = None,\n",
            "        rms_norm_eps: float = 1e-06,\n"
            "        attention_bias: bool = False,\n"
            "        # B12X-PORT PR#52816: per-layer sliding window / causality.\n"
            "        sliding_window: int | None = None,\n"
            "        causal: bool = False,\n"
            "        cache_config: CacheConfig | None = None,\n",
        ),
        # 3c. DFlashQwen3Attention: wire them into the Attention layer
        (
            "per_layer_sliding_window=sliding_window",
            "        self.attn = Attention(\n"
            "            self.num_heads,\n"
            "            self.head_dim,\n"
            "            self.scaling,\n"
            "            num_kv_heads=self.num_kv_heads,\n"
            "            cache_config=cache_config,\n"
            "            quant_config=quant_config,\n"
            '            prefix=f"{prefix}.attn",\n'
            "            attn_type=attn_type,\n"
            "        )\n"
            "        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)\n",
            "        # B12X-PORT PR#52816: a DFlash2 drafter layer is sliding-window.\n"
            "        self.sliding_window = sliding_window\n"
            "        self.attn = Attention(\n"
            "            self.num_heads,\n"
            "            self.head_dim,\n"
            "            self.scaling,\n"
            "            num_kv_heads=self.num_kv_heads,\n"
            "            cache_config=cache_config,\n"
            "            quant_config=quant_config,\n"
            "            per_layer_sliding_window=sliding_window,\n"
            '            prefix=f"{prefix}.attn",\n'
            "            attn_type=attn_type,\n"
            "        )\n"
            "        self.causal = causal\n"
            "        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)\n",
        ),
        # 3d. DFlashQwen3DecoderLayer: accept layer_idx
        (
            "layer_idx: int = 0,\n        cache_config: CacheConfig | None = None,",
            "        vllm_config: VllmConfig,\n"
            "        *,\n"
            "        config: Qwen3Config,\n"
            "        cache_config: CacheConfig | None = None,\n",
            "        vllm_config: VllmConfig,\n"
            "        *,\n"
            "        config: Qwen3Config,\n"
            "        # B12X-PORT PR#52816: DFlash2 subclasses key off the layer index.\n"
            "        layer_idx: int = 0,\n"
            "        cache_config: CacheConfig | None = None,\n",
        ),
        # 3e-i. DFlashQwen3DecoderLayer: resolve this layer's attention mode
        (
            "_resolve_layer_attention(config, layer_idx)",
            "        set_default_rope_theta(config, default_theta=1000000)\n"
            "        attn_type = AttentionType.DECODER\n"
            "\n"
            "        self.self_attn = DFlashQwen3Attention(\n",
            "        set_default_rope_theta(config, default_theta=1000000)\n"
            "        attn_type = AttentionType.DECODER\n"
            "\n"
            "        # B12X-PORT PR#52816: full vs sliding window, causal vs not.\n"
            "        sliding_window, causal = _resolve_layer_attention(config, layer_idx)\n"
            "\n"
            "        self.self_attn = DFlashQwen3Attention(\n",
        ),
        # 3e-ii. ... and pass them down
        (
            "sliding_window=sliding_window,\n            causal=causal,",
            '            attention_bias=getattr(config, "attention_bias", False),\n'
            '            head_dim=getattr(config, "head_dim", None),\n',
            '            attention_bias=getattr(config, "attention_bias", False),\n'
            "            sliding_window=sliding_window,\n"
            "            causal=causal,\n"
            '            head_dim=getattr(config, "head_dim", None),\n',
        ),
        # 3f. DFlashQwen3Model: decoder_layer_cls subclass hook
        (
            "decoder_layer_cls = DFlashQwen3DecoderLayer",
            "@support_torch_compile\nclass DFlashQwen3Model(nn.Module):\n    def __init__(\n",
            "@support_torch_compile\n"
            "class DFlashQwen3Model(nn.Module):\n"
            "    # B12X-PORT PR#52816: subclass hook for DFlash2.\n"
            "    decoder_layer_cls = DFlashQwen3DecoderLayer\n"
            "\n"
            "    def __init__(\n",
        ),
        # 3g. ... and its use site
        (
            "self.decoder_layer_cls(",
            "                DFlashQwen3DecoderLayer(\n"
            "                    current_vllm_config,\n"
            "                    config=self.config,\n"
            "                    cache_config=current_vllm_config.cache_config,\n",
            "                self.decoder_layer_cls(  # B12X-PORT PR#52816\n"
            "                    current_vllm_config,\n"
            "                    config=self.config,\n"
            "                    layer_idx=layer_idx,\n"
            "                    cache_config=current_vllm_config.cache_config,\n",
        ),
        # 3h. DFlashQwen3ForCausalLM: model_cls subclass hook
        (
            "model_cls = DFlashQwen3Model",
            "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
            '    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):\n',
            "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
            "    # B12X-PORT PR#52816: subclass hook for DFlash2.\n"
            "    model_cls = DFlashQwen3Model\n"
            "\n"
            '    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):\n',
        ),
        # 3i. ... and its use site
        (
            "self.model = self.model_cls(",
            "        self.model = DFlashQwen3Model(\n            vllm_config=vllm_config,\n",
            "        self.model = self.model_cls(  # B12X-PORT PR#52816\n"
            "            vllm_config=vllm_config,\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 3j. v1/worker/gpu/spec_decode/dflash/utils.py -- the draft attention backend
#     is selected from `get_dflash_causal`, which in the target base reads ONLY
#     the legacy `dflash_config.causal` key. The GLM-5.3 DFlash2 drafter
#     declares top-level `is_causal: false`; route the decision through the
#     same resolver the layers use so the two can never disagree.
#     (For this checkpoint both spellings yield non-causal; this makes an
#     `is_causal: true` checkpoint correct too.)
# --------------------------------------------------------------------------
patch_file(
    "v1/worker/gpu/spec_decode/dflash/utils.py",
    [
        (
            "B12X-PORT causal",
            "def get_dflash_causal(draft_model_config: ModelConfig) -> bool:\n"
            '    """Whether the DFlash draft uses causal (vs non-causal) attention."""\n'
            '    dflash_config = getattr(draft_model_config.hf_config, "dflash_config", None) or {}\n'
            '    return dflash_config.get("causal", False)\n',
            "def get_dflash_causal(draft_model_config: ModelConfig) -> bool:\n"
            '    """Whether the DFlash draft uses causal (vs non-causal) attention."""\n'
            "    # B12X-PORT causal (PR#52816): resolve through the same helper the\n"
            "    # draft layers use, so a top-level `is_causal` (GLM-5.3 DFlash2) or a\n"
            "    # `dflash_config.is_causal` is honored, not just the legacy `causal`.\n"
            "    from vllm.model_executor.models.qwen3_dflash import (\n"
            "        dflash_has_any_non_causal,\n"
            "    )\n"
            "\n"
            "    return not dflash_has_any_non_causal(draft_model_config.hf_config)\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 4. v1/worker/gpu/spec_decode/speculator.py: draft_logits_spec hook.
#    ADAPTED: the target base already allocates the cache as fp32 (the donor
#    used model_config.head_dtype), so only the FILL value changes for other
#    speculators -- and it does not: the default spec returns (fp32, 0.0),
#    byte-identical to today's torch.zeros. DFlash2 overrides it to -inf.
# --------------------------------------------------------------------------
patch_file(
    "v1/worker/gpu/spec_decode/speculator.py",
    [
        (
            "self.draft_logits_spec(",
            "        self.draft_logits: torch.Tensor | None = None\n"
            '        if self.speculative_config.draft_sample_method == "probabilistic":\n'
            "            self.draft_logits = torch.zeros(\n"
            "                self.max_num_reqs,\n"
            "                self.num_speculative_steps,\n"
            "                self.vocab_size,\n"
            "                dtype=torch.float32,\n"
            "                device=device,\n"
            "            )\n",
            "        self.draft_logits: torch.Tensor | None = None\n"
            '        if self.speculative_config.draft_sample_method == "probabilistic":\n'
            "            # B12X-PORT PR#52816: dtype/fill via draft_logits_spec so\n"
            "            # DFlash2 can cache a sparse fp32/-inf distribution. The\n"
            "            # default spec is (torch.float32, 0.0) -- identical to the\n"
            "            # torch.zeros this replaces for every other speculator.\n"
            "            dtype, fill = self.draft_logits_spec(vllm_config)\n"
            "            self.draft_logits = torch.full(\n"
            "                (\n"
            "                    self.max_num_reqs,\n"
            "                    self.num_speculative_steps,\n"
            "                    self.vocab_size,\n"
            "                ),\n"
            "                fill,\n"
            "                dtype=dtype,\n"
            "                device=device,\n"
            "            )\n",
        ),
        (
            "def draft_logits_spec(",
            "    @abstractmethod\n"
            "    def load_draft_model(\n"
            "        self,\n"
            "        target_model: nn.Module,\n"
            "        target_attn_layer_names: set[str],\n"
            "    ) -> nn.Module:\n"
            "        pass\n",
            "    def draft_logits_spec(\n"
            "        self, vllm_config: VllmConfig\n"
            "    ) -> tuple[torch.dtype, float]:\n"
            '        """Dtype and fill for the cached proposal distribution.\n'
            "\n"
            "        Speculators that write only a subset of columns each step\n"
            "        override this. (B12X-PORT PR#52816)\n"
            '        """\n'
            "        return torch.float32, 0.0\n"
            "\n"
            "    @abstractmethod\n"
            "    def load_draft_model(\n"
            "        self,\n"
            "        target_model: nn.Module,\n"
            "        target_attn_layer_names: set[str],\n"
            "    ) -> nn.Module:\n"
            "        pass\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 5. config/vllm.py: force the V2 model runner for a DFlash2 draft.
#    ADAPTED: the target base has no _dflash_needs_multi_kv_group() to anchor
#    after, so the check is inserted at the head of use_v2_model_runner's
#    body-after-the-env-override, and the helper before
#    _is_default_v2_model_runner_model().
# --------------------------------------------------------------------------
patch_file(
    "config/vllm.py",
    [
        (
            "_is_dflash2_draft():",
            "        if self.model_config is not None and self.model_config.is_diffusion:\n"
            "            return True\n",
            "        # B12X-PORT PR#52816: the DFlash2 candidate selector exists only\n"
            "        # in the V2 speculator; on V1 the same checkpoint would silently\n"
            "        # draft as DFlash1.\n"
            "        if self._is_dflash2_draft():\n"
            "            return True\n"
            "\n"
            "        if self.model_config is not None and self.model_config.is_diffusion:\n"
            "            return True\n",
        ),
        (
            "def _is_dflash2_draft(",
            "    def _is_default_v2_model_runner_model(self) -> bool:\n"
            "        model_config = self.model_config\n",
            "    def _is_dflash2_draft(self) -> bool:\n"
            '        """B12X-PORT PR#52816: whether the DFlash draft is a DFlash2\n'
            "        one, by the same signals the speculator selection uses\n"
            '        (v1/worker/gpu/spec_decode/__init__.py)."""\n'
            "        spec = self.speculative_config\n"
            '        if spec is None or getattr(spec, "method", None) != "dflash":\n'
            "            return False\n"
            '        draft_config = getattr(spec, "draft_model_config", None)\n'
            "        if draft_config is None:\n"
            "            return False\n"
            '        if "DFlash2DraftModel" in (draft_config.architectures or []):\n'
            "            return True\n"
            "        dflash_cfg = (\n"
            '            getattr(draft_config.hf_config, "dflash_config", None) or {}\n'
            "        )\n"
            '        return "selector_rank" in dflash_cfg\n'
            "\n"
            "    def _is_default_v2_model_runner_model(self) -> bool:\n"
            "        model_config = self.model_config\n",
        ),
    ],
)

# --------------------------------------------------------------------------
# 6. model_executor/layers/logits_processor.py: get_top_k_tokens.
#    ADAPTED: the target base has no _apply_head() and no lm_head.tp_size, so
#    the projection goes through lm_head.quant_method.apply and the TP degree
#    through get_tensor_model_parallel_world_size() -- exactly what the base's
#    own get_top_tokens() does. Everything else is verbatim from the PR.
# --------------------------------------------------------------------------
patch_file(
    "model_executor/layers/logits_processor.py",
    [
        (
            "_flashinfer_topk",
            "from vllm.platforms import current_platform\n",
            "from vllm.platforms import current_platform\n"
            "\n"
            "# B12X-PORT PR#52816: vocab-parallel top-k for the DFlash2 candidate\n"
            "# selector -- from the merged logits_processor.py (b389ac294).\n"
            "from collections.abc import Callable\n"
            "from functools import cache\n"
            "\n"
            "from vllm.logger import init_logger\n"
            "from vllm.utils.flashinfer import has_flashinfer\n"
            "\n"
            "logger = init_logger(__name__)\n"
            "\n"
            "\n"
            "@cache\n"
            "def _flashinfer_topk() -> (\n"
            "    Callable[..., tuple[torch.Tensor, torch.Tensor]] | None\n"
            "):\n"
            '    """FlashInfer\'s radix top-k, or None for torch.topk.\n'
            "\n"
            "    The top-k spans the vocabulary, where the radix kernel is about twice\n"
            "    torch.topk.\n"
            '    """\n'
            "    if not current_platform.is_cuda():\n"
            "        return None\n"
            "    if not has_flashinfer():\n"
            "        logger.info_once(\n"
            '            "flashinfer is unavailable; vocab-parallel top-k uses '
            'torch.topk, "\n'
            '            "at roughly half the speed."\n'
            "        )\n"
            "        return None\n"
            "    from flashinfer import top_k\n"
            "\n"
            "    return top_k\n"
            "\n"
            "\n"
            "def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:\n"
            "    impl = _flashinfer_topk()\n"
            "    if impl is None or not scores.is_cuda:\n"
            "        return torch.topk(scores, k, dim=-1)\n"
            "    return impl(scores, k, sorted=True, deterministic=True)\n",
        ),
        (
            "def get_top_k_tokens(",
            "    def extra_repr(self) -> str:\n",
            "    # B12X-PORT PR#52816: from the merged logits_processor.py, with the\n"
            "    # projection and TP degree taken the way this tree's get_top_tokens\n"
            "    # takes them (no _apply_head / lm_head.tp_size in this base).\n"
            "    def get_top_k_tokens(\n"
            "        self,\n"
            "        lm_head: VocabParallelEmbedding,\n"
            "        hidden_states: torch.Tensor,\n"
            "        k: int,\n"
            "        embedding_bias: torch.Tensor | None = None,\n"
            "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
            '        """Vocab-parallel top-k without all-gathering full logits.\n'
            "\n"
            "        The `get_top_tokens` reduction widened from one token to k,\n"
            "        returning the values as well as the global ids. Communication is\n"
            "        O(batch * 2k * tp_size) rather than O(batch * vocab_size).\n"
            "\n"
            "        Scale and soft cap are applied to the k selected values rather\n"
            "        than the whole vocabulary; both are monotonic, so the selection\n"
            "        is the same and only k entries are touched.\n"
            '        """\n'
            "        if self.scale <= 0.0 and self.scale != 1.0:\n"
            "            raise ValueError(\n"
            '                "The local top-k reduction optimization is not supported '
            'for "\n'
            '                "non-positive logit scaling factors."\n'
            "            )\n"
            "        tp_size = get_tensor_model_parallel_world_size()\n"
            "\n"
            "        logits = lm_head.quant_method.apply(\n"
            "            lm_head, hidden_states, bias=embedding_bias\n"
            "        )\n"
            "\n"
            "        # Mask out padding entries beyond org_vocab_size on this shard.\n"
            "        num_pad = lm_head.shard_indices.num_org_vocab_padding\n"
            "        if num_pad > 0:\n"
            '            logits[..., -num_pad:] = -float("inf")\n'
            "\n"
            "        values, ids = _topk(logits, k)\n"
            "        # Convert shard-local indices to global vocab indices.\n"
            "        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index\n"
            "\n"
            "        if tp_size > 1:\n"
            "            values = tensor_model_parallel_all_gather(values, dim=-1)\n"
            "            ids = tensor_model_parallel_all_gather(ids, dim=-1)\n"
            "            values, selected = _topk(values, k)\n"
            "            ids = ids.gather(-1, selected)\n"
            "\n"
            "        values = values.float()\n"
            "        if self.scale != 1.0:\n"
            "            values = values * self.scale\n"
            "        if self.soft_cap is not None:\n"
            "            values = torch.tanh(values / self.soft_cap) * self.soft_cap\n"
            "        return ids, values\n"
            "\n"
            "    def extra_repr(self) -> str:\n",
        ),
    ],
)

print("\nAll DFlash2 wiring patches applied and ast-checked. VLLM_ROOT =", VLLM_ROOT)
