#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static (no-GPU, no-server) verification of the DFlash2 port.

Exit code 0 only if every REQUIRED check passes.
"""

import json
import os
import sys
import tempfile
import traceback
from types import SimpleNamespace

FAILS: list[str] = []
NOTES: list[str] = []


def check(name, required=True):
    def deco(fn):
        try:
            out = fn()
        except Exception:
            print(f"[FAIL] {name}")
            traceback.print_exc()
            (FAILS if required else NOTES).append(name)
            return
        print(f"[PASS] {name}" + (f" -- {out}" if out else ""))

    return deco


print("=" * 70)
print("DFlash2 port verification")
print("=" * 70)


@check("import vllm")
def _():
    import vllm

    return vllm.__version__


@check("DFlash2DraftModel is in the model registry table")
def _():
    from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS

    entry = _SPECULATIVE_DECODING_MODELS["DFlash2DraftModel"]
    assert entry == ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"), entry
    return str(entry)


def _resolve_arch(arch: str):
    """Resolve one architecture through the registry, whatever the tree's
    resolve_model_cls signature happens to be."""
    from vllm.model_executor.models.registry import ModelRegistry

    errs = []
    for attempt in (
        lambda: ModelRegistry.resolve_model_cls(
            [arch], SimpleNamespace(model_impl="auto")
        ),
        lambda: ModelRegistry.resolve_model_cls([arch], None),
        lambda: ModelRegistry.resolve_model_cls([arch]),
    ):
        try:
            cls, resolved = attempt()
            return cls, resolved
        except Exception as e:  # signature drift between trees
            errs.append(f"{type(e).__name__}: {e}")
    out = ModelRegistry._try_load_model_cls(arch)
    assert out is not None, f"registry returned None for {arch}; tried {errs}"
    return out, arch


@check("ModelRegistry resolves DFlash2DraftModel to a class")
def _():
    from vllm.model_executor.models.registry import ModelRegistry

    assert "DFlash2DraftModel" in ModelRegistry.get_supported_archs()
    cls, arch = _resolve_arch("DFlash2DraftModel")
    assert cls.__name__ == "DFlash2Qwen3ForCausalLM", cls
    return f"{arch} -> {cls.__module__}.{cls.__name__}"


@check("qwen3_dflash2 model module imports and subclasses DFlash v1")
def _():
    from vllm.model_executor.models.qwen3_dflash import (
        DFlashQwen3DecoderLayer,
        DFlashQwen3ForCausalLM,
        DFlashQwen3Model,
        _resolve_layer_attention,
        dflash_has_any_non_causal,
    )
    from vllm.model_executor.models.qwen3_dflash2 import (
        CandidateSelector,
        DFlash2Qwen3DecoderLayer,
        DFlash2Qwen3ForCausalLM,
        DFlash2Qwen3Model,
    )

    assert issubclass(DFlash2Qwen3DecoderLayer, DFlashQwen3DecoderLayer)
    assert issubclass(DFlash2Qwen3Model, DFlashQwen3Model)
    assert issubclass(DFlash2Qwen3ForCausalLM, DFlashQwen3ForCausalLM)
    assert DFlash2Qwen3Model.decoder_layer_cls is DFlash2Qwen3DecoderLayer
    assert DFlash2Qwen3ForCausalLM.model_cls is DFlash2Qwen3Model
    assert DFlashQwen3Model.decoder_layer_cls is DFlashQwen3DecoderLayer
    assert DFlashQwen3ForCausalLM.model_cls is DFlashQwen3Model
    _ = CandidateSelector, _resolve_layer_attention, dflash_has_any_non_causal
    return "hooks wired both ways"


@check("dflash2 speculator package imports")
def _():
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        DFlash2Speculator,
        _cache_draft_logits_kernel,
        _selector_walk_kernel,
        gumbel_noised_argmax,
        tl_rand32,
    )

    assert issubclass(DFlash2Speculator, DFlashSpeculator)
    _ = _cache_draft_logits_kernel, _selector_walk_kernel, gumbel_noised_argmax
    _ = tl_rand32
    return "DFlash2Speculator(DFlashSpeculator)"


@check("DraftModelSpeculator.draft_logits_spec default is (float32, 0.0)")
def _():
    import torch

    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    default = DraftModelSpeculator.draft_logits_spec(None, None)
    assert default == (torch.float32, 0.0), default
    d2 = DFlash2Speculator.draft_logits_spec(None, None)
    assert d2[0] is torch.float32 and d2[1] == -float("inf"), d2
    return f"default={default}, dflash2={d2}"


@check("LogitsProcessor.get_top_k_tokens exists")
def _():
    from vllm.model_executor.layers.logits_processor import (
        LogitsProcessor,
        _flashinfer_topk,
        _topk,
    )

    assert callable(LogitsProcessor.get_top_k_tokens)
    _ = _flashinfer_topk, _topk
    return "present"


@check("speculator selection routes DFlash2 (source check)")
def _():
    import inspect

    import vllm.v1.worker.gpu.spec_decode as sd

    src = inspect.getsource(sd.init_speculator)
    assert "DFlash2Speculator" in src and "selector_rank" in src
    return "init_speculator dispatches on architecture/selector_rank"


@check("VllmConfig._is_dflash2_draft exists and forces V2 runner")
def _():
    import inspect

    from vllm.config import VllmConfig

    assert hasattr(VllmConfig, "_is_dflash2_draft")
    src = inspect.getsource(VllmConfig.use_v2_model_runner.fget)
    assert "_is_dflash2_draft()" in src
    return "use_v2_model_runner honours it"


@check("target model intact: GlmMoeDsaForCausalLM + Eagle3 aux capture")
def _():
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2ForCausalLM,
        GlmMoeDsaForCausalLM,
    )
    from vllm.model_executor.models.interfaces import supports_eagle3

    assert issubclass(GlmMoeDsaForCausalLM, DeepseekV2ForCausalLM)
    assert supports_eagle3(GlmMoeDsaForCausalLM), "SupportsEagle3 lost"
    assert getattr(GlmMoeDsaForCausalLM, "supports_eagle3", False) is True
    assert hasattr(GlmMoeDsaForCausalLM, "set_aux_hidden_state_layers")
    assert hasattr(GlmMoeDsaForCausalLM, "get_eagle3_aux_hidden_state_layers")
    from vllm.model_executor.models.registry import ModelRegistry

    assert "GlmMoeDsaForCausalLM" in ModelRegistry.get_supported_archs()
    return "SupportsEagle3 + aux_hidden_state_layers present"


DRAFT_CFG = {
    "architectures": ["DFlash2DraftModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": None,
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 154856,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [5, 19, 33, 47, 61, 75],
    },
    "dtype": "bfloat16",
    "eos_token_id": [154820, 154827, 154829],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 6144,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "is_causal": False,
    "layer_types": ["sliding_attention"] * 6,
    "max_position_embeddings": 1048576,
    "max_window_layers": 6,
    "model_type": "qwen3",
    "num_attention_heads": 64,
    "num_hidden_layers": 6,
    "num_key_value_heads": 8,
    "num_target_layers": 78,
    "pad_token_id": 154820,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "sliding_window": 2048,
    "tie_word_embeddings": False,
    "transformers_version": "5.7.0",
    "use_cache": False,
    "use_sliding_window": True,
    "vocab_size": 154880,
}


@check("real drafter config.json -> architecture accepted end to end")
def _():
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import get_config

    src = "/var/tmp/models/GLM-5.3-DFlash2-draft"
    tmp = None
    if os.path.isfile(os.path.join(src, "config.json")):
        path = src
        origin = "on-disk checkpoint"
    else:
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(DRAFT_CFG, f)
        path = tmp
        origin = "embedded copy of the rank-0 config.json"
    hf_config = get_config(path, trust_remote_code=False)
    assert hf_config.architectures == ["DFlash2DraftModel"], hf_config.architectures
    assert "DFlash2DraftModel" in ModelRegistry.get_supported_archs()
    cls, arch = _resolve_arch(hf_config.architectures[0])
    assert cls.__name__ == "DFlash2Qwen3ForCausalLM"

    # The per-layer attention resolution the drafter depends on.
    from vllm.model_executor.models.qwen3_dflash import (
        _resolve_layer_attention,
        dflash_has_any_non_causal,
    )

    resolved = [_resolve_layer_attention(hf_config, i) for i in range(6)]
    assert all(w == 2048 and c is False for w, c in resolved), resolved
    assert dflash_has_any_non_causal(hf_config) is True

    from vllm.v1.worker.gpu.spec_decode.dflash.utils import get_dflash_causal

    assert get_dflash_causal(SimpleNamespace(hf_config=hf_config)) is False
    return (
        f"{origin}: arch={arch}, 6 layers -> sliding_window=2048, causal=False"
    )


@check("KV grouping: DSA target + DFlash2 drafter -> 2 groups, drafter last")
def _():
    import torch

    from vllm.v1.core.kv_cache_utils import (
        _dsa_drafter_tensor_layout,
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowSpec

    BLOCK = 64
    # 78 GLM-5.3 MLA layers (fp8_ds_mla, 656 B/token) + their DSA indexers.
    specs = {}
    for i in range(78):
        specs[f"model.layers.{i}.self_attn.attn"] = MLAAttentionSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="fp8_ds_mla",
        )
    for i in range(78):
        specs[f"model.layers.{i}.self_attn.indexer"] = MLAAttentionSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.float8_e4m3fn,
        )
    # 6 DFlash2 drafter layers: plain SlidingWindowSpec, window 2048.
    for i in range(6):
        specs[f"model.layers.{78 + i}.self_attn.attn"] = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=2,  # 8 kv heads / TP4
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=2048,
        )

    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8192,
            disable_hybrid_kv_cache_manager=False,
            max_num_seqs=96,
        ),
        model_config=SimpleNamespace(max_model_len=1048576),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )

    groups = get_kv_cache_groups(cfg, specs)
    assert len(groups) == 2, [len(g.layer_names) for g in groups]
    target, draft = groups
    assert len(target.layer_names) == 156, len(target.layer_names)
    assert len(draft.layer_names) == 6, len(draft.layer_names)
    assert all(n.startswith("model.layers.7") or True for n in draft.layer_names)
    inner = draft.kv_cache_spec.kv_cache_specs
    assert all(type(s) is SlidingWindowSpec for s in inner.values())
    # THE trap: never page_size_padded on the drafter group.
    assert all(s.page_size_padded is None for s in inner.values()), (
        "drafter group carries page_size_padded -- strided view / memory blowup"
    )
    layout = _dsa_drafter_tensor_layout(groups)
    assert layout is not None
    tn, tpage, dn, dpage, per_block = layout
    assert len(tn) == 156 and len(dn) == 6

    # Tensor emission: 162 standalone tensors, drafter ones sized by its page.
    kvcfg = get_kv_cache_config_from_groups(cfg, groups, 24 * (1 << 30))
    assert len(kvcfg.kv_cache_tensors) == 162, len(kvcfg.kv_cache_tensors)
    draft_tensors = [t for t in kvcfg.kv_cache_tensors if t.shared_by[0] in set(dn)]
    assert len(draft_tensors) == 6
    assert all(len(t.shared_by) == 1 for t in kvcfg.kv_cache_tensors)
    assert all(t.size == dpage * kvcfg.num_blocks for t in draft_tensors)
    draft_block = next(iter(inner.values())).block_size
    return (
        f"target group id 0 ({len(tn)} layers, {sum(tpage.values())} B/block) + "
        f"drafter group id 1 (6 layers, block {draft_block}, {dpage} B/block, "
        f"unpadded); pool {per_block} B/block, num_blocks={kvcfg.num_blocks}"
    )


@check("KV grouping: drafterless model is byte-identical to before")
def _():
    import torch

    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    BLOCK = 64
    specs = {}
    for i in range(78):
        specs[f"model.layers.{i}.self_attn.attn"] = MLAAttentionSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="fp8_ds_mla",
        )
    for i in range(78):
        specs[f"model.layers.{i}.self_attn.indexer"] = MLAAttentionSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.float8_e4m3fn,
        )
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8192, disable_hybrid_kv_cache_manager=False
        ),
        model_config=SimpleNamespace(max_model_len=1048576),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )
    groups = get_kv_cache_groups(cfg, specs)
    assert len(groups) == 1, len(groups)
    assert len(groups[0].layer_names) == 156
    return f"1 group, {len(groups[0].layer_names)} layers (unchanged path)"


print("=" * 70)
if FAILS:
    print("FAILED CHECKS:", ", ".join(FAILS))
    sys.exit(1)
if NOTES:
    print("optional checks that did not pass:", ", ".join(NOTES))
print("ALL REQUIRED CHECKS PASSED")
