# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p

NO_LOGPROBS = -1
_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max

# GLM overlay: per-request bounded-lossy greedy verification controls
# (docs/LOSSY-VERIFICATION-PLAN.md). Honoured only when the boot sets
# GLM_SPEC_LOSSY=1; otherwise every request keeps exact verification.
LOSSY_ENABLED = os.environ.get("GLM_SPEC_LOSSY") == "1"
LOSSY_OFF = -1.0
LOSSY_MAX_MARGIN = 5.0
LOSSY_MAX_MIN_P = 0.5
LOSSY_SCOPES = {"all": 0, "think": 1}
# <think> / </think> ids of the served checkpoint; a request whose committed
# stream is inside a think span may relax under scope "think" only there.
LOSSY_THINK_IDS = tuple(
    int(x) for x in os.environ.get("GLM_SPEC_LOSSY_THINK_IDS", "154841,154842").split(",")
)


def lossy_scope(sampling_params: SamplingParams) -> int:
    """0 = relax anywhere the margin allows, 1 = only inside think spans."""
    xargs = getattr(sampling_params, "extra_args", None) or {}
    return LOSSY_SCOPES.get(xargs.get("spec_lossy_scope", "all"), 0)


def advance_think_state(
    think_state: torch.Tensor,        # [max_num_reqs + 1] int32, device view (slot -1 = scratch)
    idx_mapping: torch.Tensor,        # [num_reqs] request slots, -1 for masked rows
    sampled_token_ids: torch.Tensor,  # [num_reqs, 1]
    think_ids=LOSSY_THINK_IDS,
) -> None:
    """Track think spans over tokens the plain sampler emitted. Batches with no
    draft tokens (the first token after a prefill at C1, every step while the
    drafter is idle) bypass the rejection kernels, whose bookkeeping otherwise
    keeps the state; a `</think>` emitted there would leave the request
    "inside" the template-opened span for its whole output, relaxing tool
    calls and code under the think scope. No host sync: masked rows write the
    scratch slot."""
    open_id, close_id = think_ids
    tok = sampled_token_ids[:, 0].to(torch.int64)
    slot = torch.where(idx_mapping >= 0, idx_mapping, think_state.shape[0] - 1)
    cur = think_state[slot]
    new = torch.where(tok == open_id, torch.ones_like(cur),
                      torch.where(tok == close_id, torch.zeros_like(cur), cur))
    think_state[slot] = new


def initial_think_state(token_ids, think_ids=LOSSY_THINK_IDS) -> int:
    """1 if the last think marker among the trailing tokens is an opening one.

    The chat template ends a thinking-enabled prompt with `<think>` and a
    thinking-disabled one with `<think></think>`, so the generation starts
    inside or outside a span accordingly; re-added requests carry their
    generated tokens too.
    """
    open_id, close_id = think_ids
    for tok in reversed(list(token_ids or [])[-64:]):
        if tok == open_id:
            return 1
        if tok == close_id:
            return 0
    return 0


def _number(value):
    """A finite JSON number (not a bool, which the wire may turn into an int)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def lossy_controls(sampling_params: SamplingParams) -> tuple[float, float]:
    """(margin in nats, log p_min) for this request, or (off, -inf).

    Fails closed: any invalid, out-of-range or ineligible combination leaves
    the request on exact verification. Greedy only; no structured outputs,
    logit bias, allowed/bad token lists or penalties, mirroring the adaptive
    policy's eligibility rule.
    """
    off = (LOSSY_OFF, float("-inf"))
    xargs = getattr(sampling_params, "extra_args", None) or {}
    margin = _number(xargs.get("spec_lossy_margin"))
    if margin is None or not 0.0 < margin <= LOSSY_MAX_MARGIN:
        return off
    rank = xargs.get("spec_lossy_rank", 2)
    if isinstance(rank, bool) or rank != 2:
        return off
    min_p = _number(xargs.get("spec_lossy_min_p", 0))
    if min_p is None or not 0.0 <= min_p < LOSSY_MAX_MIN_P:
        return off
    sp = sampling_params
    if (sp.temperature != 0.0 or getattr(sp, "structured_outputs", None) is not None
            or getattr(sp, "logit_bias", None) or getattr(sp, "allowed_token_ids", None)
            or getattr(sp, "bad_words", None)
            or getattr(sp, "presence_penalty", 0) != 0
            or getattr(sp, "frequency_penalty", 0) != 0
            or getattr(sp, "repetition_penalty", 1) != 1):
        return off
    return margin, (math.log(min_p) if min_p > 0.0 else float("-inf"))


class SamplingStates:
    def __init__(self, max_num_reqs: int, vocab_size: int):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        self.temperature = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.top_k = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.top_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.min_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.seeds = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        # Tracks whether `seed` was set explicitly by the user, so callers
        # can fall back from RNG paths that don't honor per-request seeds.
        self.seeds_set = np.zeros(max_num_reqs, dtype=bool)
        # Bounded-lossy greedy verification: margin (< 0 = off) and log p_min.
        self.lossy_margin = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.lossy_margin.np.fill(LOSSY_OFF)
        self.lossy_margin.copy_to_uva()
        self.lossy_min_logp = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.lossy_min_logp.np.fill(float("-inf"))
        self.lossy_min_logp.copy_to_uva()
        self.lossy_scope = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.lossy_scope.copy_to_uva()
        # Think-span state lives on the GPU (the rejection kernels update it);
        # the host writes only the initial value from the prompt at admission.
        # One extra scratch slot absorbs masked rows in advance_think_state.
        self.think_state = UvaBackedTensor(max_num_reqs + 1, dtype=torch.int32)
        self.think_state.copy_to_uva()

        # Initialize top_k and top_p manually because 0 is an invalid value for them.
        self.top_k.np.fill(self.vocab_size)
        self.top_k.copy_to_uva()
        self.top_p.np.fill(1.0)
        self.top_p.copy_to_uva()

        self.num_logprobs = np.empty(self.max_num_reqs, dtype=np.int32)
        # -1 means no logprobs are requested.
        self.num_logprobs.fill(NO_LOGPROBS)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        self.temperature.np[req_idx] = sampling_params.temperature
        self.top_p.np[req_idx] = sampling_params.top_p
        top_k = sampling_params.top_k
        if top_k <= 0 or top_k > self.vocab_size:
            top_k = self.vocab_size
        self.top_k.np[req_idx] = top_k
        self.min_p.np[req_idx] = sampling_params.min_p

        seed = sampling_params.seed
        self.seeds_set[req_idx] = seed is not None
        if seed is None:
            seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
        self.seeds.np[req_idx] = seed

        margin, min_logp = (lossy_controls(sampling_params) if LOSSY_ENABLED
                            else (LOSSY_OFF, float("-inf")))
        self.lossy_margin.np[req_idx] = margin
        self.lossy_min_logp.np[req_idx] = min_logp
        self.lossy_scope.np[req_idx] = lossy_scope(sampling_params) if margin >= 0 else 0
        self.think_state.np[req_idx] = 0

        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = NO_LOGPROBS
        elif num_logprobs == -1:
            num_logprobs = self.vocab_size
        self.num_logprobs[req_idx] = num_logprobs

    def apply_staged_writes(self) -> None:
        self.temperature.copy_to_uva()
        self.top_p.copy_to_uva()
        self.top_k.copy_to_uva()
        self.min_p.copy_to_uva()
        self.seeds.copy_to_uva()
        self.lossy_margin.copy_to_uva()
        self.lossy_min_logp.copy_to_uva()
        self.lossy_scope.copy_to_uva()
        self.think_state.copy_to_uva()

    def set_think_state(self, req_idx: int, token_ids) -> None:
        """Initial think-span state for a newly admitted request (host side;
        copied with the other staged writes)."""
        self.think_state.np[req_idx] = initial_think_state(token_ids)

    def update_think_state_from_sampled(
        self, idx_mapping: torch.Tensor, sampled_token_ids: torch.Tensor
    ) -> None:
        """Plain-sampler path (no draft tokens in the batch): keep the think
        state in step with the emitted token. No-op unless lossy is enabled."""
        if not LOSSY_ENABLED:
            return
        advance_think_state(self.think_state.gpu, idx_mapping, sampled_token_ids)

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        temp_np = self.temperature.np[idx_mapping_np]
        if np.all((temp_np == 0.0) | (temp_np == 1.0)):
            # No request requires temperature. Skip the kernel launch.
            return

        apply_temperature(logits, expanded_idx_mapping, self.temperature.gpu)

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if np.all(self.min_p.np[idx_mapping_np] == 0.0):
            # No request uses min_p. Skip the kernel launch.
            return
        apply_min_p(logits, expanded_idx_mapping, self.min_p.gpu)

    def get_top_k_top_p(
        self, expanded_idx_mapping: torch.Tensor, idx_mapping_np: np.ndarray
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        do_top_k = np.any(self.top_k.np[idx_mapping_np] != self.vocab_size)
        do_top_p = np.any(self.top_p.np[idx_mapping_np] != 1.0)
        top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None
        top_p = self.top_p.gpu[expanded_idx_mapping] if do_top_p else None
        return top_k, top_p

    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        top_k, top_p = self.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        if top_k is None and top_p is None:
            return logits
        return apply_top_k_top_p(logits, top_k, top_p)

    def any_greedy(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.temperature.np[idx_mapping_np] == 0.0))

    def any_explicit_seed(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.seeds_set[idx_mapping_np]))

    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int:
        return int(np.max(self.num_logprobs[idx_mapping_np]))
