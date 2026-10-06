# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from vllm.config import SpeculativeConfig
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


# GLM overlay: bounded-lossy greedy verification (docs/LOSSY-VERIFICATION-PLAN.md).
# Token ids a relaxed accept may never emit or displace: EOS, role and think
# markers of the served checkpoint; overridable per boot.
DEFAULT_LOSSY_STOP_IDS = "154820,154827,154829,154841,154842,154828"


def lossy_think_ids() -> tuple[int, int]:
    raw = os.environ.get("GLM_SPEC_LOSSY_THINK_IDS", "154841,154842")
    a, b = (int(x) for x in raw.split(",")[:2])
    return a, b


def lossy_stop_ids(device: torch.device) -> torch.Tensor:
    raw = os.environ.get("GLM_SPEC_LOSSY_STOP_IDS", DEFAULT_LOSSY_STOP_IDS)
    ids = sorted({int(x) for x in raw.split(",") if x.strip()})
    return torch.tensor(ids, dtype=torch.int64, device=device)


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.lossy_stop_ids = lossy_stop_ids(device)
        self.lossy_think_ids = lossy_think_ids()
        # Trial-only rank-agreement proof: every 64th call all-gathers the
        # accepted counts across TP and fails loudly on any disagreement.
        self.lossy_check = os.environ.get("GLM_SPEC_LOSSY_CHECK") == "1"
        self.lossy_calls = 0
        self.rejection_sample_method = spec_config.rejection_sample_method
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if self.rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )

    def _get_logprobs_tensors(
        self,
        input_batch: InputBatch,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
    ) -> LogprobsTensors | None:
        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = input_batch.cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            input_batch.cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != input_batch.idx_mapping.shape[0]
        return compute_topk_logprobs(
            logits,
            max_num_logprobs,
            flat_sampled,
            input_batch.cu_num_logits_np.tolist() if expanded_logits else None,
        )

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            draft_sampled,
            input_batch.expanded_local_pos,
        )
        states = self.sampler.sampling_states
        lossy_margin = getattr(states, "lossy_margin", None)
        lossy_min_logp = getattr(states, "lossy_min_logp", None)
        sampled, num_sampled, num_relaxed = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            input_batch.cu_num_logits,
            pos,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            states.temperature.gpu,
            states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            lossy_margin=None if lossy_margin is None else lossy_margin.gpu,
            lossy_min_logp=None if lossy_min_logp is None else lossy_min_logp.gpu,
            stop_ids=self.lossy_stop_ids,
            lossy_scope=getattr(getattr(states, "lossy_scope", None), "gpu", None),
            think_state=getattr(getattr(states, "think_state", None), "gpu", None),
            think_ids=self.lossy_think_ids,
        )
        if self.lossy_check:
            self.lossy_calls += 1
            if self.lossy_calls % 64 == 0:
                self._assert_rank_agreement(num_sampled)
        logprobs_tensors = self._get_logprobs_tensors(
            input_batch,
            sampled,
            num_sampled,
            processed_logits
            if self.sampler.logprobs_mode == "processed_logprobs"
            else logits,
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )

        output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        # Per-request relaxed accepts; carried to the scheduler's verify rows
        # by AsyncOutput. Optional attribute: the stock dataclass is unchanged.
        output.num_relaxed = num_relaxed
        return output

    @staticmethod
    def _assert_rank_agreement(num_sampled: torch.Tensor) -> None:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        if group.world_size == 1:
            return
        gathered = group.all_gather(num_sampled.unsqueeze(0), dim=0)
        if not bool((gathered == gathered[0]).all()):
            raise RuntimeError(
                "lossy verification rank disagreement: "
                f"{gathered.tolist()} (GLM_SPEC_LOSSY_CHECK)"
            )
