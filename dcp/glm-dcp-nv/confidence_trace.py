# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostic ownership for the previous DFlash2 C1 greedy proposal.

No cap selection or sampling changes. GPU values are never inspected on the
host until AsyncOutput's existing copy event has completed.
"""
import math
import os

import torch

PACKET_LIMIT = 128
PACKET_VERSION = 1


def enabled(config, tp_rank):
    spec = config.speculative_config
    return bool(
        os.environ.get("GLM_SPEC_CONFIDENCE_TRACE") == "1"
        and tp_rank == 0 and config.use_v2_model_runner
        and spec is not None and spec.method == "dflash"
        and config.num_speculative_tokens == 7
        and getattr(spec, "rejection_sample_method", "standard") != "synthetic"
        and config.parallel_config.pipeline_parallel_size == 1
        and config.parallel_config.data_parallel_size == 1
        and config.scheduler_config.async_scheduling
    )


def request_eligible(data):
    sp = data.sampling_params
    if sp is None:
        return False
    xargs = getattr(sp, "extra_args", None) or {}
    capture = xargs.get("spec_confidence_trace", True)
    return bool(
        sp.temperature == 0 and type(capture) in (bool, int) and capture == 1
        and not getattr(sp, "structured_outputs", None)
        and not getattr(data, "resumable", False)
        and not getattr(data, "lora_request", None)
        and not getattr(data, "mm_features", None)
        and getattr(sp, "presence_penalty", 0) == 0
        and getattr(sp, "frequency_penalty", 0) == 0
        and getattr(sp, "repetition_penalty", 1) == 1
        and not getattr(sp, "allowed_token_ids", None)
        and not getattr(sp, "bad_words", None)
        and not getattr(sp, "logit_bias", None)
    )


class ConfidenceCollector:
    def __init__(self):
        self.incarnation = 0
        self.worker_step = 0
        self.requests = {}
        self.previous = None

    def add(self, data):
        self.remove(data.req_id)
        self.incarnation += 1
        self.requests[data.req_id] = {
            "epoch": self.incarnation, "eligible": request_eligible(data),
            "serial": 0, "captured": 0,
        }
        # Even a one-row scheduled batch cannot inherit ownership across C2.
        self.previous = None

    def remove(self, rid):
        self.requests.pop(rid, None)
        if self.previous is not None and self.previous["request"] == rid:
            self.previous = None

    def _owner(self, batch):
        if len(self.requests) != 1 or batch.num_reqs != 1 or len(batch.req_ids) != 1:
            return None
        rid = batch.req_ids[0]
        state = self.requests.get(rid)
        if (state is None or not state["eligible"] or state["captured"] >= PACKET_LIMIT
                or batch.has_structured_output_reqs or bool(batch.is_prefilling_np[0])
                or batch.num_draft_tokens not in (1, 3, 5, 7)
                or batch.num_tokens != batch.num_draft_tokens + 1):
            return None
        return rid, state

    def capture(self, batch, speculator):
        """Clone only old proposal data, on the current main stream."""
        self.worker_step += 1
        owned = self._owner(batch)
        previous = self.previous
        if owned is None:
            self.previous = None
            return None
        rid, state = owned
        if (previous is None or previous["request"] != rid
                or previous["epoch"] != state["epoch"]
                or previous["worker_step"] != self.worker_step - 1):
            self.previous = None
            return None
        scores = getattr(speculator, "_selector_scores", None)
        sample_pos = getattr(speculator, "sample_pos", None)
        drafts = getattr(speculator, "draft_tokens", None)
        if (scores is None or scores.ndim != 3 or tuple(scores.shape[1:]) != (7, 16)
                or scores.shape[0] < 1 or scores.dtype != torch.float32
                or sample_pos is None or sample_pos.ndim != 1 or sample_pos.shape[0] < 7
                or sample_pos.dtype != torch.int64
                or drafts is None or drafts.ndim != 2 or drafts.shape[0] < 1
                or drafts.shape[1] != 7 or drafts.dtype != torch.int64
                or batch.positions.ndim != 1 or batch.positions.dtype != torch.int64
                or batch.input_ids.ndim != 1 or batch.input_ids.dtype != torch.int32
                or min(batch.positions.shape[0], batch.input_ids.shape[0]) < batch.num_tokens):
            self.previous = None
            return None
        n = batch.num_tokens
        tensors = {
            "realized_scores": scores[0, :7, :16].clone(),
            "sample_positions": sample_pos[:7].clone(),
            "draft_tokens": drafts[0, :7].clone(),
            "target_positions": batch.positions[:n].clone(),
            "target_tokens": batch.input_ids[:n].clone(),
        }
        state["captured"] += 1
        return {"request": rid, "meta": {
            "version": PACKET_VERSION, "epoch": state["epoch"],
            "proposal_id": previous["proposal_id"],
            "proposal_worker_step": previous["worker_step"],
            "verified_worker_step": self.worker_step,
            "proposal_age_steps": self.worker_step - previous["worker_step"],
            "verified_k": batch.num_draft_tokens,
        }, "tensors": tensors}

    def proposed(self, batch):
        """Record enqueue ownership after propose; no GPU or host-data read."""
        owned = self._owner(batch)
        if owned is None:
            self.previous = None
            return
        rid, state = owned
        state["serial"] += 1
        self.previous = {"request": rid, "epoch": state["epoch"],
                         "proposal_id": state["serial"], "worker_step": self.worker_step}


def packet_to_host(packet):
    """Run inside the existing copy stream, before its existing event record.

    Explicitly pinned destinations avoid relying on pageable .to(cpu) overlap.
    The AsyncOutput keeps both source clones and destination tensors alive.
    """
    values = {}
    for key, tensor in packet["tensors"].items():
        target = torch.empty_like(tensor, device="cpu", pin_memory=True)
        target.copy_(tensor, non_blocking=True)
        values[key] = target
    return {"request": packet["request"], "meta": dict(packet["meta"]),
            "tensors": values}


def finish_packet(packet):
    """CPU only, called after the existing event; reject diagnostic mismatches."""
    row = {**packet["meta"], **{k: v.tolist() for k, v in packet["tensors"].items()}}
    reason = None
    k = row["verified_k"]
    positions, target = row["target_positions"], row["target_tokens"]
    sample, drafts, scores = row["sample_positions"], row["draft_tokens"], row["realized_scores"]
    if (k not in (1, 3, 5, 7) or len(positions) != k + 1 or len(target) != k + 1
            or len(sample) != 7 or len(drafts) != 7
            or len(scores) != 7 or any(len(values) != 16 for values in scores)):
        reason = "shape_mismatch"
    elif any(not math.isfinite(v) for values in scores for v in values):
        reason = "nonfinite_score"
    elif (positions[0] < 0 or sample[0] - 1 != positions[0]
            or positions != list(range(positions[0], positions[0] + k + 1))
            or sample != list(range(positions[0] + 1, positions[0] + 8))):
        reason = "anchor_or_position_mismatch"
    elif target[1:] != drafts[:k]:
        reason = "verified_draft_token_mismatch"
    row["valid"] = reason is None
    row["invalid_reason"] = reason
    if reason is None:
        row["anchor"] = positions[0]
    else:
        # Do not emit invalid/nonfinite scores into JSON or calibration data.
        row.pop("realized_scores", None)
    return {packet["request"]: row}
