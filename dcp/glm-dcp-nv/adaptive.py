# SPDX-License-Identifier: Apache-2.0
"""C1 verification policy. Stdlib only; draft capacity and sampling stay unchanged.

The scheduler trims the *next* placeholder/proposal list before budgeting.
AsyncScheduler still creates seven placeholders and the V2 speculator still
writes seven GPU proposals. Existing scheduled-list lengths remain the sole
authority for target inputs, output placeholders and rejection rollback.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import queue
import threading
import time
import weakref
from dataclasses import dataclass, field

CAPS = (1, 3, 5, 7)
VERSION = 5
PRIOR_STRENGTH = 2.0


def validated_prior(values) -> tuple[float, ...] | None:
    if (isinstance(values, list) and len(values) == 7 and all(
        type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values
    )):
        return tuple(float(v) for v in values)
    return None


@dataclass(frozen=True)
class HintPriors:
    """Server-owned weak priors; client labels never supply probabilities."""
    bounds: tuple[int, int]
    domains: dict[str, tuple[float, ...]]

    @classmethod
    def parse(cls, data, signature):
        try:
            bounds = data['context_range']
            raw = data['domains']
            if (data['config'] != signature or data.get('prior_strength') != PRIOR_STRENGTH
                    or not isinstance(bounds, list) or len(bounds) != 2
                    or any(type(x) is not int for x in bounds)
                    or not 0 <= bounds[0] < bounds[1] <= signature['max_model_len']
                    or not isinstance(raw, dict) or set(raw) != {'code', 'prose'}):
                raise ValueError
            domains = {name: validated_prior(values) for name, values in raw.items()}
            if any(values is None for values in domains.values()):
                raise ValueError
            return cls(tuple(bounds), domains)
        except (KeyError, TypeError, ValueError):
            raise ValueError('Invalid server workload prior table') from None

    def select(self, xargs, context):
        # The OpenAI vllm_xargs schema accepts numeric/string primitives and
        # normalizes JSON booleans to integer 0/1 before SamplingParams.
        enabled = xargs.get('spec_use_hints', True)
        if (not self.bounds[0] <= context < self.bounds[1]
                or type(enabled) not in (bool, int) or enabled != 1
                or xargs.get('spec_hint_strength') != 'weak'
                or xargs.get('spec_phase') != 'user_turn'):
            return None, None
        workload = xargs.get('spec_workload')
        domain = ('code' if workload in ('code_generate', 'code_edit', 'code_review')
                  else 'prose' if workload == 'prose' else None)
        return (self.domains[domain], domain) if domain else (None, None)


@dataclass(frozen=True)
class CostCurve:
    """Bounded interpolation of measured context points; no GPU work or I/O."""
    bounds: tuple[int, int]
    points: tuple

    @classmethod
    def parse(cls, data, max_model_len):
        try:
            bounds = data['context_range']
            rows = data['calibration_points']
            if (not isinstance(bounds,list) or len(bounds)!=2
                    or not all(type(v) is int for v in bounds)
                    or not 0<=bounds[0]<bounds[1]<=max_model_len
                    or not isinstance(rows,list) or not 1<=len(rows)<=8):
                raise ValueError
            points = []
            for row in rows:
                x = row['context']
                raw = row['cycle_ms']
                if (type(x) is not int or not bounds[0]<=x<bounds[1]
                        or (points and x<=points[-1][0]) or not isinstance(raw,dict)
                        or len(raw)!=len(CAPS) or any(type(v) not in (int,float) for v in raw.values())):
                    raise ValueError
                costs = {int(k):float(v) for k,v in raw.items()}
                prior = validated_prior(row.get('conditional_acceptance_prior'))
                if (set(costs)!=set(CAPS) or not all(math.isfinite(v) and v>0 for v in costs.values())
                        or ('conditional_acceptance_prior' in row and prior is None)
                        or row.get('prior_strength',PRIOR_STRENGTH)!=PRIOR_STRENGTH):
                    raise ValueError
                points.append((x,costs,prior))
            return cls(tuple(bounds),tuple(points))
        except (KeyError,TypeError,ValueError,OverflowError):
            raise ValueError('Invalid context cost curve') from None

    def at(self, context):
        if not self.bounds[0]<=context<self.bounds[1]:
            return {},None
        if context<=self.points[0][0]:
            return self.points[0][1].copy(),self.points[0][2]
        for left,right in zip(self.points,self.points[1:]):
            if context<=right[0]:
                weight=(context-left[0])/(right[0]-left[0])
                costs={k:left[1][k]+weight*(right[1][k]-left[1][k]) for k in CAPS}
                prior=(tuple(a+weight*(b-a) for a,b in zip(left[2],right[2]))
                       if left[2] is not None and right[2] is not None else None)
                return costs,prior
        return self.points[-1][1].copy(),self.points[-1][2]


def policy_mode() -> str:
    mode = os.environ.get("GLM_SPEC_POLICY", "off")
    if mode not in ("off", "shadow", "fixed", "adaptive"):
        raise ValueError(f"Invalid GLM_SPEC_POLICY: {mode}")
    return mode


def extra_capture_sizes(is_target: bool) -> tuple[int, ...]:
    return (2, 4, 6) if is_target and policy_mode() != "off" else ()


def lossy_request_fields(xargs) -> dict:
    """Echo the bounded-lossy controls a request asked for (audit only).

    The worker decides whether they are honoured (GLM_SPEC_LOSSY=1, greedy,
    valid ranges); rows carry both the request and the boot switch so a
    report can prove activation or its absence.
    """
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(value) else None
    return {
        "lossy_margin": number(xargs.get("spec_lossy_margin")),
        "lossy_min_p": number(xargs.get("spec_lossy_min_p", 0)),
        "lossy_scope": xargs.get("spec_lossy_scope", "all") if xargs.get("spec_lossy_scope", "all") in ("all", "think") else "invalid",
        "lossy_enabled": os.environ.get("GLM_SPEC_LOSSY") == "1",
    }


def eligible(request, c1: bool) -> bool:
    sp = request.sampling_params
    return bool(
        c1 and sp is not None and sp.temperature == 0
        and not request.is_prefill_chunk
        and not request.use_structured_output
        and not getattr(request, "resumable", False)
        and not getattr(request, "lora_request", None)
        and not getattr(request, "mm_features", None)
        and not getattr(request, "async_tokens_to_discard", 0)
        and getattr(sp, "presence_penalty", 0) == 0
        and getattr(sp, "frequency_penalty", 0) == 0
        and getattr(sp, "repetition_penalty", 1) == 1
        and not getattr(sp, "allowed_token_ids", None)
        and not getattr(sp, "bad_words", None)
        and not getattr(sp, "logit_bias", None)
    )


@dataclass
class PrefixStats:
    # Conditional acceptance risk sets. A position is at risk only when its
    # predecessors passed AND it was actually verified. Multiplying these
    # conditional rates handles right censoring without selection bias.
    successes: list[float] = field(default_factory=lambda: [0.0] * 7)
    trials: list[float] = field(default_factory=lambda: [0.0] * 7)
    observations: int = 0
    cap: int = 7
    prior: tuple[float, ...] | None = None

    def observe(self, k: int, accepted: int, decay: float = 0.97) -> None:
        if not 1 <= k <= 7 or not 0 <= accepted <= k:
            raise ValueError("invalid accepted-prefix observation")
        for j in range(7):
            self.successes[j] *= decay
            self.trials[j] *= decay
            if j < k and accepted >= j:
                self.trials[j] += 1
                self.successes[j] += int(j < accepted)
        self.observations += 1

    def expected(self, k: int, allow_prior_only: bool = False) -> float:
        survival = 1.0
        total = 1.0
        for j in range(k):
            if j == 0 and self.trials[j] < 2 and not (allow_prior_only and self.prior is not None):
                return float("nan")
            # With inadequate tail evidence use its optimistic bound. This
            # favors full verification/exploration, not premature shortening.
            if self.prior is not None:
                conditional = ((self.successes[j] + PRIOR_STRENGTH * self.prior[j])
                               / (self.trials[j] + PRIOR_STRENGTH))
            else:
                conditional = (self.successes[j] / self.trials[j]
                               if self.trials[j] >= 2 else 1.0)
            survival *= conditional
            total += survival
        return total

    def choose(self, costs: dict[int, float], scheduled_steps: int,
               cold_start: bool = False) -> tuple[int, bool]:
        probe = scheduled_steps % 16 == 0
        minimum = 0 if cold_start and self.prior is not None else 8
        if self.observations < minimum or probe or set(costs) != set(CAPS):
            return 7, probe
        rates = {k: self.expected(k, allow_prior_only=cold_start) / costs[k] for k in CAPS}
        if not all(math.isfinite(r) and r > 0 for r in rates.values()):
            return 7, probe
        best = max(CAPS, key=lambda k: (rates[k], k))
        if rates[best] > rates[self.cap] * 1.03:
            self.cap = best
        return self.cap, probe


class JsonlSink:
    """Bounded background writer: inference never waits on filesystem I/O."""
    def __init__(self, path: str):
        self.path = path
        self.queue = queue.Queue(maxsize=4096)
        self.dropped = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def emit(self, row: dict) -> None:
        if self.error or self.closed:
            self.dropped += 1
            return
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _write(self) -> None:
        try:
            with open(self.path, "a", buffering=65536) as out:
                dirty = False
                while True:
                    try:
                        row = self.queue.get(timeout=1 if dirty else None)
                    except queue.Empty:
                        out.flush()
                        dirty = False
                        continue
                    if row is None:
                        break
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                    dirty = True
                out.write(json.dumps({"event": "writer_close", "dropped": self.dropped}) + "\n")
        except Exception as exc:
            self.error = repr(exc)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=1)
                self.thread.join(timeout=2)
            except queue.Full:
                pass


class VerificationPolicy:
    def __init__(self, config):
        self.mode = policy_mode()
        self.enabled = self.mode != "off"
        self.pending = {}
        self.chosen = {}
        self.states = weakref.WeakKeyDictionary()
        self.costs = {}
        self.prior = None
        self.curve = None
        self.hint_priors = None
        self.cost_context = (0, 0)
        self.lane = (
            f"tp{config.parallel_config.tensor_parallel_size}"
            f"-dcp{config.parallel_config.decode_context_parallel_size}"
            f"-l{config.model_config.max_model_len}-k7"
        )
        self.signature = {'tp':config.parallel_config.tensor_parallel_size,
                          'dcp':config.parallel_config.decode_context_parallel_size,
                          'max_model_len':config.model_config.max_model_len,'draft_capacity':7}
        self.sink = None
        self.fixed = int(os.environ.get("GLM_SPEC_VERIFY_CAP", "7"))
        if self.fixed not in CAPS:
            raise ValueError("GLM_SPEC_VERIFY_CAP must be 1, 3, 5 or 7")
        if not self.enabled:
            return
        if not (
            config.use_v2_model_runner and config.num_speculative_tokens == 7
            and config.speculative_config.method == "dflash"
            and getattr(config.speculative_config, "rejection_sample_method", "standard") != "synthetic"
            and config.parallel_config.pipeline_parallel_size == 1
            and config.parallel_config.data_parallel_size == 1
            and config.scheduler_config.async_scheduling
        ):
            raise ValueError("Adaptive verification requires async V2 DFlash K=7, PP1, DP1 and real rejection sampling")
        path = os.environ.get("GLM_SPEC_COSTS", "")
        if path:
            with open(path) as f:
                data = json.load(f)
            expected = {
                "tp": config.parallel_config.tensor_parallel_size,
                "dcp": config.parallel_config.decode_context_parallel_size,
                "max_model_len": config.model_config.max_model_len,
                "draft_capacity": 7,
            }
            if data.get("config") != expected:
                raise ValueError("Speculation cost table does not match this lane")
            if 'calibration_points' in data:
                self.curve = CostCurve.parse(data,config.model_config.max_model_len)
                # The first point also supplies the startup diagnostic fields.
                first = data['calibration_points'][0]
                data = {**data,**{k:first[k] for k in ('cycle_ms','conditional_acceptance_prior','prior_strength') if k in first}}
            self.costs = {int(k): float(v) for k, v in data["cycle_ms"].items()}
            bounds = data.get("context_range")
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or any(type(x) is not int for x in bounds)
                    or not 0 <= bounds[0] < bounds[1] <= config.model_config.max_model_len):
                raise ValueError("Speculation cost table needs a calibrated context_range")
            self.cost_context = tuple(bounds)
            self.prior = validated_prior(data.get('conditional_acceptance_prior'))
            if 'conditional_acceptance_prior' in data and self.prior is None:
                raise ValueError('Invalid conditional acceptance prior')
            if self.prior is not None and data.get('prior_strength', PRIOR_STRENGTH) != PRIOR_STRENGTH:
                raise ValueError('Unsupported acceptance prior strength')
            if set(self.costs) != set(CAPS) or not all(
                math.isfinite(v) and v > 0 for v in self.costs.values()
            ):
                raise ValueError("Invalid speculation cycle cost table")
        path = os.environ.get("GLM_SPEC_HINT_PRIORS", "")
        if path:
            with open(path) as f:
                raw = f.read(16385)
            if len(raw) > 16384:
                raise ValueError('Server workload prior table exceeds 16 KiB')
            self.hint_priors = HintPriors.parse(json.loads(raw), self.signature)
        path = os.environ.get("GLM_SPEC_TRACE", "")
        if path:
            self.sink = JsonlSink(path)
            self.sink.emit({"event": "policy_start", "version": VERSION,
                            "mode": self.mode, "fixed": self.fixed,
                            "costs_ms": self.costs, "acceptance_prior": self.prior,
                            "workload_hints_enabled": self.hint_priors is not None,
                            "confidence_trace_enabled": os.environ.get("GLM_SPEC_CONFIDENCE_TRACE") == "1",
                            "time": time.time()})

    def begin(self) -> None:
        self.chosen.clear()

    def select(self, request, c1: bool, step: int) -> int:
        decision_ns = time.monotonic_ns()
        ok = eligible(request, c1)
        mode = self.mode
        cap, probe = 7, False
        xargs = getattr(request.sampling_params, "extra_args", None) or {}
        state = self.states.setdefault(request, {"stats": PrefixStats(), "scheduled": 0})
        context = request.num_tokens + getattr(request, "num_output_placeholders", 0)
        costs = self.costs if self.cost_context[0] <= context < self.cost_context[1] else {}
        prior = self.prior if costs else None
        if self.curve is not None:
            costs,prior = self.curve.at(context)
        curve_source = xargs.get('spec_cost_table')
        if isinstance(curve_source,(dict,str)) and state.get('curve_source') is not curve_source:
            # Request controls are immutable after admission; parse once per
            # request incarnation instead of on each decode cycle.
            state['curve_source'],state['curve'] = curve_source,None
            try:
                # OpenAI vllm_xargs permits scalars and flat lists, not nested
                # objects. Bound the encoded table before parsing it once.
                curve_data = (json.loads(curve_source) if isinstance(curve_source,str)
                              and len(curve_source)<=16384 else curve_source)
                if (isinstance(curve_data,dict) and curve_data.get('config')==self.signature
                        and curve_data.get('lane')==self.lane):
                    state['curve'] = CostCurve.parse(curve_data,self.signature['max_model_len'])
            except (ValueError,RecursionError):
                pass
        if (isinstance(curve_source,(dict,str)) and state.get('curve_source') is curve_source
                and state.get('curve') is not None):
            proposed_costs,proposed_prior = state['curve'].at(context)
            if proposed_costs:
                costs,prior = proposed_costs,proposed_prior
        # Experimental request-scoped calibration permits a measured A/B in
        # the same boot. No file I/O or GPU synchronization on the decode path.
        # Invalid/mismatched controls fail closed to the boot's cost table.
        values = xargs.get("spec_cycle_ms")
        bounds = xargs.get("spec_cost_context")
        if (xargs.get("spec_cost_lane") == self.lane and isinstance(values, list)
                and len(values) == len(CAPS)
                and isinstance(bounds, list) and len(bounds) == 2
                and all(type(v) is int for v in bounds) and 0 <= bounds[0] <= context < bounds[1]
                and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values)):
            costs = dict(zip(CAPS, values))
            prior = validated_prior(xargs.get('spec_acceptance_prior'))
        # Per-request controls let one experimental boot compare all cap sizes.
        if xargs.get("spec_policy") in ("off", "shadow", "fixed", "adaptive"):
            mode = xargs["spec_policy"]
        hint_domain = None
        if (ok and mode == 'adaptive' and costs and self.hint_priors is not None
                and state['stats'].observations < 8):
            # Labels only shorten the initial warm-up. Once actual feedback
            # is sufficient, return to the ordinary inference-only controller.
            hint_prior, hint_domain = self.hint_priors.select(xargs, context)
            if hint_prior is not None:
                prior = hint_prior
        state['stats'].prior = prior
        if ok:
            state["scheduled"] += 1
            if mode == "fixed":
                value = xargs.get("spec_verify_cap", self.fixed)
                cap = value if type(value) is int and value in CAPS else 7
            elif mode == "adaptive":
                cap, probe = state["stats"].choose(
                    costs, state["scheduled"], cold_start=hint_domain is not None)
        self.chosen[request.request_id] = {
            "request": weakref.ref(request), "step": step, "cap": cap,
            "eligible": ok, "mode": mode, "probe": probe,
            "observations": state["stats"].observations,
            "costs_ms": costs,
            "acceptance_prior": prior,
            "hint_domain": hint_domain,
            "label": str(xargs.get("spec_label", ""))[:96],
            "decision_ns": decision_ns,
            **lossy_request_fields(xargs),
        }
        return cap

    def scheduled(self, output) -> None:
        rows = {}
        for rid, tokens in output.scheduled_spec_decode_tokens.items():
            if rid in self.chosen:
                row = self.chosen[rid].copy()
                row["scheduled_k"] = len(tokens)
                row["target_tokens"] = output.num_scheduled_tokens[rid]
                row["scheduled_at"] = time.monotonic()
                rows[rid] = row
        if rows:
            if len(self.pending) >= 64:
                raise RuntimeError("Adaptive verification feedback queue exceeded 64 steps")
            self.pending[id(output)] = rows

    def complete(self, output, runner_output, requests) -> None:
        receipt_ns = time.monotonic_ns()
        rows = self.pending.pop(id(output), {})
        kv_output = getattr(runner_output, "kv_connector_output", None)
        if kv_output and getattr(kv_output, "invalid_block_ids", None):
            return
        now = time.monotonic()
        for rid, row in rows.items():
            request = row.pop("request")()
            if request is None or requests.get(rid) is not request or request.is_finished():
                continue
            state = self.states.get(request)
            if state is None or request.async_tokens_to_discard:
                continue
            i = runner_output.req_id_to_index.get(rid)
            if i is None or not runner_output.sampled_token_ids:
                continue
            tokens = runner_output.sampled_token_ids[i]
            k = row["scheduled_k"]
            if not tokens or not 1 <= k <= 7 or not 1 <= len(tokens) <= k + 1:
                continue
            accepted = len(tokens) - 1
            sp = request.sampling_params
            terminal = (
                request.num_output_tokens + len(tokens) >= request.max_tokens
                or (not getattr(sp, "ignore_eos", False) and
                    getattr(request, "eos_token_id", None) in tokens)
                or any(t in tokens for t in (getattr(sp, "stop_token_ids", None) or []))
                or (not getattr(sp, "ignore_eos", False) and any(
                    t in tokens for t in (getattr(sp, "all_stop_token_ids", None) or [])
                ))
                or bool(getattr(sp, "stop", None))
            )
            valid = row["eligible"] and not terminal and k == row["cap"]
            if valid:
                state["stats"].observe(k, accepted)
            previous = state.get("completed_at")
            state["completed_at"] = now
            if self.sink:
                confidence = (getattr(runner_output, "spec_confidence", None) or {}).get(rid)
                if confidence is not None:
                    confidence = dict(confidence)
                    if (confidence.get("valid") and
                            (confidence.get("verified_k") != k or
                             tokens[:accepted] != confidence.get("draft_tokens", [])[:accepted])):
                        confidence.update(valid=False, invalid_reason="acceptance_join_mismatch")
                        confidence.pop("realized_scores", None)
                    confidence["learnable"] = bool(valid and confidence.get("valid"))
                self.sink.emit({
                    "event": "verify", "version": VERSION,
                    "request": hashlib.sha256(rid.encode()).hexdigest()[:16],
                    "time": time.time(), **row,
                    "accepted": accepted, "sampled": len(tokens),
                    "context": request.num_tokens, "terminal": terminal,
                    "learned": valid, "cycle_ms": (now - previous) * 1000 if previous else None,
                    "relaxed": int((getattr(runner_output, "spec_relaxed", None) or {}).get(rid, 0)),
                    "latency_ms": (now - row["scheduled_at"]) * 1000,
                    "dropped": self.sink.dropped, "writer_error": self.sink.error,
                    "receipt_ns": receipt_ns,
                    **({"confidence": confidence} if confidence is not None else {}),
                })
