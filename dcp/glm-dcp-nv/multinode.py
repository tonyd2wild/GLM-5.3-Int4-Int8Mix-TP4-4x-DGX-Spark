# SPDX-License-Identifier: Apache-2.0
"""Multi-node tiering for the native OffloadingConnector.

The fork's ``TieringOffloadingSpec`` keeps the CPU tier in one ``/dev/shm``
mmap shared by the scheduler and every worker, and runs secondary-tier I/O
(the filesystem tier) inside the scheduler process over that mmap. That is
single-host by construction: with one worker per node, only rank 0's shard
would ever reach disk and the other ranks would load stale pages.

This module keeps the scheduler-side tiering *bookkeeping* (which keys live
in which tier, promotions, ref counts: ``TieringOffloadingManager``) and moves
the *bytes* to where they already are:

- the CPU tier is the per-worker pinned buffer of the single-tier spec (no
  shared mmap), so each rank keeps its own shard;
- the filesystem tier's store/load I/O runs in each worker, against its own
  local disk, as ordinary connector transfer jobs (``CPU <-> FS``) that the
  scheduler ships in the connector metadata and that complete when all
  workers have reported, exactly like the existing ``GPU <-> CPU`` jobs;
- the scheduler answers ``lookup`` from rank 0's own files, which is correct
  because a store job only completes when every rank has written its file.

Files: ``<root>/<model>_<digest>_r<global rank>/<hhh>/<hh>_g<group>/<hash>.bin``
(the documented ``FileMapper`` layout); each file holds this rank's canonical
tensor pages for the block, for the tensors its KV-cache group references.
Buffered I/O with ``fdatasync`` and ``FADV_DONTNEED`` (the indexer tensor's
page is not 512-byte aligned, so ``O_DIRECT`` is not an option here).

A failed load (missing or short file on any rank) completes the promotion
with ``success=False``; the tiering manager then does not mark the block
present in the CPU tier, the next lookup is a miss, and the tokens are
recomputed. Nothing is ever loaded into the GPU from a failed promotion.

Wiring (all out-of-tree, no patch to existing connector files):

    --kv-transfer-config '{
      "kv_connector": "MultiNodeOffloadingConnector",
      "kv_connector_module_path": "vllm.v1.kv_offload.tiering.multinode",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "spec_name": "MultiNodeTieringOffloadingSpec",
        "spec_module_path": "vllm.v1.kv_offload.tiering.multinode",
        "cpu_bytes_to_use": 4000000000,
        "secondary_tiers": [{"type": "fs_worker", "root_dir": "/kvcache",
                             "n_read_threads": 8, "n_write_threads": 8}]}}'
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import struct
import json
import os
import threading
import time
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.base import OffloadingSpec as OffloadingSpecBase
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput

logger = init_logger(__name__)

FS_WORKER_TIER_TYPE = "fs_worker"


# --------------------------------------------------------------------------
# Specs and metadata
# --------------------------------------------------------------------------


class FSLoadStoreSpec(LoadStoreSpec):
    """Blocks addressed by their offload keys in the per-rank file tier."""

    def __init__(self, keys: list[OffloadKey]):
        self.keys: list[OffloadKey] = list(keys)

    @staticmethod
    def medium() -> str:
        return "FS"

    def __repr__(self) -> str:
        return f"FS({len(self.keys)} keys)"


@dataclass
class MultiNodeConnectorMetadata(OffloadingConnectorMetadata):
    """Scheduler -> worker: the base jobs plus worker-executed tier jobs.

    Tier jobs are kept out of ``load_jobs``/``store_jobs`` on purpose: the
    base worker maps ``load_jobs`` to request ids and emits
    ``finished_recving`` for them, which tier jobs must never do.
    """

    tier_jobs: dict[int, TransferSpec] = field(default_factory=dict)


@dataclass
class MultiNodeWorkerMetadata(OffloadingWorkerMetadata):
    """Worker -> scheduler: completions plus the tier jobs that failed."""

    failed_jobs: set[int] = field(default_factory=set)

    def aggregate(self, other):  # type: ignore[override]
        merged = super().aggregate(other)
        return MultiNodeWorkerMetadata(
            completed_jobs=merged.completed_jobs,
            transfer_stats=merged.transfer_stats,
            failed_jobs=self.failed_jobs | set(getattr(other, "failed_jobs", ())),
        )


# --------------------------------------------------------------------------
# Worker side: file I/O against the worker's own pinned CPU tier
# --------------------------------------------------------------------------


def _tmp_suffix() -> str:
    return f".{os.getpid()}.{threading.get_ident()}.tmp"


def _drain_iov(views: list[memoryview], n: int) -> list[memoryview]:
    """Drop the first ``n`` bytes from a list of buffers (after a partial
    ``writev``/``readv``) and return what remains."""
    out = list(views)
    while out and n >= len(out[0]):
        n -= len(out[0])
        out.pop(0)
    if out and n > 0:
        out[0] = out[0][n:]
    return out


def write_block_file(path: str, views: list[memoryview]) -> None:
    """Write the concatenation of ``views`` to ``path`` atomically
    (temp file + rename), durably (``fdatasync``), and without leaving the
    data in the page cache. A file that already exists is left alone: the
    same key always carries the same bytes."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + _tmp_suffix()
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        try:
            remaining = [v.cast("B") for v in views]
            while remaining:
                n = os.writev(fd, remaining)
                if n <= 0:
                    raise OSError(f"writev returned {n} for {path}")
                remaining = _drain_iov(remaining, n)
            os.fdatasync(fd)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def read_block_file(path: str, views: list[memoryview]) -> None:
    """Read ``path`` into ``views`` (exact size required). An unreadable or
    wrong-sized file is removed so the next lookup misses instead of
    failing again."""
    expected = sum(len(v) for v in views)
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            size = os.fstat(fd).st_size
            if size != expected:
                raise OSError(f"{path}: size {size} != expected {expected}")
            remaining = [v.cast("B") for v in views]
            while remaining:
                n = os.readv(fd, remaining)
                if n <= 0:
                    raise OSError(f"{path}: short read")
                remaining = _drain_iov(remaining, n)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass
        finally:
            os.close(fd)
    except BaseException:
        try:
            os.remove(path)
        except OSError:
            pass
        raise


class WorkerFsHandler(OffloadingHandler):
    """One direction of CPU-tier <-> local-file transfers for this worker.

    ``cpu_tensors`` are the worker's pinned CPU-tier tensors, one per
    canonical KV tensor, shape ``(num_cpu_blocks, cpu_page_bytes)`` int8.
    A key's file carries, in order, the pages of the tensors referenced by
    the key's KV-cache group (``group_data_refs``), each truncated to the
    group's un-padded page size times ``block_size_factor``.
    """

    def __init__(
        self,
        cpu_tensors: list[torch.Tensor],
        group_data_refs: list[list[CanonicalKVCacheRef]],
        block_size_factor: int,
        file_mapper: FileMapper,
        store: bool,
        n_threads: int = 8,
    ):
        self.store = store
        self.file_mapper = file_mapper
        # One pool per direction: a shared pool's finished queue would hand
        # one handler the other's completions.
        n_threads = max(1, int(n_threads))
        self.pool = DualQueueThreadPool(
            0 if store else n_threads,
            n_threads if store else 0,
            thread_name_prefix="vllm_kv_fs_store" if store else "vllm_kv_fs_load",
        )
        self.transfer_type = ("CPU", "FS") if store else ("FS", "CPU")
        self._arrays: list[np.ndarray] = []
        for t in cpu_tensors:
            assert t.dtype == torch.int8 and t.ndim == 2 and t.device.type == "cpu"
            self._arrays.append(t.numpy())  # zero-copy view of the pinned buffer
        self.group_layout: list[list[tuple[int, int]]] = [
            [(ref.tensor_idx, ref.page_size_bytes * block_size_factor) for ref in refs]
            for refs in group_data_refs
        ]
        for layout in self.group_layout:
            for tensor_idx, nbytes in layout:
                assert nbytes <= self._arrays[tensor_idx].shape[1]
        self._inflight: dict[int, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def block_views(self, key: OffloadKey, cpu_block_id: int) -> list[memoryview]:
        group_idx = get_offload_group_idx(key)
        return [
            memoryview(self._arrays[tensor_idx][cpu_block_id, :nbytes])
            for tensor_idx, nbytes in self.group_layout[group_idx]
        ]

    def block_nbytes(self, key: OffloadKey) -> int:
        return sum(n for _, n in self.group_layout[get_offload_group_idx(key)])

    def _do_one(self, key: OffloadKey, cpu_block_id: int) -> None:
        path = self.file_mapper.get_file_name(key)
        views = self.block_views(key, cpu_block_id)
        if self.store:
            write_block_file(path, views)
        else:
            read_block_file(path, views)

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src, dst = transfer_spec
        cpu_spec, fs_spec = (src, dst) if self.store else (dst, src)
        assert isinstance(cpu_spec, CPULoadStoreSpec), cpu_spec
        assert isinstance(fs_spec, FSLoadStoreSpec), fs_spec
        keys = fs_spec.keys
        block_ids = [int(b) for b in cpu_spec.block_ids]
        assert len(keys) == len(block_ids), (len(keys), len(block_ids))
        if not keys:
            return False
        total = sum(self.block_nbytes(k) for k in keys)
        with self._lock:
            self._inflight[job_id] = (total, time.perf_counter())
        tasks = [functools.partial(self._do_one, k, b) for k, b in zip(keys, block_ids)]
        if self.store:
            self.pool.enqueue_store(job_id, len(tasks), tasks)
        else:
            self.pool.enqueue_load(job_id, len(tasks), tasks)
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id, success in self.pool.get_finished():
            with self._lock:
                size, t0 = self._inflight.pop(job_id, (None, None))
            results.append(
                TransferResult(
                    job_id=job_id,
                    success=success,
                    transfer_size=size,
                    transfer_time=(time.perf_counter() - t0) if t0 else None,
                    transfer_type=self.transfer_type,
                )
            )
        return results

    def wait(self, job_ids: set[int]) -> None:
        self.pool.wait_idle()

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False)


# --------------------------------------------------------------------------
# Scheduler side: a secondary tier that emits transfer jobs instead of I/O
# --------------------------------------------------------------------------


RUN_CONFIG_MARKER = ".run_config.json"


def discover_rank0_base_path(root_dir: str, model_name: str) -> str | None:
    """Find the base path rank 0's worker writes under on this node:
    ``<root>/<safe_model>_<digest>_r0``. The scheduler's own digest may differ
    from the workers' (their KV-cache configs are not byte-identical), so the
    scheduler trusts the directory rank 0 actually created, newest first."""
    safe = model_name.replace("/", "_")
    try:
        names = os.listdir(root_dir)
    except OSError:
        return None
    cands = [n for n in names if n.startswith(f"{safe}_") and n.endswith("_r0")
             and os.path.isdir(os.path.join(root_dir, n))]
    if not cands:
        return None
    newest = max(cands, key=lambda n: os.stat(os.path.join(root_dir, n)).st_mtime)
    return os.path.join(root_dir, newest[: -len("_r0")])


class _RankZeroFileLookup(AsyncLookupManager):
    """Answers lookups from this (scheduler) node's own files, under the
    directory rank 0's worker created (see ``discover_rank0_base_path``)."""

    def __init__(self, file_mapper: FileMapper, tier_type: str):
        super().__init__(tier_type=tier_type)
        self._file_mapper = file_mapper   # rank 0, scheduler-side digest
        self._resolved = False

    def _resolve(self) -> bool:
        if self._resolved:
            return True
        root = os.path.dirname(self._file_mapper.base_path)
        found = discover_rank0_base_path(root, self._file_mapper.fields["model_name"])
        if found is None:
            return False  # rank 0 has not stored anything yet
        if found != self._file_mapper.base_path:
            logger.warning(
                "Worker fs tier: scheduler digest path %s differs from rank 0's %s; "
                "using rank 0's (see %s there)",
                self._file_mapper.base_path, found,
                os.path.join(found + "_r0", RUN_CONFIG_MARKER),
            )
            self._file_mapper.base_path = found
        self._resolved = True
        return True

    def batch_lookup(self, keys: list[OffloadKey], req_context: ReqContext) -> Iterable[bool]:
        if not self._resolve():
            return [False] * len(keys)
        return [os.path.exists(self._file_mapper.get_file_name(k)) for k in keys]

    def path_for(self, key: OffloadKey) -> str | None:
        """Rank 0's file for ``key`` once the directory is known (main thread)."""
        if not self._resolve():
            return None
        return self._file_mapper.get_file_name(key)


class WorkerFileSystemTier(SecondaryTierManager):
    """Filesystem tier whose bytes move in the workers.

    ``submit_store``/``submit_load`` queue ``(tier job id, TransferSpec)``
    pairs; the connector scheduler ships them to the workers each step and
    calls ``mark_finished`` once every worker has reported. The tiering
    manager consumes results through ``get_finished_jobs`` as usual.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpecLike",
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 8,
        n_write_threads: int = 8,
    ):
        super().__init__(offloading_spec, memoryview(b""), tier_type)
        self.root_dir = root_dir
        self.n_read_threads = int(n_read_threads)
        self.n_write_threads = int(n_write_threads)
        self.file_mapper = make_file_mapper(offloading_spec, root_dir)
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(self.file_mapper.get_run_config(), f, indent=2, sort_keys=True)
        self._lookup = _RankZeroFileLookup(self.file_mapper, tier_type)
        self._pending: list[tuple[int, TransferSpec]] = []
        self._finished: list[JobResult] = []
        self._in_flight = 0

    # -- transfers -----------------------------------------------------------
    def submit_store(self, job_metadata: JobMetadata) -> None:
        spec: TransferSpec = (
            CPULoadStoreSpec([int(b) for b in job_metadata.block_ids]),
            FSLoadStoreSpec(list(job_metadata.keys)),
        )
        self._pending.append((job_metadata.job_id, spec))
        self._in_flight += 1

    def submit_load(self, job_metadata: JobMetadata) -> None:
        spec: TransferSpec = (
            FSLoadStoreSpec(list(job_metadata.keys)),
            CPULoadStoreSpec([int(b) for b in job_metadata.block_ids]),
        )
        self._pending.append((job_metadata.job_id, spec))
        self._in_flight += 1

    def take_pending(self) -> list[tuple[int, TransferSpec]]:
        pending, self._pending = self._pending, []
        return pending

    def mark_finished(self, tier_job_id: int, success: bool) -> None:
        self._finished.append(JobResult(job_id=tier_job_id, success=success))
        self._in_flight -= 1

    def get_finished_jobs(self) -> Iterable[JobResult]:
        finished, self._finished = self._finished, []
        return finished

    def has_pending_work(self) -> bool:
        return bool(self._pending) or self._in_flight > 0

    def drain_jobs(self) -> None:
        # Completions arrive through the connector on later engine steps;
        # there is nothing to block on in the scheduler process.
        return

    # -- lookups / request lifecycle -------------------------------------------
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        return self._lookup.lookup(key, req_context)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup.cleanup(req_context.req_id)

    def on_schedule_end(self) -> None:
        self._lookup.flush()

    def shutdown(self) -> None:
        self._lookup.shutdown()


class CPUPrimaryTierNoMmap(CPUOffloadingManager):
    """The CPU primary tier without a shared mmap region: the tiering
    manager only needs the read/write aliases."""

    def __init__(self, num_blocks: int, cache_policy: str = "lru", enable_events: bool = False):
        super().__init__(
            num_blocks=num_blocks,
            cache_policy=cache_policy,  # type: ignore[arg-type]
            enable_events=enable_events,
        )
        self.prepare_read = self.prepare_load
        self.complete_read = self.complete_load
        self.prepare_write = self.prepare_store
        self.complete_write = self.complete_store


class MultiNodeTieringOffloadingSpec(TieringOffloadingSpec):
    """``TieringOffloadingSpec`` with per-worker pinned CPU tiers and a
    worker-executed filesystem tier (``type: "fs_worker"``)."""

    # No shared mmap, buffered file I/O: CPU pages need no page alignment.
    BLOCK_SIZE_ALIGNMENT = 1

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config, kv_cache_config)
        _fix_per_group_block_sizes(self)
        tiers = [dict(t) for t in self.secondary_tier_configs]
        if len(tiers) != 1 or tiers[0].get("type") != FS_WORKER_TIER_TYPE:
            raise ValueError(
                "MultiNodeTieringOffloadingSpec needs exactly one secondary tier "
                f"of type {FS_WORKER_TIER_TYPE!r}; got {self.secondary_tier_configs!r}"
            )
        self._tier_config: dict[str, Any] = tiers[0]
        self._worker_tiers: list[WorkerFileSystemTier] = []

    @property
    def worker_tiers(self) -> list[WorkerFileSystemTier]:
        return self._worker_tiers

    def _tier_kwargs(self) -> dict[str, Any]:
        cfg = dict(self._tier_config)
        cfg.pop("type", None)
        cfg.pop("locality", None)
        cfg.pop("enable_kv_events", None)
        return cfg

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )
            if int(self.extra_config.get("store_threshold", 0)) >= 2:
                raise ValueError("store_threshold is not supported for tiering specs")
            primary = CPUPrimaryTierNoMmap(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,
                enable_events=enable_events,
            )
            tier = WorkerFileSystemTier(
                self, tier_type=FS_WORKER_TIER_TYPE, **self._tier_kwargs()
            )
            self._worker_tiers = [tier]
            self._manager = TieringOffloadingManager(
                primary_tier=primary,  # type: ignore[arg-type]
                secondary_tiers=[tier],
                enable_events=enable_events,
            )
            logger.info(
                "MultiNodeTieringOffloadingSpec: CPU tier %d blocks (%s), "
                "worker fs tier at %s",
                self.num_blocks,
                self.eviction_policy,
                tier.root_dir,
            )
        return self._manager

    def create_handlers(self, kv_caches: CanonicalKVCaches):
        # Per-worker pinned tensors, no shared region (the single-tier path).
        from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers

        return CpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
        )

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        yield from super().get_handlers(kv_caches)  # GPU <-> CPU, pinned
        assert self._handlers is not None
        cpu_tensors = self._handlers.gpu_to_cpu_handler.dst_tensors
        cfg = self._tier_kwargs()
        file_mapper = make_file_mapper(self, cfg["root_dir"])
        common = dict(
            cpu_tensors=cpu_tensors,
            group_data_refs=kv_caches.group_data_refs,
            block_size_factor=self.block_size_factor,
            file_mapper=file_mapper,
        )
        rank_dir = f"{file_mapper.base_path}_r{file_mapper.rank}"
        os.makedirs(rank_dir, exist_ok=True)
        with open(os.path.join(rank_dir, RUN_CONFIG_MARKER), "w") as f:
            json.dump(file_mapper.get_run_config(), f, indent=2, sort_keys=True)
        logger.info("Worker fs tier: rank %d writes under %s", file_mapper.rank, rank_dir)
        yield CPULoadStoreSpec, FSLoadStoreSpec, WorkerFsHandler(
            store=True, n_threads=int(cfg.get("n_write_threads", 8)), **common
        )
        yield FSLoadStoreSpec, CPULoadStoreSpec, WorkerFsHandler(
            store=False, n_threads=int(cfg.get("n_read_threads", 8)), **common
        )


# For type hints only; the real OffloadingSpec is what gets passed.
OffloadingSpecLike = Any


def make_file_mapper(offloading_spec: Any, root_dir: str) -> FileMapper:
    """A ``FileMapper`` whose run digest is identical in the scheduler and in
    every worker. ``FileMapper.from_offloading_spec`` hashes each group's
    layer-name list, and the per-worker KV-cache config carries different
    layer names than the scheduler's, so the scheduler would look up under a
    different base path than the one the workers write to (seen on the first
    deployment: ``_1c3a6cf092fe`` vs ``_b58237a2c54f``). Only fields that are
    the same everywhere go into the digest; ``rank`` stays outside it."""
    vllm_config = offloading_spec.vllm_config
    parallel_config = vllm_config.parallel_config
    dtype = str(vllm_config.cache_config.cache_dtype).replace("torch.", "")
    groups = [
        {"block_size": int(group.kv_cache_spec.block_size)}
        for group in offloading_spec.kv_cache_config.kv_cache_groups
    ]
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    draft = getattr(spec_cfg, "model", None) if spec_cfg is not None else None
    return FileMapper(
        root_dir=root_dir,
        model_name=vllm_config.model_config.model,
        hash_block_size=int(offloading_spec.hash_block_size),
        gpu_blocks_per_file=int(offloading_spec.block_size_factor),
        tp_size=parallel_config.tensor_parallel_size,
        pp_size=parallel_config.pipeline_parallel_size,
        pcp_size=parallel_config.prefill_context_parallel_size,
        dcp_size=parallel_config.decode_context_parallel_size,
        rank=parallel_config.rank,
        dtype=dtype,
        kv_cache_groups=groups,
        # the drafter's KV lives in the cache too: a different draft model
        # with the same shapes must not share a directory
        inference_engine=f"vllm-multinode|draft={draft}",
        parallel_agnostic=False,
    )


def _fix_per_group_block_sizes(spec: Any) -> None:
    """The base spec scales every group's block size by the DCP factor. Under
    the DCP overlay a group can be replicated instead of sharded (the DFlash
    drafter's sliding-window group), in which case its manager block size is
    the kernel block size. Mirror the overlay's per-group rule so offload keys
    line up with the KV manager (every 4th request hash for 256-token target
    blocks, every hash for 64-token drafter blocks)."""
    spec.gpu_block_size = tuple(
        group.kv_cache_spec.block_size * _group_cp_factor(spec.vllm_config, group.kv_cache_spec)
        for group in spec.kv_cache_config.kv_cache_groups
    )
    for block_size in spec.gpu_block_size:
        assert block_size % spec.hash_block_size == 0, (
            f"gpu_block_size={block_size} not divisible by hash_block_size={spec.hash_block_size}"
        )


def _group_cp_factor(vllm_config: "VllmConfig", kv_cache_spec: Any) -> int:
    """Context-parallel factor for one KV-cache group: the DCP overlay's
    ``cp_world_size_for_kv_cache_spec`` (full attention / MLA sharded,
    sliding-window and others replicated); the base rule otherwise."""
    parallel_config = vllm_config.parallel_config
    total = (
        parallel_config.decode_context_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    try:
        from vllm.v1.kv_cache_interface import cp_world_size_for_kv_cache_spec
    except ImportError:
        return total
    return int(cp_world_size_for_kv_cache_spec(kv_cache_spec, total))


# --------------------------------------------------------------------------
# Direct NVMe tier: the disk is the offload store, no CPU cache tier
# --------------------------------------------------------------------------
#
# The tiered design above can only serve a reload whose whole prefix fits the
# CPU tier at once (the connector counts a hit only when every block of the
# prefix, plus the drafter's trailing window, is present in the primary
# tier; promotions allocate primary blocks one per block and the lookup
# collapses to zero once the primary is full). With a 1 GB tier that caps a
# reload at ~260 blocks, and a 300k context would need ~4.7 GB of pinned
# memory per rank. Here the scheduler treats NVMe as the offload store
# (lookups by file existence on rank 0, no promotion) and the workers move
# blocks GPU <-> NVMe through a small pinned bounce buffer, reusing the
# fork's GPU<->CPU copy kernels (``SingleDirectionOffloadingHandler``) and
# driving all CUDA work from the main thread on each engine step.


class FSOffloadingManager(OffloadingManager):
    """Scheduler-side bookkeeping for a disk-resident offload store."""

    def __init__(self, file_mapper_rank0: FileMapper, known_cap: int = 500_000):
        self._lookup = _RankZeroFileLookup(file_mapper_rank0, FS_DIRECT_TIER_TYPE)
        self._inflight_store: set[OffloadKey] = set()
        self._inflight_load: set[OffloadKey] = set()
        self._known: "collections.OrderedDict[OffloadKey, None]" = collections.OrderedDict()
        self._known_cap = known_cap

    def _remember(self, key: OffloadKey) -> None:
        self._known[key] = None
        self._known.move_to_end(key)
        while len(self._known) > self._known_cap:
            self._known.popitem(last=False)

    def _on_disk_now(self, key: OffloadKey) -> bool:
        path = self._lookup.path_for(key)
        return path is not None and os.path.exists(path)

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        if key in self._known:
            return True
        if key in self._inflight_store:
            return None  # on disk in a moment; ask again next step
        result = self._lookup.lookup(key, req_context)
        if result is True:
            self._remember(key)
        return result

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> LoadStoreSpec:
        self._inflight_load.update(keys)
        return FSLoadStoreSpec(list(keys))

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        return

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._inflight_load.difference_update(keys)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        to_store = [
            k for k in keys
            if k not in self._known and k not in self._inflight_store and not self._on_disk_now(k)
        ]
        self._inflight_store.update(to_store)
        return PrepareStoreOutput(
            keys_to_store=to_store, store_spec=FSLoadStoreSpec(to_store), evicted_keys=[]
        )

    def complete_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext, success: bool = True
    ) -> None:
        for k in keys:
            self._inflight_store.discard(k)
            if success:
                self._remember(k)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup.cleanup(req_context.req_id)

    def on_schedule_end(self) -> None:
        self._lookup.flush()

    def take_events(self) -> Iterable[OffloadingEvent]:
        return ()

    def has_pending_work(self) -> bool:
        return bool(self._inflight_store or self._inflight_load)

    def reset_cache(self) -> None:
        self._known.clear()

    def get_stats(self):
        return None

    def shutdown(self) -> None:
        self._lookup.shutdown()


FS_DIRECT_TIER_TYPE = "fs_direct"


@dataclass
class _DirectJob:
    job_id: int
    store: bool
    items: list[tuple[OffloadKey, int]]        # (key, gpu block id), group-major
    next_item: int = 0                          # first item not yet in flight
    done_items: int = 0
    failed: bool = False
    t0: float = field(default_factory=time.perf_counter)
    nbytes: int = 0
    # store: copy sub-job id -> slots/keys awaiting file writes
    copies: dict[int, list[tuple[int, OffloadKey]]] = field(default_factory=dict)
    # load: read sub-job id -> (slots, gpu ids, keys) awaiting the copy to GPU
    reads: dict[int, tuple[list[int], list[int], list[OffloadKey]]] = field(default_factory=dict)
    slots: dict[OffloadKey, int] = field(default_factory=dict)   # slab: key -> disk slot
    epoch: int = 0
    seq: int = 0


class BounceController:
    """Owns the bounce buffer and drives the two-hop transfers.

    ``pump()`` runs on the main thread (from the handlers' ``get_finished``,
    which the connector calls every engine step) so the CUDA copies see the
    same stream semantics as the fork's own GPU<->CPU transfers; file I/O
    runs on the two thread pools.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        file_mapper: FileMapper,
        n_bounce: int = 48,
        n_read_threads: int = 8,
        n_write_threads: int = 8,
    ):
        from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

        pin = is_pin_memory_available()
        self.n_bounce = int(n_bounce)
        self.num_groups = len(kv_caches.group_data_refs)
        gpu_tensors: list[torch.Tensor] = []
        self.bounce: list[torch.Tensor] = []
        for t in kv_caches.tensors:
            gpu_tensors.append(t.tensor.view(torch.int8).view((-1, t.page_size_bytes)))
            self.bounce.append(
                torch.zeros((self.n_bounce, t.page_size_bytes), dtype=torch.int8, device="cpu", pin_memory=pin)
            )
        self.g2b = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors, cpu_tensors=self.bounce, block_size_factor=1,
            kv_cache_groups_data_refs=kv_caches.group_data_refs, gpu_to_cpu=True,
        )
        self.b2g = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors, cpu_tensors=self.bounce, block_size_factor=1,
            kv_cache_groups_data_refs=kv_caches.group_data_refs, gpu_to_cpu=False,
        )
        self.files = WorkerFsHandler(
            self.bounce, kv_caches.group_data_refs, 1, file_mapper, store=True, n_threads=n_write_threads
        )
        self.reader = WorkerFsHandler(
            self.bounce, kv_caches.group_data_refs, 1, file_mapper, store=False, n_threads=n_read_threads
        )
        self.free: collections.deque[int] = collections.deque(range(self.n_bounce))
        self.jobs: dict[int, _DirectJob] = {}
        self._sub_counter = 0
        self._results: dict[bool, list[TransferResult]] = {True: [], False: []}
        self._sub_to_job: dict[int, int] = {}
        self._failed_gpu_blocks: set[int] = set()
        self.total_bytes = sum(t.page_size_bytes for t in kv_caches.tensors)

    # -- helpers ----------------------------------------------------------------
    def _sub_id(self) -> int:
        self._sub_counter += 1
        return self._sub_counter

    def _gpu_spec(self, keys: list[OffloadKey], gpu_ids: list[int]) -> GPULoadStoreSpec:
        sizes = [0] * self.num_groups
        for k in keys:
            sizes[get_offload_group_idx(k)] += 1
        return GPULoadStoreSpec(gpu_ids, group_sizes=sizes, block_indices=[0] * self.num_groups)

    def submit(self, job_id: int, store: bool, spec: TransferSpec) -> bool:
        src, dst = spec
        gpu_spec, fs_spec = (src, dst) if store else (dst, src)
        assert isinstance(gpu_spec, GPULoadStoreSpec) and isinstance(fs_spec, FSLoadStoreSpec), spec
        gpu_ids = [int(b) for b in gpu_spec.block_ids]
        keys = list(fs_spec.keys)
        if len(gpu_ids) != len(keys):
            logger.error("Direct fs tier job %d: %d keys vs %d GPU blocks", job_id, len(keys), len(gpu_ids))
            return False
        if not keys:
            return False
        job = _DirectJob(job_id=job_id, store=store, items=list(zip(keys, gpu_ids)))
        job.nbytes = sum(self.files.block_nbytes(k) for k in keys)
        self.jobs[job_id] = job
        return True

    def take_results(self, store: bool) -> list[TransferResult]:
        out, self._results[store] = self._results[store], []
        return out

    def take_failed_gpu_blocks(self) -> set[int]:
        out, self._failed_gpu_blocks = self._failed_gpu_blocks, set()
        return out

    def _finish(self, job: _DirectJob) -> None:
        del self.jobs[job.job_id]
        self._results[job.store].append(
            TransferResult(
                job_id=job.job_id, success=not job.failed,
                transfer_size=job.nbytes if not job.failed else None,
                transfer_time=time.perf_counter() - job.t0,
                transfer_type=("GPU", "FS") if job.store else ("FS", "GPU"),
            )
        )

    # -- the state machine -------------------------------------------------------
    def pump(self) -> None:
        # 1. GPU->bounce copies done: hand the slots to the file writers
        for r in self.g2b.get_finished():
            job = self.jobs.get(self._sub_to_job.pop(r.job_id, -1))
            if job is None:
                continue
            slots_keys = job.copies.pop(r.job_id)
            wid = self._sub_id()
            self._sub_to_job[wid] = job.job_id
            job.copies[wid] = slots_keys
            self._enqueue_store_tasks(job, wid, slots_keys)
        # 2. file writes done: free the slots, count
        for wid, ok in self.files.pool.get_finished():
            job = self.jobs.get(self._sub_to_job.pop(wid, -1))
            if job is None:
                continue
            slots_keys = job.copies.pop(wid)
            for slot, _ in slots_keys:
                self.free.append(slot)
            job.done_items += len(slots_keys)
            job.failed |= not ok
            if job.done_items == len(job.items):
                self._finish(job)
        # 3. file reads done: copy bounce->GPU
        for rid, ok in self.reader.pool.get_finished():
            job = self.jobs.get(self._sub_to_job.pop(rid, -1))
            if job is None:
                continue
            slots, gpu_ids, keys = job.reads.pop(rid)
            if not ok:
                job.failed = True
                self._failed_gpu_blocks.update(gpu_ids)
                for slot in slots:
                    self.free.append(slot)
                job.done_items += len(slots)
                if job.done_items == len(job.items):
                    self._finish(job)
                continue
            cid = self._sub_id()
            self._sub_to_job[cid] = job.job_id
            job.reads[cid] = (slots, gpu_ids, keys)
            assert self.b2g.transfer_async(cid, (CPULoadStoreSpec(slots), self._gpu_spec(keys, gpu_ids)))
        # 4. bounce->GPU copies done: free the slots, count
        for r in self.b2g.get_finished():
            job = self.jobs.get(self._sub_to_job.pop(r.job_id, -1))
            if job is None:
                continue
            slots, _, _ = job.reads.pop(r.job_id)
            for slot in slots:
                self.free.append(slot)
            job.done_items += len(slots)
            if job.done_items == len(job.items):
                self._finish(job)
        # 5. start new chunks while slots are free (loads first: a waiting
        #    request is more urgent than a store)
        for store_pass in (False, True):
            for job in list(self.jobs.values()):
                if job.store != store_pass or job.next_item >= len(job.items) or not self.free:
                    continue
                n = min(len(self.free), len(job.items) - job.next_item)
                chunk = job.items[job.next_item : job.next_item + n]
                job.next_item += n
                slots = [self.free.popleft() for _ in range(n)]
                keys = [k for k, _ in chunk]
                gpu_ids = [g for _, g in chunk]
                sid = self._sub_id()
                self._sub_to_job[sid] = job.job_id
                if job.store:
                    job.copies[sid] = list(zip(slots, keys))
                    assert self.g2b.transfer_async(sid, (self._gpu_spec(keys, gpu_ids), CPULoadStoreSpec(slots)))
                else:
                    job.reads[sid] = (slots, gpu_ids, keys)
                    self._enqueue_load_tasks(job, sid, keys, slots)

    # hooks for the disk layer (per-key files here; slots in SlabBounceController)
    def _enqueue_store_tasks(self, job: _DirectJob, wid: int, slots_keys: list[tuple[int, OffloadKey]]) -> None:
        self.files.pool.enqueue_store(
            wid, len(slots_keys), [functools.partial(self.files._do_one, k, slot) for slot, k in slots_keys]
        )

    def _enqueue_load_tasks(self, job: _DirectJob, sid: int, keys: list[OffloadKey], slots: list[int]) -> None:
        self.reader.pool.enqueue_load(
            sid, len(keys), [functools.partial(self.reader._do_one, k, slot) for k, slot in zip(keys, slots)]
        )

    def wait_all(self, timeout: float | None = None) -> None:
        """Block until every job is done. Returning early would let the
        connector reuse GPU blocks a store is still reading, so a stall is
        reported but never given up on (unless a timeout is explicitly asked
        for, in which case it raises)."""
        t0 = time.monotonic()
        last_log = t0
        while self.jobs:
            self.pump()
            time.sleep(0.005)
            now = time.monotonic()
            if timeout is not None and now - t0 > timeout:
                raise TimeoutError(f"{len(self.jobs)} tier jobs still in flight after {timeout:.0f}s")
            if now - last_log > 30:
                logger.warning("Tier wait: %d jobs still in flight after %.0fs", len(self.jobs), now - t0)
                last_log = now

    def shutdown(self) -> None:
        self.files.shutdown()
        self.reader.shutdown()
        self.g2b.shutdown()
        self.b2g.shutdown()


class DirectFsHandler(OffloadingHandler):
    """One direction (GPU->FS store or FS->GPU load) over a shared controller."""

    def __init__(self, controller: BounceController, store: bool):
        self.controller = controller
        self.store = store

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        return self.controller.submit(job_id, self.store, transfer_spec)

    def get_finished(self) -> list[TransferResult]:
        self.controller.pump()
        return self.controller.take_results(self.store)

    def wait(self, job_ids: set[int]) -> None:
        self.controller.wait_all()

    def shutdown(self) -> None:
        if self.store:  # the controller is shared; shut it down once
            self.controller.shutdown()


class MultiNodeDirectFsOffloadingSpec(OffloadingSpecBase):
    """OffloadingSpec for the direct NVMe tier.

    kv_connector_extra_config keys: ``root_dir`` (required), ``bounce_blocks``
    (default 48: pinned bounce slots per worker, ~4 MB each here),
    ``n_read_threads`` / ``n_write_threads`` (default 8).
    """

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config, kv_cache_config)
        _fix_per_group_block_sizes(self)
        self.root_dir = self.extra_config.get("root_dir")
        if not self.root_dir:
            raise ValueError("MultiNodeDirectFsOffloadingSpec needs 'root_dir' in kv_connector_extra_config")
        self.n_bounce = int(self.extra_config.get("bounce_blocks", 48))
        self.n_read_threads = int(self.extra_config.get("n_read_threads", 8))
        self.n_write_threads = int(self.extra_config.get("n_write_threads", 8))
        self._manager: FSOffloadingManager | None = None
        self.controller: BounceController | None = None

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            fm = make_file_mapper(self, self.root_dir)
            config_path = fm.get_config_file_path()
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    json.dump(fm.get_run_config(), f, indent=2, sort_keys=True)
            self._manager = FSOffloadingManager(fm)
            logger.info("Direct fs tier: scheduler looks up under rank 0's directory in %s", self.root_dir)
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        fm = make_file_mapper(self, self.root_dir)
        rank_dir = f"{fm.base_path}_r{fm.rank}"
        os.makedirs(rank_dir, exist_ok=True)
        with open(os.path.join(rank_dir, RUN_CONFIG_MARKER), "w") as f:
            json.dump(fm.get_run_config(), f, indent=2, sort_keys=True)
        self.controller = BounceController(
            kv_caches, fm, n_bounce=self.n_bounce,
            n_read_threads=self.n_read_threads, n_write_threads=self.n_write_threads,
        )
        logger.info(
            "Direct fs tier: rank %d writes under %s; %d bounce slots (%.0f MB pinned)",
            fm.rank, rank_dir, self.n_bounce, self.n_bounce * self.controller.total_bytes / 1e6,
        )
        yield GPULoadStoreSpec, FSLoadStoreSpec, DirectFsHandler(self.controller, store=True)
        yield FSLoadStoreSpec, GPULoadStoreSpec, DirectFsHandler(self.controller, store=False)


class MultiNodeDirectConnectorWorker(OffloadingConnectorWorker):
    """Base worker plus load-failure tolerance: a failed load (a file missing
    on this rank) completes the job and reports the GPU blocks as invalid so
    the scheduler recomputes them."""

    def __init__(self, spec: MultiNodeDirectFsOffloadingSpec):
        super().__init__(spec)
        self._invalid_block_ids: set[int] = set()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_recving: set[str] = set()
        meta = self._connector_worker_meta
        for result in self.worker.get_finished():
            job_id = result.job_id
            if result.success and result.transfer_time is not None and result.transfer_size is not None:
                is_load = job_id in self._load_jobs
                stats = meta.transfer_stats.load if is_load else meta.transfer_stats.store
                stats.record(result.transfer_size, result.transfer_time)
            elif not result.success:
                logger.warning("Direct fs tier: job %d failed on this rank", job_id)
            meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
        ctrl = getattr(self.spec, "controller", None)
        if ctrl is not None:
            self._invalid_block_ids |= ctrl.take_failed_gpu_blocks()
        return set(), finished_recving

    def take_invalid_block_ids(self) -> set[int]:
        out, self._invalid_block_ids = self._invalid_block_ids, set()
        return out


class MultiNodeDirectConnector(OffloadingConnector):
    """``OffloadingConnector`` with the direct NVMe tier: base scheduler,
    failure-tolerant worker. Selected with ``kv_connector_module_path``."""

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)
        if not isinstance(spec, MultiNodeDirectFsOffloadingSpec):
            raise ValueError("MultiNodeDirectConnector requires spec_name=MultiNodeDirectFsOffloadingSpec")
        self.connector_scheduler = None
        self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = MultiNodeDirectConnectorWorker(spec)

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.connector_worker is None:
            return set()
        return self.connector_worker.take_invalid_block_ids()


# --------------------------------------------------------------------------
# Slab store: fixed-size ring buffer on NVMe (bounded, no janitor)
# --------------------------------------------------------------------------
#
# Two preallocated-by-count slab files per rank (one per KV-cache group),
# fixed slots, LRU reuse. It is a cache: any doubt is a miss, and the one
# invariant is that a hit never serves wrong bytes on any rank. Mechanisms,
# deliberately few (docs/SLAB-DESIGN.md, review 2026-09-04):
#   1. store failures on any rank keep the key out of the index; load
#      failures drop it (failed_jobs channel, see _FailureAwareScheduler);
#   2. all-or-nothing placement per prepare_store call (None = retry);
#   3. every slot header carries the key; loads verify it, so cross-rank
#      divergence self-heals lazily;
#   4. slot write order: blank header, payload, header, so a process crash
#      leaves a blank or a complete slot; a node reboot (boot_id changed)
#      wipes the rank's slabs;
#   5. drafter blocks stored in the same call as a target block are touched
#      with it on a hit, so the sliding-window tail stays as young as the
#      prefix it belongs to.

import zlib  # noqa: E402  (CRC32 for slab crash-safety)

SLAB_MAGIC = b"GLMSLAB1"
SLAB_HEADER_BYTES = 128
SLAB_VERSION = 2   # v2 adds payload_crc + header_crc; v1 slabs (no CRC) are rejected on read/scan
SLAB_META_VERSION = 3   # v3: the gate also compares content_identity (weights, dtype/quant/rope, overlay)
try:
    from vllm.version import __version__ as _VLLM_VERSION   # gates the KV-block hash scheme (GLM review)
except Exception:  # noqa: BLE001
    _VLLM_VERSION = "unknown"
# magic 8 | version u32 | group u32 | epoch u32 | length u32 | seq u64 | key 36 (32 hash + 4 group)
#   | payload_crc u32 | header_crc u32 | pad 52
_SLAB_HDR = struct.Struct("<8sIIIIQ36sII52x")
assert _SLAB_HDR.size == SLAB_HEADER_BYTES

def _crc(*bufs) -> int:
    c = 0
    for b in bufs:
        c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF
SLAB_META = "slab-meta.json"
SLAB_EPOCH = "slab-epoch"
DRAFTER_PER_TARGET = 4  # 64-token drafter blocks per 256-token target block


def _boot_id() -> str:
    try:
        return pathlib_read("/proc/sys/kernel/random/boot_id")
    except OSError:
        return "unknown"


def pathlib_read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _atomic_write_json(path: str, obj) -> None:
    """Publish metadata durably: temp file -> fsync -> rename -> directory fsync, so a crash
    leaves either the old file or the complete new one, never a truncated one."""
    d = os.path.dirname(path) or "."
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _slab_persist_value(v) -> bool:
    """Coerce a persist flag from extra_config (JSON may carry "false"/"0") or env. bool("false")
    is True in Python, so strings get the accepted-values rule."""
    if isinstance(v, str):
        return v.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(v)


def _slab_persist_default() -> bool:
    """Default for cross-reboot slab persistence: ON, overridable by env for ops."""
    return _slab_persist_value(os.environ.get("SLAB_PERSIST_ACROSS_REBOOT", "1"))


def _dir_fingerprint(path) -> str | None:
    """Stat-based identity of a local weights directory: sha256 over the sorted (relative name,
    size, mtime_ns) of its checkpoint/config files. No file reads. None if not a local directory
    (an HF repo id) or empty. Detects a checkpoint swapped in place at the same path."""
    if not path or not os.path.isdir(str(path)):
        return None
    import hashlib
    h = hashlib.sha256(); n = 0
    for root, _dirs, files in os.walk(str(path)):
        for fn in sorted(files):
            if not fn.endswith((".safetensors", ".json", ".bin", ".pt", ".pth", ".gguf")):
                continue
            fp = os.path.join(root, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, str(path))
            h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode()); n += 1
    return h.hexdigest() if n else None


def _content_identity(vllm_config, draft, extra_config) -> dict:
    """Everything that changes the KV BYTES a token-hash key maps to, beyond what run_config
    already carries. The key is sha256(parent, tokens): it names the INPUT, not the output, so
    a persisted slot is only trustworthy while all of this is unchanged. Any difference wipes."""
    mc = getattr(vllm_config, "model_config", None); cc = getattr(vllm_config, "cache_config", None)
    pc = getattr(vllm_config, "parallel_config", None); hf = getattr(mc, "hf_config", None)
    def js(x):
        try: return json.dumps(x, sort_keys=True, default=str)
        except Exception: return str(x)  # noqa: BLE001
    return {
        "weights_target": _dir_fingerprint(getattr(mc, "model", None)),
        "weights_draft": _dir_fingerprint(draft),
        "dtype": str(getattr(mc, "dtype", None)),
        "quantization": getattr(mc, "quantization", None),
        "revision": getattr(mc, "revision", None),
        "hf_overrides": js(getattr(mc, "hf_overrides", None)),
        "rope_theta": getattr(hf, "rope_theta", None),
        "rope_scaling": js(getattr(hf, "rope_scaling", None)),
        "max_position_embeddings": getattr(hf, "max_position_embeddings", None),
        "cache_dtype": str(getattr(cc, "cache_dtype", None)),
        "cp_kv_cache_interleave_size": getattr(pc, "cp_kv_cache_interleave_size", None),
        "prefix_caching_hash_algo": getattr(cc, "prefix_caching_hash_algo", None),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "slab_salt": (extra_config or {}).get("slab_salt"),   # overlay digest from the launcher
    }


def _slab_should_wipe(old, slot_bytes, slot_counts, run_config, engine_version, boot_id, persist,
                      content_identity=None) -> bool:
    """Whether to start the rank's slabs empty. Wipe when they cannot mean the same thing:
    no/old meta, a different format version, changed geometry, a different engine build (KV-block
    hash scheme), or any run_config difference. When persist is False, also wipe on a new boot
    (the pre-v2 behaviour); when True (default), a reboot keeps the slabs (payload CRC makes a
    persisted slot safe to trust)."""
    if (old is None
            or old.get("version") != SLAB_META_VERSION
            or old.get("slot_bytes") != slot_bytes
            or old.get("slot_counts") != slot_counts
            or old.get("engine_version") != engine_version
            or old.get("run_config") != run_config
            or old.get("content_identity") != content_identity):
        return True
    if not persist and old.get("boot_id") != boot_id:
        return True
    return False


def _atomic_write_text(path: str, text: str) -> None:
    """Durable text publish, same barrier as _atomic_write_json. Used for the slab-epoch file:
    now that a reboot no longer wipes, the epoch is the ONLY cross-reboot invalidation barrier,
    so an epoch bump (reset_cache) must be durable before we act on it (GLM review)."""
    d = os.path.dirname(path) or "."
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


class SlabLoadStoreSpec(LoadStoreSpec):
    """Blocks addressed by (key, slot) in the per-rank slab of their group.
    Stores also carry the scheduler's epoch and write sequence number, so
    every rank writes the same header and the index rebuilds in order."""

    def __init__(self, keys: list[OffloadKey], slots: list[int], epoch: int = 0, seq: int = 0):
        assert len(keys) == len(slots)
        self.keys: list[OffloadKey] = list(keys)
        self.slots: list[int] = [int(x) for x in slots]
        self.epoch = int(epoch)
        self.seq = int(seq)

    @staticmethod
    def medium() -> str:
        return "SLAB"

    def __repr__(self) -> str:
        return f"SLAB({len(self.keys)} keys)"


def slab_geometry(group_bytes: list[int], disk_bytes_per_rank: int) -> tuple[list[int], list[int]]:
    """(slot_bytes, slot_count) per group from the group's payload bytes and
    the per-rank disk budget. Group 0 (target) gets N rows, every other group
    DRAFTER_PER_TARGET * N; header included in slot bytes, 4 KB aligned."""
    slot_bytes = [_round_up(SLAB_HEADER_BYTES + b, 4096) for b in group_bytes]
    per_row = slot_bytes[0] + sum(DRAFTER_PER_TARGET * b for b in slot_bytes[1:])
    n = max(1, int(disk_bytes_per_rank) // per_row)
    counts = [n] + [DRAFTER_PER_TARGET * n] * (len(group_bytes) - 1)
    return slot_bytes, counts


class SlabIO:
    """Slot-level I/O for one rank: header/payload layout, key verification,
    crash-safe write order. One fd per group, opened once."""

    def __init__(self, rank_dir: str, slot_bytes: list[int], slot_counts: list[int], epoch: int = 0):
        self.rank_dir = rank_dir
        self.slot_bytes = slot_bytes
        self.slot_counts = slot_counts
        self.epoch = epoch
        os.makedirs(rank_dir, exist_ok=True)
        self.fds: list[int] = []
        for g in range(len(slot_bytes)):
            fd = os.open(os.path.join(rank_dir, f"g{g}.slab"), os.O_CREAT | os.O_RDWR, 0o644)
            cap = slot_bytes[g] * slot_counts[g]
            if os.fstat(fd).st_size > cap:
                os.ftruncate(fd, cap)  # a smaller budget drops the slots past it
            self.fds.append(fd)

    def path(self, group: int) -> str:
        return os.path.join(self.rank_dir, f"g{group}.slab")

    def _off(self, group: int, slot: int) -> int:
        assert 0 <= slot < self.slot_counts[group], (group, slot)
        return slot * self.slot_bytes[group]

    def header(self, key: OffloadKey, length: int, seq: int, payload_crc: int, epoch: int | None = None) -> bytes:
        kb = bytes(key)
        assert len(kb) == 36, len(kb)   # 32-byte hash + 4-byte group; "36s" would silently pad/truncate
        ep = self.epoch if epoch is None else int(epoch)
        base = _SLAB_HDR.pack(SLAB_MAGIC, SLAB_VERSION, get_offload_group_idx(key), ep,
                              length, seq, kb, payload_crc, 0)
        return _SLAB_HDR.pack(SLAB_MAGIC, SLAB_VERSION, get_offload_group_idx(key), ep,
                              length, seq, kb, payload_crc, _crc(base))

    @staticmethod
    def parse_header(raw: bytes):
        if len(raw) < SLAB_HEADER_BYTES:
            return None
        magic, version, group, epoch, length, seq, key, pcrc, hcrc = _SLAB_HDR.unpack(raw[:SLAB_HEADER_BYTES])
        if magic != SLAB_MAGIC or version != SLAB_VERSION:
            return None
        base = _SLAB_HDR.pack(magic, version, group, epoch, length, seq, key, pcrc, 0)
        if _crc(base) != hcrc:   # torn/partial header (a buffered 128B write is not power-loss atomic)
            return None
        return group, epoch, length, seq, OffloadKey(key), pcrc

    @staticmethod
    def _pwrite_all(fd: int, data: bytes, off: int) -> None:
        view = memoryview(data)
        while view:
            n = os.pwrite(fd, view, off)
            if n <= 0:
                raise OSError(f"pwrite returned {n}")
            off += n
            view = view[n:]

    def write(self, key: OffloadKey, slot: int, views: list[memoryview], seq: int, epoch: int | None = None) -> None:
        g = get_offload_group_idx(key)
        fd, off = self.fds[g], self._off(g, slot)
        length = sum(len(v) for v in views)
        assert SLAB_HEADER_BYTES + length <= self.slot_bytes[g], (length, self.slot_bytes[g])
        payload_crc = _crc(*[v.cast("B") for v in views])         # 0. crc the payload (verified on read)
        self._pwrite_all(fd, b"\0" * SLAB_HEADER_BYTES, off)    # 1. blank: slot invalid while in progress
        pos = off + SLAB_HEADER_BYTES
        remaining = [v.cast("B") for v in views]
        while remaining:
            n = os.pwritev(fd, remaining, pos)                    # 2. payload
            if n <= 0:
                raise OSError(f"pwritev returned {n}")
            pos += n
            remaining = _drain_iov(remaining, n)
        # seal with the epoch this store was issued under; do not mutate self.epoch from the
        # writer threads (a store from an older epoch must never be sealed as the newer one)
        self._pwrite_all(fd, self.header(key, length, seq, payload_crc, epoch=epoch), off)  # 3. header last, in full

    def read(self, key: OffloadKey, slot: int, views: list[memoryview], epoch: int | None = None) -> None:
        g = get_offload_group_idx(key)
        fd, off = self.fds[g], self._off(g, slot)
        hdr = self.parse_header(os.pread(fd, SLAB_HEADER_BYTES, off))
        expected = sum(len(v) for v in views)
        if hdr is None or hdr[4] != key or hdr[2] != expected:
            raise OSError(f"slot {slot} of group {g} does not hold the requested block")
        if epoch is not None and hdr[1] != int(epoch):
            # a complete, CRC-valid slot from a PREVIOUS epoch (e.g. this rank lost the post-reset
            # rewrite to a power cut while rank 0 kept it): stale KV, never serve it
            raise OSError(f"slot {slot} of group {g} holds epoch {hdr[1]}, load is epoch {epoch}")
        pos = off + SLAB_HEADER_BYTES
        remaining = [v.cast("B") for v in views]
        while remaining:
            n = os.preadv(fd, remaining, pos)
            if n <= 0:
                raise OSError("short read")
            pos += n
            remaining = _drain_iov(remaining, n)
        if _crc(*[v.cast("B") for v in views]) != hdr[5]:   # torn payload under a valid header, or bit-rot
            raise OSError(f"slot {slot} of group {g} failed payload CRC")

    def sync(self, group: int, slots: list[int]) -> None:
        """Push a chunk's writes out and drop them from the page cache."""
        fd = self.fds[group]
        os.fdatasync(fd)
        for slot in slots:
            try:
                os.posix_fadvise(fd, self._off(group, slot), self.slot_bytes[group], os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass

    def scan(self, group: int, epoch: int):
        """Yield (slot, seq, key) for every valid slot of the current epoch."""
        fd, sb = self.fds[group], self.slot_bytes[group]
        for slot in range(self.slot_counts[group]):
            raw = os.pread(fd, SLAB_HEADER_BYTES, slot * sb)
            if len(raw) < SLAB_HEADER_BYTES:
                return  # sparse tail: nothing written past here
            hdr = self.parse_header(raw)
            if hdr is None or hdr[0] != group or hdr[1] != epoch:
                continue
            yield slot, hdr[3], hdr[4]

    def wipe(self) -> None:
        for fd in self.fds:
            os.ftruncate(fd, 0)
            os.fsync(fd)   # the truncate must be durable before the new meta is published

    def max_epoch(self) -> int | None:
        """Highest epoch found in any valid header, None if the slabs are empty. Used when the
        epoch file is missing or corrupt: the next epoch must be ABOVE everything on disk."""
        best = None
        for g in range(len(self.fds)):
            fd, sb = self.fds[g], self.slot_bytes[g]
            for slot in range(self.slot_counts[g]):
                raw = os.pread(fd, SLAB_HEADER_BYTES, slot * sb)
                if len(raw) < SLAB_HEADER_BYTES:
                    break
                hdr = self.parse_header(raw)
                if hdr is not None and (best is None or hdr[1] > best):
                    best = hdr[1]
        return best

    def close(self) -> None:
        for fd in self.fds:
            os.close(fd)
        self.fds = []


class SlabOffloadingManager(OffloadingManager):
    """Scheduler-side index of the slab store: per group an LRU (key -> slot)
    and a free list. Slot counts come from rank 0's slab-meta.json (the
    workers own the byte math); the index is rebuilt from rank 0's slot
    headers at startup."""

    def __init__(self, root_dir: str, model_name: str, rebuild: bool = True):
        self.root_dir = root_dir
        self.model_name = model_name
        self._rank0: str | None = None
        self.io: SlabIO | None = None
        self.counts: list[int] = []
        self.index: list["collections.OrderedDict[OffloadKey, int]"] = []
        self.free: list[collections.deque[int]] = []
        self.inflight_store: dict[OffloadKey, int] = {}
        self.inflight_load: dict[OffloadKey, int] = {}
        self.condemned: dict[OffloadKey, int] = {}   # dropped keys whose slot still has readers
        self.failed_store: set[OffloadKey] = set()
        self.siblings: dict[OffloadKey, list[OffloadKey]] = {}
        self.seq = 0
        self.epoch = 0
        self._pending_reset = False
        self._rebuild = rebuild
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "stores": 0, "load_failures": 0}

    # -- lazy attach to rank 0's slabs ---------------------------------------------
    def _attach(self) -> bool:
        if self.io is not None:
            return True
        base = discover_rank0_base_path(self.root_dir, self.model_name)
        if base is None:
            return False
        rank_dir = base + "_r0"
        meta_path = os.path.join(rank_dir, SLAB_META)
        if not os.path.exists(meta_path):
            return False
        try:
            meta = json.load(open(meta_path))
            self.counts = list(meta["slot_counts"])
            slot_bytes = list(meta["slot_bytes"])
        except (OSError, ValueError, KeyError, TypeError):
            return False  # torn or unreadable meta: treat as not there yet
        self._rank0 = rank_dir
        epoch_path = os.path.join(rank_dir, SLAB_EPOCH)
        epoch_known = False
        try:
            if os.path.exists(epoch_path):
                self.epoch = int(pathlib_read(epoch_path)); epoch_known = True
        except ValueError:
            epoch_known = False   # corrupt: decided below from the headers, never assumed 0
        if self._pending_reset:
            self._pending_reset = False
            self.epoch += 1
            _atomic_write_text(epoch_path, str(self.epoch))
        self.io = SlabIO(rank_dir, slot_bytes, self.counts, self.epoch)
        if not epoch_known:
            # No usable epoch file. If the slabs hold anything, every slot on disk belongs to an
            # epoch we can no longer vouch for: start ABOVE the highest one (logical wipe) rather
            # than at 0, which would resurrect never-overwritten epoch-0 slots. Then publish it so
            # a later absence is an anomaly, not the normal state.
            top = self.io.max_epoch()
            self.epoch = 0 if top is None else top + 1
            self.io.epoch = self.epoch
            _atomic_write_text(epoch_path, str(self.epoch))
            logger.warning("Slab store: epoch file missing/corrupt at %s; using epoch %d (max on disk %s)",
                           epoch_path, self.epoch, top)
        self.index = [collections.OrderedDict() for _ in self.counts]
        self.free = [collections.deque() for _ in self.counts]
        used = 0
        for g, n in enumerate(self.counts):
            present = {}
            if self._rebuild:
                present = {slot: (seq, key) for slot, seq, key in self.io.scan(g, self.epoch)}
            for slot, (seq, key) in sorted(present.items(), key=lambda kv: kv[1][0]):
                if key in self.index[g]:
                    self.index[g].pop(key)  # duplicate key: the newer write wins
                self.index[g][key] = slot
                self.seq = max(self.seq, seq)
            taken = set(self.index[g].values())
            self.free[g].extend(slot for slot in range(n) if slot not in taken)
            used += len(self.index[g])
        logger.info(
            "Slab store: attached to %s, counts=%s, %d blocks indexed from headers, epoch %d",
            rank_dir, self.counts, used, self.epoch,
        )
        return True

    # -- OffloadingManager --------------------------------------------------------
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        if not self._attach():
            return False
        g = get_offload_group_idx(key)
        idx = self.index[g]
        if key in idx:
            idx.move_to_end(key)
            for sib in self.siblings.get(key, ()):
                gs = get_offload_group_idx(sib)
                if sib in self.index[gs]:
                    self.index[gs].move_to_end(sib)
            self.stats["hits"] += 1
            return True
        if key in self.inflight_store:
            return None
        self.stats["misses"] += 1
        return False

    def _evictable(self, g: int, protect: set[OffloadKey]) -> int:
        return sum(1 for k in self.index[g] if self.inflight_load.get(k, 0) == 0 and k not in protect)

    def _take_slot(self, g: int, protect: set[OffloadKey], evicted: list[OffloadKey]) -> int:
        if self.free[g]:
            return self.free[g].popleft()
        for key in self.index[g]:  # LRU first
            if self.inflight_load.get(key, 0) > 0 or key in protect:
                continue
            slot = self.index[g].pop(key)
            self.siblings.pop(key, None)
            self.stats["evictions"] += 1
            evicted.append(key)
            return slot
        raise RuntimeError("no evictable slot after a successful preflight")

    def prepare_store(self, keys: Collection[OffloadKey], req_context: ReqContext) -> PrepareStoreOutput | None:
        if not self._attach():
            return None
        keys = list(dict.fromkeys(keys))  # dedupe, keep order
        protect = set(keys)               # never evict a key of this very call
        new = [k for k in keys if k not in self.index[get_offload_group_idx(k)] and k not in self.inflight_store]
        need: dict[int, int] = {}
        for k in new:
            g = get_offload_group_idx(k)
            need[g] = need.get(g, 0) + 1
        for g, n in need.items():  # transactional: preflight, or retry next step untouched
            if len(self.free[g]) + self._evictable(g, protect) < n:
                return None
        evicted: list[OffloadKey] = []
        taken = [(k, self._take_slot(get_offload_group_idx(k), protect, evicted)) for k in new]
        for k, slot in taken:
            self.inflight_store[k] = slot
        targets = [k for k in new if get_offload_group_idx(k) == 0]
        drafters = [k for k in keys if get_offload_group_idx(k) != 0]
        for i, t in enumerate(targets):  # bounded: the i-th target's 4 drafter blocks of this call
            sib = drafters[DRAFTER_PER_TARGET * i : DRAFTER_PER_TARGET * (i + 1)]
            if sib:
                self.siblings[t] = sib
        self.seq += 1
        self.stats["stores"] += len(taken)
        return PrepareStoreOutput(
            keys_to_store=[k for k, _ in taken],
            store_spec=SlabLoadStoreSpec([k for k, _ in taken], [s for _, s in taken], epoch=self.epoch, seq=self.seq),
            evicted_keys=evicted,
        )

    def mark_store_failed(self, keys: Collection[OffloadKey]) -> None:
        self.failed_store.update(keys)

    def complete_store(self, keys: Collection[OffloadKey], req_context: ReqContext, success: bool = True) -> None:
        if not self.index:
            return
        for k in keys:
            slot = self.inflight_store.pop(k, None)
            if slot is None:
                continue
            g = get_offload_group_idx(k)
            if success and k not in self.failed_store:
                self.index[g][k] = slot
            else:
                self.free[g].append(slot)
                self.siblings.pop(k, None)
            self.failed_store.discard(k)

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> LoadStoreSpec:
        slots = []
        attached = self._attach()
        for k in keys:
            g = get_offload_group_idx(k)
            # a key that vanished between lookup and load gets slot -1: the
            # read fails on every rank and the blocks are recomputed
            slots.append(self.index[g].get(k, -1) if attached else -1)
            self.inflight_load[k] = self.inflight_load.get(k, 0) + 1
        return SlabLoadStoreSpec(list(keys), slots, epoch=self.epoch, seq=self.seq)

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        if not self.index:
            return
        for k in keys:
            n = self.inflight_load.get(k, 0) - 1
            if n <= 0:
                self.inflight_load.pop(k, None)
                slot = self.condemned.pop(k, None)
                if slot is not None:  # last reader gone: the slot may be reused now
                    self.free[get_offload_group_idx(k)].append(slot)
            else:
                self.inflight_load[k] = n

    def mark_load_failed(self, keys: Collection[OffloadKey]) -> None:
        """A rank could not serve these blocks: drop them from the index at
        once (a miss from now on) but keep the slot out of reuse while any
        rank may still be reading it (a reader must never see another key's
        payload under this key's header)."""
        if not self.index:
            return
        for k in keys:
            g = get_offload_group_idx(k)
            slot = self.index[g].pop(k, None)
            if slot is None:
                continue
            self.stats["load_failures"] += 1
            if self.inflight_load.get(k, 0) > 0:
                self.condemned[k] = slot
            else:
                self.free[g].append(slot)
            self.siblings.pop(k, None)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        if not self.index:  # called for every request, attached or not
            return
        for k in keys:
            g = get_offload_group_idx(k)
            if k in self.index[g]:
                self.index[g].move_to_end(k)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def on_request_finished(self, req_context: ReqContext) -> None:
        return

    def on_schedule_end(self) -> None:
        return

    def take_events(self) -> Iterable[OffloadingEvent]:
        return ()

    def has_pending_work(self) -> bool:
        return bool(self.inflight_store or self.inflight_load)

    def reset_cache(self) -> None:
        if self.io is None:
            self._pending_reset = True  # applied when the slabs are attached
            return
        self.epoch += 1
        _atomic_write_text(os.path.join(self._rank0 or "", SLAB_EPOCH), str(self.epoch))
        self.io.epoch = self.epoch
        for g, n in enumerate(self.counts):
            self.index[g].clear()
            self.free[g] = collections.deque(range(n))
        self.inflight_store.clear(); self.inflight_load.clear(); self.siblings.clear()
        self.condemned.clear(); self.failed_store.clear()

    def get_stats(self):
        return None

    def shutdown(self) -> None:
        if self.io is not None:
            self.io.close()


class SlabBounceController(BounceController):
    """BounceController whose disk layer is a SlabIO: stores are one task per
    chunk (sequential slot writes + one fdatasync, off the main thread),
    loads one task per block (parallel preads), both verified by key."""

    def __init__(self, kv_caches: CanonicalKVCaches, slab: SlabIO, n_bounce: int = 48,
                 n_read_threads: int = 8, n_write_threads: int = 8):
        # the FileMapper is unused by the slab layer; a dummy keeps the base happy
        super().__init__(kv_caches, file_mapper=None, n_bounce=n_bounce,
                         n_read_threads=n_read_threads, n_write_threads=n_write_threads)
        self.slab = slab

    def submit(self, job_id: int, store: bool, spec: TransferSpec) -> bool:
        src, dst = spec
        gpu_spec, slab_spec = (src, dst) if store else (dst, src)
        assert isinstance(gpu_spec, GPULoadStoreSpec) and isinstance(slab_spec, SlabLoadStoreSpec), spec
        gpu_ids = [int(b) for b in gpu_spec.block_ids]
        if len(gpu_ids) != len(slab_spec.keys) or not slab_spec.keys:
            if slab_spec.keys:
                logger.error("Slab job %d: %d keys vs %d GPU blocks", job_id, len(slab_spec.keys), len(gpu_ids))
            return False
        job = _DirectJob(job_id=job_id, store=store, items=list(zip(slab_spec.keys, gpu_ids)))
        job.slots = dict(zip(slab_spec.keys, slab_spec.slots))
        job.epoch, job.seq = slab_spec.epoch, slab_spec.seq
        job.nbytes = sum(self.files.block_nbytes(k) for k in slab_spec.keys)
        self.jobs[job_id] = job
        return True

    def _store_chunk(self, job: _DirectJob, slots_keys: list[tuple[int, OffloadKey]]) -> None:
        groups: dict[int, list[int]] = {}
        for bounce_slot, key in slots_keys:
            disk_slot = job.slots[key]
            self.slab.write(key, disk_slot, self.files.block_views(key, bounce_slot), job.seq, epoch=job.epoch)
            groups.setdefault(get_offload_group_idx(key), []).append(disk_slot)
        for g, dslots in groups.items():
            self.slab.sync(g, dslots)

    def _load_one(self, job: _DirectJob, key: OffloadKey, bounce_slot: int) -> None:
        self.slab.read(key, job.slots[key], self.reader.block_views(key, bounce_slot), epoch=job.epoch)

    def _enqueue_store_tasks(self, job: _DirectJob, wid: int, slots_keys: list[tuple[int, OffloadKey]]) -> None:
        self.files.pool.enqueue_store(wid, 1, [functools.partial(self._store_chunk, job, slots_keys)])

    def _enqueue_load_tasks(self, job: _DirectJob, sid: int, keys: list[OffloadKey], slots: list[int]) -> None:
        self.reader.pool.enqueue_load(
            sid, len(keys), [functools.partial(self._load_one, job, k, s) for k, s in zip(keys, slots)]
        )


class MultiNodeSlabOffloadingSpec(OffloadingSpecBase):
    """Fixed-size slab store. extra_config: root_dir (required),
    disk_bytes_per_rank (default 150e9), bounce_blocks (48), n_read_threads /
    n_write_threads (8)."""

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config, kv_cache_config)
        _fix_per_group_block_sizes(self)
        self.root_dir = self.extra_config.get("root_dir")
        if not self.root_dir:
            raise ValueError("MultiNodeSlabOffloadingSpec needs 'root_dir' in kv_connector_extra_config")
        self.disk_bytes_per_rank = int(float(self.extra_config.get("disk_bytes_per_rank", 150e9)))
        self.n_bounce = int(self.extra_config.get("bounce_blocks", 48))
        self.n_read_threads = int(self.extra_config.get("n_read_threads", 8))
        self.n_write_threads = int(self.extra_config.get("n_write_threads", 8))
        self._manager: SlabOffloadingManager | None = None
        self.controller: SlabBounceController | None = None

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = SlabOffloadingManager(self.root_dir, self.vllm_config.model_config.model)
            logger.info("Slab store: scheduler will index rank 0's slabs under %s", self.root_dir)
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        fm = make_file_mapper(self, self.root_dir)
        rank_dir = f"{fm.base_path}_r{fm.rank}"
        os.makedirs(rank_dir, exist_ok=True)
        group_bytes = [
            sum(ref.page_size_bytes * self.block_size_factor for ref in refs)
            for refs in kv_caches.group_data_refs
        ]
        slot_bytes, counts = slab_geometry(group_bytes, self.disk_bytes_per_rank)
        meta_path = os.path.join(rank_dir, SLAB_META)
        boot = _boot_id()
        run_config = fm.get_run_config()
        spec_cfg = getattr(self.vllm_config, "speculative_config", None)
        draft = getattr(spec_cfg, "model", None) if spec_cfg is not None else None
        content_identity = _content_identity(self.vllm_config, draft, self.extra_config)
        try:
            old = json.load(open(meta_path)) if os.path.exists(meta_path) else None
        except (OSError, ValueError):
            old = None
        io = SlabIO(rank_dir, slot_bytes, counts)
        # Cross-reboot persistence is a toggle (default ON): the payload CRC (v2) makes a persisted
        # slot safe to trust, so by default a reboot keeps the slabs. extra_config
        # "persist_across_reboot" (or env SLAB_PERSIST_ACROSS_REBOOT) can turn it off to restore the
        # old wipe-every-boot behaviour. Either way the slabs are wiped when they cannot mean the
        # same thing (format version, geometry, engine build, or run_config).
        persist = self.extra_config.get("persist_across_reboot")
        persist = _slab_persist_default() if persist is None else _slab_persist_value(persist)
        wiped = False
        if _slab_should_wipe(old, slot_bytes, counts, run_config, _VLLM_VERSION, boot, persist, content_identity):
            io.wipe()
            wiped = old is not None
        _atomic_write_json(meta_path, {
            "version": SLAB_META_VERSION, "boot_id": boot, "slot_bytes": slot_bytes, "slot_counts": counts,
            "group_bytes": group_bytes, "disk_bytes_per_rank": self.disk_bytes_per_rank,
            "engine_version": _VLLM_VERSION, "run_config": run_config, "content_identity": content_identity})
        epoch_path = os.path.join(rank_dir, SLAB_EPOCH)
        try:
            io.epoch = int(pathlib_read(epoch_path)) if os.path.exists(epoch_path) else 0
        except ValueError:
            io.epoch = 0   # fallback only: every store/load carries the scheduler's epoch explicitly
        self.controller = SlabBounceController(
            kv_caches, io, n_bounce=self.n_bounce,
            n_read_threads=self.n_read_threads, n_write_threads=self.n_write_threads,
        )
        logger.info(
            "Slab store: rank %d at %s, slot bytes %s, slots %s (%.1f GB cap%s), %d bounce slots",
            fm.rank, rank_dir, slot_bytes, counts,
            sum(b * c for b, c in zip(slot_bytes, counts)) / 1e9,
            ", wiped" if wiped else (", reused across reboot" if persist else ""), self.n_bounce,
        )
        yield GPULoadStoreSpec, SlabLoadStoreSpec, DirectFsHandler(self.controller, store=True)
        yield SlabLoadStoreSpec, GPULoadStoreSpec, DirectFsHandler(self.controller, store=False)


class _FailureAwareScheduler(OffloadingConnectorScheduler):
    """Base scheduler plus: failed jobs reported by any worker mark the
    manager before the base completion path runs."""

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        meta = connector_output.kv_connector_worker_meta
        failed = set(getattr(meta, "failed_jobs", ()) or ())
        if failed:
            for job_id in failed:
                status = self._jobs.get(job_id)
                if status is None:
                    continue
                if status.is_store:
                    self.manager.mark_store_failed(status.keys)  # type: ignore[attr-defined]
                else:
                    self.manager.mark_load_failed(status.keys)  # type: ignore[attr-defined]
        super().update_connector_output(connector_output)


class MultiNodeSlabConnectorWorker(MultiNodeDirectConnectorWorker):
    """Direct worker plus the failed_jobs channel to the scheduler."""

    def __init__(self, spec):
        super().__init__(spec)
        self._connector_worker_meta = MultiNodeWorkerMetadata()

    def _meta(self) -> MultiNodeWorkerMetadata:
        if not isinstance(self._connector_worker_meta, MultiNodeWorkerMetadata):
            self._connector_worker_meta = MultiNodeWorkerMetadata(
                completed_jobs=self._connector_worker_meta.completed_jobs,
                transfer_stats=self._connector_worker_meta.transfer_stats,
            )
        return self._connector_worker_meta

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_recving: set[str] = set()
        meta = self._meta()
        for result in self.worker.get_finished():
            job_id = result.job_id
            if result.success and result.transfer_time is not None and result.transfer_size is not None:
                is_load = job_id in self._load_jobs
                stats = meta.transfer_stats.load if is_load else meta.transfer_stats.store
                stats.record(result.transfer_size, result.transfer_time)
            elif not result.success:
                meta.failed_jobs.add(job_id)
                logger.warning("Slab store: job %d failed on this rank", job_id)
            meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
        ctrl = getattr(self.spec, "controller", None)
        if ctrl is not None:
            self._invalid_block_ids |= ctrl.take_failed_gpu_blocks()
        return set(), finished_recving

    def build_connector_worker_meta(self):
        meta = self._meta()
        if not meta.completed_jobs:
            return None
        self._connector_worker_meta = MultiNodeWorkerMetadata()
        return meta


class MultiNodeSlabConnector(OffloadingConnector):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)
        if not isinstance(spec, MultiNodeSlabOffloadingSpec):
            raise ValueError("MultiNodeSlabConnector requires spec_name=MultiNodeSlabOffloadingSpec")
        self.connector_scheduler = None
        self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = _FailureAwareScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = MultiNodeSlabConnectorWorker(spec)

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.connector_worker is None:
            return set()
        return self.connector_worker.take_invalid_block_ids()


# --------------------------------------------------------------------------
# Connector: scheduler and worker subclasses that carry the tier jobs
# --------------------------------------------------------------------------


class MultiNodeOffloadingConnectorScheduler(OffloadingConnectorScheduler):
    def __init__(self, spec: MultiNodeTieringOffloadingSpec):
        super().__init__(spec)  # calls spec.get_manager()
        self._tiers: list[WorkerFileSystemTier] = list(spec.worker_tiers)
        # connector job id -> [tier, tier job id, workers still pending, failed]
        self._tier_jobs: dict[int, list[Any]] = {}

    def build_connector_meta(self, scheduler_output: "SchedulerOutput"):
        meta = super().build_connector_meta(scheduler_output)
        assert isinstance(meta, OffloadingConnectorMetadata)
        tier_jobs: dict[int, TransferSpec] = {}
        for tier in self._tiers:
            for tier_job_id, spec in tier.take_pending():
                job_id = self._generate_job_id()
                self._tier_jobs[job_id] = [tier, tier_job_id, self.config.num_workers, False]
                tier_jobs[job_id] = spec
        return MultiNodeConnectorMetadata(
            load_jobs=meta.load_jobs,
            store_jobs=meta.store_jobs,
            jobs_to_flush=meta.jobs_to_flush,
            tier_jobs=tier_jobs,
        )

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        meta = connector_output.kv_connector_worker_meta
        if isinstance(meta, OffloadingWorkerMetadata) and self._tier_jobs:
            failed = set(getattr(meta, "failed_jobs", ()))
            rest: dict[int, int] = {}
            for job_id, count in meta.completed_jobs.items():
                entry = self._tier_jobs.get(job_id)
                if entry is None:
                    rest[job_id] = count
                    continue
                entry[2] -= count
                if job_id in failed:
                    entry[3] = True
                if entry[2] <= 0:
                    tier, tier_job_id, _, did_fail = entry
                    del self._tier_jobs[job_id]
                    if did_fail:
                        logger.warning("Worker fs tier job %d failed on a rank", job_id)
                    tier.mark_finished(tier_job_id, not did_fail)
            meta = OffloadingWorkerMetadata(
                completed_jobs=rest, transfer_stats=meta.transfer_stats
            )
            connector_output = dataclasses.replace(
                connector_output, kv_connector_worker_meta=meta
            )
        super().update_connector_output(connector_output)

    def has_pending_push_work(self) -> bool:
        return bool(self._tier_jobs) or super().has_pending_push_work()


class MultiNodeOffloadingConnectorWorker(OffloadingConnectorWorker):
    def __init__(self, spec: MultiNodeTieringOffloadingSpec):
        super().__init__(spec)
        self._tier_job_ids: set[int] = set()
        self._connector_worker_meta = MultiNodeWorkerMetadata()

    def _meta(self) -> MultiNodeWorkerMetadata:
        if not isinstance(self._connector_worker_meta, MultiNodeWorkerMetadata):
            self._connector_worker_meta = MultiNodeWorkerMetadata(
                completed_jobs=self._connector_worker_meta.completed_jobs,
                transfer_stats=self._connector_worker_meta.transfer_stats,
            )
        return self._connector_worker_meta

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        super().start_kv_transfers(metadata)
        for job_id, spec in getattr(metadata, "tier_jobs", {}).items():
            if self.worker.transfer_async(job_id, spec):
                self._tier_job_ids.add(job_id)
            else:
                meta = self._meta()
                meta.mark_completed(job_id)
                meta.failed_jobs.add(job_id)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_recving: set[str] = set()
        meta = self._meta()
        for result in self.worker.get_finished():
            job_id = result.job_id
            is_tier = job_id in self._tier_job_ids
            if not result.success:
                assert is_tier, f"non-tier transfer {job_id} failed"
                meta.failed_jobs.add(job_id)
            if (
                result.success
                and result.transfer_time is not None
                and result.transfer_size is not None
            ):
                is_load = job_id in self._load_jobs or (
                    is_tier and result.transfer_type == ("FS", "CPU")
                )
                stats = meta.transfer_stats.load if is_load else meta.transfer_stats.store
                stats.record(result.transfer_size, result.transfer_time)
            meta.mark_completed(job_id)
            self._tier_job_ids.discard(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
        return set(), finished_recving

    def build_connector_worker_meta(self):
        meta = self._meta()
        if not meta.completed_jobs:
            return None
        self._connector_worker_meta = MultiNodeWorkerMetadata()
        return meta



class MultiNodeOffloadingConnector(OffloadingConnector):
    """``OffloadingConnector`` whose scheduler/worker halves carry the
    worker-executed tier jobs. Selected with ``kv_connector_module_path``."""

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)
        if not isinstance(spec, MultiNodeTieringOffloadingSpec):
            raise ValueError(
                "MultiNodeOffloadingConnector requires spec_name="
                "MultiNodeTieringOffloadingSpec"
            )
        self.connector_scheduler = None
        self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MultiNodeOffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = MultiNodeOffloadingConnectorWorker(spec)
