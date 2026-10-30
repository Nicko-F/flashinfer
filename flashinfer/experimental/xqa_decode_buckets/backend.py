# SPDX-License-Identifier: Apache-2.0
"""Compact plans, direct tile geometry, and independent native partial workspaces."""

from dataclasses import dataclass
import math

import torch

from ...utils import (
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    register_custom_op,
    register_fake_op,
)
from .jit import get_module
from .geometry import plan_capacity, plan_geometry


def _tensor(tensor, name, *, dtype, shape, device, contiguous=True, alignment=1):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.device != device or tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA device {device}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must have {alignment}-byte aligned storage")


def _host_lengths(lengths, batch):
    if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.int32:
        raise TypeError("host_seq_lens must be a CPU int32 tensor")
    if lengths.device.type != "cpu" or tuple(lengths.shape) != (batch,):
        raise ValueError(f"host_seq_lens must have CPU shape ({batch},)")
    if not lengths.is_contiguous():
        raise ValueError("host_seq_lens must be contiguous")


def _descriptor_views(storage, batch, ctas):
    capacity = ctas + batch - 1
    tasks_end = 4 * capacity
    requests_end = tasks_end + 2 * batch
    return (
        storage[:tasks_end].view(capacity, 4),
        storage[tasks_end:requests_end].view(batch, 2),
        storage[requests_end:-4].view(ctas, 2),
        storage[-4:],
    )


@register_custom_op(
    "flashinfer::experimental_xqa_bucket_plan",
    mutates_args=("tasks", "requests", "spans", "header"),
)
def _prepare(
    lengths: torch.Tensor,
    tasks: torch.Tensor,
    requests: torch.Tensor,
    spans: torch.Tensor,
    header: torch.Tensor,
    ctas: int,
    aligned: bool,
) -> None:
    module, _ = get_module(not aligned, lengths.device)
    module.plan(lengths, tasks, requests, spans, header, ctas, aligned)


@register_fake_op("flashinfer::experimental_xqa_bucket_plan")
def _fake_prepare(
    lengths: torch.Tensor,
    tasks: torch.Tensor,
    requests: torch.Tensor,
    spans: torch.Tensor,
    header: torch.Tensor,
    ctas: int,
    aligned: bool,
) -> None:
    pass


@register_custom_op(
    "flashinfer::experimental_xqa_bucket_compute",
    mutates_args=("semaphores", "scratch", "output"),
)
def _compute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    tasks: torch.Tensor,
    requests: torch.Tensor,
    spans: torch.Tensor,
    header: torch.Tensor,
    semaphores: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    ctas: int,
    aligned: bool,
    bmm1_scale: float,
    bmm2_scale: float,
    enable_pdl: bool,
) -> None:
    # Empty headers are the factory's explicit static direct-plan representation.
    direct = header.numel() == 0
    module, _ = get_module(not aligned and not direct, q.device, direct=direct)
    module.run(
        q,
        k,
        v,
        ks,
        vs,
        table,
        lengths,
        tasks,
        requests,
        spans,
        header,
        semaphores,
        scratch,
        output,
        ctas,
        aligned,
        bmm1_scale,
        bmm2_scale,
        enable_pdl,
    )


@register_fake_op("flashinfer::experimental_xqa_bucket_compute")
def _fake_compute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    tasks: torch.Tensor,
    requests: torch.Tensor,
    spans: torch.Tensor,
    header: torch.Tensor,
    semaphores: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    ctas: int,
    aligned: bool,
    bmm1_scale: float,
    bmm2_scale: float,
    enable_pdl: bool,
) -> None:
    pass


@dataclass(frozen=True)
class BucketWorkspace:
    """Mutable native partial/counter storage for ordered compute calls."""

    scratch: torch.Tensor
    semaphores: torch.Tensor


@dataclass(frozen=True)
class BucketPlan:
    """Fixed buffer addresses; refresh device contents once per forward.

    Direct plans use geometry inside compute; prepare() is a no-op and all four
    descriptor tensors are empty. They apply to B=1 or when the static table's
    total tile bound fits the device SM count. Both mode names select the same
    one-task-per-CTA specialization for direct plans. Otherwise call prepare()
    once per forward after updating lengths, before its compute consumers.
    Compatible layers share the same request ordering and lengths. Replays and
    refreshes must be ordered on the same stream or by explicit CUDA events.
    Separate concurrent computes require independent BucketWorkspace instances.

    Host plans retain caller-owned CPU host_seq_lens; callers keep its values
    identical to lengths without a device readback. Host prepare runs outside
    capture before each compute-only Graph replay. One pinned staging backing is
    reused only after its previous asynchronous H2D finishes. This protects the
    staging lifetime; stream ordering still protects the device descriptors.
    """

    lengths: torch.Tensor
    tasks: torch.Tensor
    requests: torch.Tensor
    spans: torch.Tensor
    header: torch.Tensor
    ctas: int
    table_width: int
    mode: str
    scratch_bytes: int
    resources: tuple
    direct: bool
    planner: str
    host_seq_lens: torch.Tensor | None
    _host_descriptors: torch.Tensor | None
    _device_descriptors: torch.Tensor | None
    _copy_done: torch.cuda.Event | None

    def prepare(self):
        if self.direct:
            return self
        with torch.cuda.device(self.lengths.device):
            if self.planner == "host":
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Host prepare must run before Graph replay, outside capture"
                    )
                _host_lengths(self.host_seq_lens, self.lengths.numel())
                # A single bounded staging buffer; never overwrite an in-flight
                # DMA source. This waits only when the prior copy is unfinished.
                if not self._copy_done.query():
                    self._copy_done.synchronize()
                module, _ = get_module(self.mode == "persistent", self.lengths.device)
                module.plan_host(
                    self.host_seq_lens,
                    self._host_descriptors,
                    self.ctas,
                    self.table_width,
                    self.mode == "aligned",
                )
                self._device_descriptors.copy_(
                    self._host_descriptors, non_blocking=True
                )
                self._copy_done.record()
                return self
            _prepare(
                self.lengths,
                self.tasks,
                self.requests,
                self.spans,
                self.header,
                self.ctas,
                self.mode == "aligned",
            )
        return self

    def create_workspace(self):
        """Allocate cold, before capture; sequential layers may reuse this workspace."""
        return BucketWorkspace(
            torch.empty(
                self.scratch_bytes, dtype=torch.uint8, device=self.lengths.device
            ),
            torch.zeros(
                self.lengths.numel(), dtype=torch.int32, device=self.lengths.device
            ),
        )


def create_plan(
    seq_lens, *, page_table_width, mode="persistent", planner="gpu", host_seq_lens=None
):
    if mode not in ("persistent", "aligned"):
        raise ValueError("mode must be persistent or aligned")
    if planner not in ("gpu", "host"):
        raise ValueError("planner must be gpu or host")
    if not isinstance(seq_lens, torch.Tensor) or seq_lens.ndim != 1:
        raise ValueError("seq_lens must be a 1D CUDA int32 tensor")
    device = seq_lens.device
    _tensor(
        seq_lens, "seq_lens", dtype=torch.int32, shape=seq_lens.shape, device=device
    )
    if get_compute_capability(device) != (12, 0):
        raise ValueError("XQA Decode buckets currently support SM120")
    batch = seq_lens.numel()
    if planner == "host":
        _host_lengths(host_seq_lens, batch)
    elif host_seq_lens is not None:
        raise ValueError("host_seq_lens is only used with planner='host'")
    ctas, direct = plan_geometry(batch, page_table_width, get_device_sm_count(device))
    capacity = plan_capacity(batch, ctas, page_table_width)
    module, resources = get_module(
        mode == "persistent" and not direct, device, direct=direct
    )
    host_descriptors = device_descriptors = copy_done = None
    if planner == "host" and not direct:
        words = 6 * (ctas + batch)
        host_descriptors = torch.empty(words, dtype=torch.int32, pin_memory=True)
        device_descriptors = torch.empty(words, dtype=torch.int32, device=device)
        tasks, requests, spans, header = _descriptor_views(
            device_descriptors, batch, ctas
        )
        copy_done = torch.cuda.Event()
    else:
        tasks = torch.empty(
            (0 if direct else capacity, 4), dtype=torch.int32, device=device
        )
        requests = torch.empty(
            (0 if direct else batch, 2), dtype=torch.int32, device=device
        )
        spans = torch.empty(
            (0 if direct else ctas, 2), dtype=torch.int32, device=device
        )
        header = torch.empty(0 if direct else 4, dtype=torch.int32, device=device)
    plan = BucketPlan(
        seq_lens,
        tasks,
        requests,
        spans,
        header,
        ctas,
        page_table_width,
        mode,
        module.scratch_bytes(capacity),
        resources,
        direct,
        planner,
        host_seq_lens,
        host_descriptors,
        device_descriptors,
        copy_done,
    )
    # The initial generation is usable immediately. Later forwards refresh it explicitly.
    return plan.prepare()


def run(
    query,
    k_cache,
    v_cache,
    k_sf_cache,
    v_sf_cache,
    page_table,
    plan,
    workspace,
    *,
    bmm1_scale,
    bmm2_scale,
    out=None,
    enable_pdl=None,
):
    if not isinstance(plan, BucketPlan) or not isinstance(workspace, BucketWorkspace):
        raise TypeError("Expected a bucket plan and its caller-owned workspace")
    device, batch = plan.lengths.device, plan.lengths.numel()
    shape = (batch, 6, 256)
    _tensor(
        query, "query", dtype=torch.bfloat16, shape=shape, device=device, alignment=16
    )
    _tensor(
        page_table,
        "page_table",
        dtype=torch.int32,
        shape=(batch, plan.table_width),
        device=device,
    )
    if k_cache.ndim != 4:
        raise ValueError("Expected packed NHD page32 NVFP4 cache")
    pages = k_cache.shape[0]
    for name, value, dtype, width, unit in (
        ("k_cache", k_cache, torch.uint8, 128, 128),
        ("v_cache", v_cache, torch.uint8, 128, 128),
        ("k_sf_cache", k_sf_cache, torch.float8_e4m3fn, 16, 16),
        ("v_sf_cache", v_sf_cache, torch.float8_e4m3fn, 16, 16),
    ):
        _tensor(
            value,
            name,
            dtype=dtype,
            shape=(pages, 32, 1, width),
            device=device,
            contiguous=False,
            alignment=16,
        )
        if value.stride(-1) != 1 or value.stride(0) % unit or value.stride(1) % unit:
            raise ValueError(f"{name} must preserve native packed-head byte strides")
    if (
        k_cache.stride() != v_cache.stride()
        or k_sf_cache.stride() != v_sf_cache.stride()
    ):
        raise ValueError("K/V and their block scales must use matching strides")
    _tensor(
        workspace.scratch,
        "scratch",
        dtype=torch.uint8,
        shape=(plan.scratch_bytes,),
        device=device,
        alignment=16,
    )
    _tensor(
        workspace.semaphores,
        "semaphores",
        dtype=torch.int32,
        shape=(batch,),
        device=device,
    )
    if not all(math.isfinite(float(s)) for s in (bmm1_scale, bmm2_scale)):
        raise ValueError("BMM1 and BMM2 scales must be finite")
    if out is None:
        out = torch.empty_like(query)
    _tensor(out, "out", dtype=torch.bfloat16, shape=shape, device=device, alignment=16)
    if any(
        out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
        for t in (query, k_cache, v_cache, k_sf_cache, v_sf_cache, page_table)
    ):
        raise ValueError("out must not alias query, cache or metadata storage")
    if enable_pdl is None:
        enable_pdl = device_support_pdl(device)
    if type(enable_pdl) is not bool:
        raise TypeError("enable_pdl must be a boolean or None")
    with torch.cuda.device(device):
        _compute(
            query,
            k_cache,
            v_cache,
            k_sf_cache,
            v_sf_cache,
            page_table,
            plan.lengths,
            plan.tasks,
            plan.requests,
            plan.spans,
            plan.header,
            workspace.semaphores,
            workspace.scratch,
            out,
            plan.ctas,
            plan.mode == "aligned",
            float(bmm1_scale),
            float(bmm2_scale),
            enable_pdl,
        )
    return out
