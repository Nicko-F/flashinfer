# SPDX-License-Identifier: Apache-2.0
"""Runnable module example; benchmarks include assignment for prepared plans."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from flashinfer.decode import xqa_batch_decode_with_kv_cache
from flashinfer.testing import bench_gpu_time
from flashinfer.xqa import create_xqa_decode_bucket_plan, xqa_decode_with_buckets


def compare_planners(args):
    """Static metadata, shared operands; host planning stays outside Graphs."""
    from flashinfer.experimental.xqa_decode_buckets.jit import get_module

    records = []
    scenarios = (
        ("balanced_b32_adapted_from_tencent_b64", [4096] * 32),
        ("skew_b8", [65536] + [4096] * 7),
        ("skew_b32", [131072] + [4096] * 31),
        ("short_direct_control", [512]),
    )
    stream = torch.cuda.current_stream()

    def wall_samples(call):
        for _ in range(5):
            call()
            stream.synchronize()
        samples = []
        for _ in range(20):
            start = time.perf_counter_ns()
            call()
            stream.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1000)
        return samples

    def graph_samples(call):
        return [
            value * 1000
            for value in bench_gpu_time(
                call,
                enable_cupti=False,
                use_cuda_graph=True,
                cold_l2_cache=False,
                dry_run_iters=5,
                repeat_iters=20,
            )
        ]

    for label, lengths in scenarios:
        torch.manual_seed(args.seed)
        batch, width = (
            len(lengths),
            (max(lengths) + args.page_size - 1) // args.page_size,
        )
        cpu_lengths = torch.tensor(lengths, dtype=torch.int32)
        lens = cpu_lengths.to(args.device)
        table = torch.zeros((batch, width), dtype=torch.int32, device=args.device)
        pages = 0
        for row, length in enumerate(lengths):
            count = (length + args.page_size - 1) // args.page_size
            table[row, :count] = torch.arange(pages, pages + count, device=args.device)
            pages += count
        plans = {
            planner: create_xqa_decode_bucket_plan(
                lens,
                page_table_width=width,
                page_size=args.page_size,
                num_kv_heads=args.num_kv_heads,
                head_group_ratio=args.head_group_ratio,
                mode=args.mode,
                planner=planner,
                host_seq_lens=cpu_lengths if planner == "host" else None,
            )
            for planner in ("gpu", "host")
        }
        bank = []
        for _ in range(args.layers):
            q = (
                torch.randn(
                    (batch, args.num_kv_heads * args.head_group_ratio, 256),
                    dtype=torch.bfloat16,
                    device=args.device,
                )
                * 0.25
            )
            kv = [
                torch.randint(
                    0,
                    256,
                    (pages, args.page_size, args.num_kv_heads, 128),
                    dtype=torch.uint8,
                    device=args.device,
                )
                for _ in range(2)
            ]
            sf = [
                torch.ones(
                    (pages, args.page_size, args.num_kv_heads, 16),
                    dtype=torch.float8_e4m3fn,
                    device=args.device,
                )
                for _ in range(2)
            ]
            outputs = {p: torch.empty_like(q) for p in plans}
            workspaces = {p: plans[p].create_workspace() for p in plans}
            bank.append((q, kv, sf, outputs, workspaces))

        def compute(planner):
            for q, kv, sf, outputs, workspaces in bank:
                xqa_decode_with_buckets(
                    q,
                    *kv,
                    *sf,
                    table,
                    plans[planner],
                    workspaces[planner],
                    bmm1_scale=0.7 / 16,
                    bmm2_scale=1.3,
                    out=outputs[planner],
                )

        compute_graphs = {}
        for planner in plans:
            compute(planner)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compute(planner)
            compute_graphs[planner] = graph
        full_gpu = torch.cuda.CUDAGraph()
        with torch.cuda.graph(full_gpu):
            plans["gpu"].prepare()
            compute("gpu")

        def external(planner):
            plans[planner].prepare()
            compute_graphs[planner].replay()

        samples = {}
        # Bracket the host arm with the GPU arm on the same data and placement.
        for name, call in (
            ("gpu_captured_completed_wall_before_us", full_gpu.replay),
            ("host_prepare_h2d_graph_completed_wall_us", lambda: external("host")),
            ("gpu_captured_completed_wall_after_us", full_gpu.replay),
            ("gpu_external_prepare_graph_completed_wall_us", lambda: external("gpu")),
        ):
            samples[name] = wall_samples(call)
        for planner in compute_graphs:
            samples[planner + "_compute_only_graph_gpu_us"] = graph_samples(
                lambda planner=planner: compute(planner)
            )

        def gpu_window():
            plans["gpu"].prepare()
            compute("gpu")

        samples["gpu_assign_compute_graph_gpu_us"] = graph_samples(gpu_window)
        host = plans["host"]
        if not host.direct:
            module, _ = get_module(
                args.mode == "persistent",
                lens.device,
                page_size=args.page_size,
                head_group_ratio=args.head_group_ratio,
            )
            staging = torch.empty_like(host._host_descriptors, pin_memory=True)
            fill = lambda: module.plan_host(
                cpu_lengths, staging, host.ctas, width, args.mode == "aligned"
            )
            for _ in range(5):
                fill()
            fill_samples = []
            for _ in range(20):
                start = time.perf_counter_ns()
                for _ in range(20):
                    fill()
                fill_samples.append((time.perf_counter_ns() - start) / 20000)
            samples["host_cpp_ffi_fill_cpu_us"] = fill_samples
            samples["descriptor_h2d_completed_wall_us"] = wall_samples(
                lambda: host._device_descriptors.copy_(staging, non_blocking=True)
            )
        external("host")
        full_gpu.replay()
        stream.synchronize()
        errors = []
        for _, _, _, outputs, workspaces in bank:
            error = (outputs["host"].float() - outputs["gpu"].float()).norm() / outputs[
                "gpu"
            ].float().norm().clamp_min(1e-8)
            errors.append(error.item())
            if not torch.isfinite(outputs["host"]).all() or errors[-1] >= 0.03:
                raise RuntimeError("Host/GPU output check failed")
            if any(workspace.semaphores.any() for workspace in workspaces.values()):
                raise RuntimeError("Counter reset check failed")
        record = dict(
            scenario=label,
            num_kv_heads=args.num_kv_heads,
            head_group_ratio=args.head_group_ratio,
            page_size=args.page_size,
            lengths=lengths,
            layers=args.layers,
            mode=args.mode,
            seed=args.seed,
            direct=host.direct,
            ctas=host.ctas,
            resources=host.resources,
            descriptor_bytes=0 if host.direct else host._host_descriptors.numel() * 4,
            cpu_metadata="caller-provided exact CPU lengths; unchanged during timing",
            length_upload="excluded equally: static metadata already resident for both arms",
            graph_scope="compute-only except gpu_assign_compute graph; host prepare is external",
            samples=samples,
            medians={key: statistics.median(v) for key, v in samples.items()},
            host_vs_gpu_relative_rms=errors,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
        if args.output_json:
            target = Path(args.output_json)
            temporary = target.with_suffix(target.suffix + ".partial")
            temporary.write_text(
                json.dumps(dict(complete=False, records=records), indent=2)
            )
            temporary.replace(target)
    if args.output_json:
        target = Path(args.output_json)
        temporary = target.with_suffix(target.suffix + ".partial")
        temporary.write_text(json.dumps(dict(complete=True, records=records), indent=2))
        temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="131072,4096,4096,4096")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--page-size", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--head-group-ratio", type=int, choices=(4, 6, 8), default=6)
    parser.add_argument(
        "--capacity-length", type=int, help="Reserved table capacity in tokens"
    )
    parser.add_argument(
        "--mode", choices=("persistent", "aligned"), default="persistent"
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--compare-planners", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Time planner and compute windows separately",
    )
    args = parser.parse_args()
    if args.num_kv_heads < 1:
        parser.error("--num-kv-heads must be positive")
    if args.compare_planners:
        if args.layers < 1:
            parser.error("Use at least one layer")
        device = torch.device(args.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        args.device = device
        torch.cuda.set_device(device)
        compare_planners(args)
        return
    lengths = [int(n) for n in args.lengths.split(",")]
    if not 1 <= len(lengths) <= 32 or min(lengths) < 1 or args.layers < 1:
        parser.error("Use 1-32 positive lengths and at least one layer")
    capacity = max(lengths) if args.capacity_length is None else args.capacity_length
    if capacity < max(lengths):
        parser.error("--capacity-length must cover every live length")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    counts = [(n + args.page_size - 1) // args.page_size for n in lengths]
    width = (capacity + args.page_size - 1) // args.page_size
    table = torch.zeros((len(lengths), width), dtype=torch.int32, device=device)
    offset = 0
    for row, count in enumerate(counts):
        table[row, :count] = torch.arange(offset, offset + count, device=device)
        offset += count
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    plan = create_xqa_decode_bucket_plan(
        lens,
        page_table_width=width,
        page_size=args.page_size,
        mode=args.mode,
        num_kv_heads=args.num_kv_heads,
        head_group_ratio=args.head_group_ratio,
    )
    bank = []
    for _ in range(args.layers):
        q = (
            torch.randn(
                (len(lengths), args.num_kv_heads * args.head_group_ratio, 256),
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.25
        )
        k, v = [
            torch.randint(
                0,
                256,
                (offset, args.page_size, args.num_kv_heads, 128),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(2)
        ]
        ks, vs = [
            torch.ones(
                (offset, args.page_size, args.num_kv_heads, 16),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            for _ in range(2)
        ]
        out, native_out = torch.empty_like(q), torch.empty_like(q)
        workspace = plan.create_workspace()
        native_scratch = torch.zeros(40 * 2**20, dtype=torch.uint8, device=device)
        bank.append((q, k, v, ks, vs, out, native_out, workspace, native_scratch))

    def compute():
        for q, k, v, ks, vs, out, _, workspace, _ in bank:
            xqa_decode_with_buckets(
                q,
                k,
                v,
                ks,
                vs,
                table,
                plan,
                workspace,
                bmm1_scale=0.7 / 16,
                bmm2_scale=1.3,
                out=out,
            )

    def candidate():
        plan.prepare()
        compute()

    def native():
        for q, k, v, ks, vs, _, native_out, _, scratch in bank:
            xqa_batch_decode_with_kv_cache(
                q,
                (k, v),
                scratch,
                table,
                lens,
                max(lengths),
                bmm1_scale=0.7 / 16,
                bmm2_scale=1.3,
                out=native_out,
                kv_cache_sf=(ks, vs),
            )

    native()
    candidate()
    torch.cuda.synchronize()

    def check_outputs():
        errors = []
        for _, _, _, _, _, out, native_out, workspace, _ in bank:
            error = (
                out.float() - native_out.float()
            ).norm() / native_out.float().norm().clamp_min(1e-8)
            errors.append(error.item())
            if (
                not torch.isfinite(out).all()
                or errors[-1] >= 0.03
                or workspace.semaphores.any()
            ):
                raise RuntimeError("Module output/counter check failed")
        return errors

    check_outputs()
    print(
        dict(
            device=str(device),
            lengths=lengths,
            layers=args.layers,
            mode=args.mode,
            seed=args.seed,
            resources=plan.resources,
            capacity_length=width * args.page_size,
            page_size=args.page_size,
            num_kv_heads=args.num_kv_heads,
            head_group_ratio=args.head_group_ratio,
            direct=plan.direct,
            assignment_per_window=0 if plan.direct else 1,
        ),
        flush=True,
    )
    if args.diagnose:
        if plan.direct:
            # Explicitly derived outside timing; not a profiler launch record.
            live_lengths = plan.lengths.cpu().tolist()
            per_request = plan.ctas // len(live_lengths)
            request_bounds = []
            for request, live in enumerate(live_lengths):
                tiles = (live + 255) // 256
                parts = min(tiles, per_request) if len(live_lengths) == 1 else tiles
                base = request * per_request
                request_bounds.append(
                    dict(
                        request=request,
                        live_length=live,
                        partial_begin_group0=base,
                        partial_count=parts,
                        group_stride=parts if len(live_lengths) == 1 else plan.ctas,
                        total_slots=(parts if len(live_lengths) == 1 else plan.ctas)
                        * args.num_kv_heads,
                        intervals=[
                            [i * tiles // parts, (i + 1) * tiles // parts]
                            for i in range(parts)
                        ],
                    )
                )
            topology = dict(
                source="derived from live device length outside timed region",
                representation="direct_empty_descriptors",
                request_bounds_derived=request_bounds,
                grid=[per_request, args.num_kv_heads, len(live_lengths)],
                ctas_per_group=plan.ctas,
                grid_ctas=plan.ctas * args.num_kv_heads,
                nonempty_tile_ctas_derived=sum(
                    r["partial_count"] for r in request_bounds
                )
                * args.num_kv_heads,
            )
            print(
                dict(
                    arm="assign_only",
                    gpu_launches=0,
                    reason="direct prepare is a no-op",
                ),
                flush=True,
            )
        else:
            header = plan.header.cpu().tolist()
            topology = dict(
                source="prepared device descriptors",
                header=header,
                tasks=plan.tasks[: header[1]].cpu().tolist(),
                requests=plan.requests.cpu().tolist(),
                grid_ctas=plan.ctas * args.num_kv_heads,
                ctas_per_group=plan.ctas,
                active_ctas=header[2] * args.num_kv_heads,
            )
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        splits = min(
            max(1, sms // (len(lengths) * args.num_kv_heads)),
            (width * args.page_size + 255) // 256,
        )
        print(
            dict(
                plan_topology=topology,
                native_launch_derived=dict(
                    source="csrc/xqa/mha.cu:3211-3222",
                    formula="splits=min(max(1,SMs/(B*KVheads)),ceil(table_width*page_size/256))",
                    sm_count=sms,
                    kv_heads=args.num_kv_heads,
                    grid=[splits, args.num_kv_heads, len(lengths)],
                    grid_ctas=splits * len(lengths) * args.num_kv_heads,
                    nonempty_tile_ctas=sum(
                        min(splits, (n + 255) // 256) for n in lengths
                    )
                    * args.num_kv_heads,
                ),
                compute_math="native QK/PV; transposed compute is not implemented",
            ),
            flush=True,
        )
    if args.benchmark or args.diagnose:
        # Warm both and bracket the candidate with native; report absolute samples.
        windows = (
            ("native_before", native),
            ("candidate", candidate),
            ("native_after", native),
        )
        if args.diagnose:
            assignment_windows = () if plan.direct else (("assign_only", plan.prepare),)
            windows = (
                ("native_before", native),
                *assignment_windows,
                ("compute_only", compute),
                ("assign_and_compute", candidate),
                ("native_after", native),
            )
        for label, call in windows:
            times = bench_gpu_time(
                call,
                enable_cupti=False,
                use_cuda_graph=True,
                cold_l2_cache=False,
                dry_run_iters=5,
                repeat_iters=20,
            )
            print(
                dict(
                    arm=label,
                    median_window_us=1000 * statistics.median(times),
                    samples_window_us=[1000 * t for t in times],
                ),
                flush=True,
            )
        if args.diagnose:
            print(
                dict(diagnostic_relative_rms=check_outputs(), counters_reset=True),
                flush=True,
            )


if __name__ == "__main__":
    main()
