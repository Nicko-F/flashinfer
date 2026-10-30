# SPDX-License-Identifier: Apache-2.0
"""Runnable module example; --benchmark counts one plan per compatible layer window."""

import argparse
import statistics

import torch

from flashinfer.decode import xqa_batch_decode_with_kv_cache
from flashinfer.testing import bench_gpu_time
from flashinfer.xqa import create_xqa_decode_bucket_plan, xqa_decode_with_buckets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="131072,4096,4096,4096")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("persistent", "aligned"), default="persistent"
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Time planner and compute windows separately",
    )
    args = parser.parse_args()
    lengths = [int(n) for n in args.lengths.split(",")]
    if not 1 <= len(lengths) <= 32 or min(lengths) < 1 or args.layers < 1:
        parser.error("Use 1-32 positive lengths and at least one layer")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    counts = [(n + 31) // 32 for n in lengths]
    width = max(counts)
    table = torch.zeros((len(lengths), width), dtype=torch.int32, device=device)
    offset = 0
    for row, count in enumerate(counts):
        table[row, :count] = torch.arange(offset, offset + count, device=device)
        offset += count
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    plan = create_xqa_decode_bucket_plan(lens, page_table_width=width, mode=args.mode)
    bank = []
    for _ in range(args.layers):
        q = (
            torch.randn((len(lengths), 6, 256), dtype=torch.bfloat16, device=device)
            * 0.25
        )
        k, v = [
            torch.randint(
                0, 256, (offset, 32, 1, 128), dtype=torch.uint8, device=device
            )
            for _ in range(2)
        ]
        ks, vs = [
            torch.ones((offset, 32, 1, 16), dtype=torch.float8_e4m3fn, device=device)
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
            assignment_per_window=1,
        ),
        flush=True,
    )
    if args.diagnose:
        header = plan.header.cpu().tolist()
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        splits = min(max(1, sms // len(lengths)), (max(lengths) + 255) // 256)
        print(
            dict(
                plan_topology=dict(
                    header=header,
                    tasks=plan.tasks[: header[1]].cpu().tolist(),
                    requests=plan.requests.cpu().tolist(),
                    grid_ctas=plan.ctas,
                    active_ctas=header[2],
                ),
                native_launch_derived=dict(
                    source="csrc/xqa/mha.cu:3211-3222",
                    formula="splits=min(max(1,SMs/(B*KVheads)),ceil(maxlen/256))",
                    sm_count=sms,
                    kv_heads=1,
                    grid=[splits, 1, len(lengths)],
                    grid_ctas=splits * len(lengths),
                    nonempty_tile_ctas=sum(
                        min(splits, (n + 255) // 256) for n in lengths
                    ),
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
            windows = (
                ("native_before", native),
                ("assign_only", plan.prepare),
                ("compute_only", compute),
                ("assign_and_compute", candidate),
                ("native_after", native),
            )
        for label, call in windows:
            times = bench_gpu_time(
                call,
                enable_cupti=False,
                use_cuda_graph=True,
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
