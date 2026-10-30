# SPDX-License-Identifier: Apache-2.0
"""CPU arithmetic and SM120 native-XQA bucket behavior tests."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch


def _check_aligned_boundaries(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise unittest.SkipTest(
            "A C++17 host compiler is needed for the shared arithmetic test"
        )
    header = (
        Path(__file__).resolve().parents[2] / "flashinfer/experimental/"
        "xqa_decode_buckets/csrc/task_geometry.h"
    )
    source = tmp_path / "geometry.cpp"
    source.write_text(f'''#include "{header.as_posix()}"
#include <cassert>
#include <cstdint>
#include <initializer_list>
int main() {{
  using flashinfer::xqa_buckets::aligned_boundary;
  using flashinfer::xqa_buckets::direct_boundary;
  using flashinfer::xqa_buckets::direct_split_count;
  using flashinfer::xqa_buckets::direct_tile_count;
  assert(direct_split_count(0, 2) == 0);
  assert(direct_split_count(512, 170) == 2);
  assert(direct_boundary(13, 512, 512) == 13);
  assert(direct_boundary(512, 512, 512) == 512);
  for (int length : {{0, 1, 255, 256, 257, 512, 8193, 2147483647}}) {{
    for (int cap : {{1, 2, 32, 170, 1024}}) {{
      int tiles = direct_tile_count(length), count = direct_split_count(length, cap);
      assert(tiles == (int64_t(length) + 255) / 256);
      assert(count >= 0 && count <= cap && count <= tiles);
      int previous = 0;
      for (int i = 0; i < count; ++i) {{
        assert(aligned_boundary(i, tiles, count) == previous);
        int end = aligned_boundary(i + 1, tiles, count);
        assert(end > previous && end <= tiles);
        previous = end;
      }}
      assert(previous == tiles);
    }}
  }}
  for (int n : {{1, 2, 33, 512, 8388607}}) {{
    for (int count : {{1, 2, 32, 170, 340, 1024}}) {{
      if (count > n) continue;
      int previous = 0;
      for (int i = 0; i <= count; ++i) {{
        int boundary = aligned_boundary(i, n, count);
        assert(direct_boundary(i, n, count) == boundary);
        assert(boundary == int(int64_t(i) * n / count));
        assert(boundary >= previous && boundary <= n);
        if (i) assert(boundary > previous);
        previous = boundary;
      }}
      assert(previous == n);
    }}
  }}
}}
''')
    binary = tmp_path / "geometry.exe"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)


def _require_sm120():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise unittest.SkipTest("Requires SM120")


def _case(lengths, *, interleaved=False, capacity_length=None):
    device = torch.device("cuda", torch.cuda.current_device())
    # Reserve the same per-request capacity for changed row lengths/page mappings.
    capacity_length = max(lengths) if capacity_length is None else capacity_length
    assert capacity_length >= max(lengths)
    width = max(1, (capacity_length + 31) // 32)
    batch, pages = len(lengths), width * len(lengths)
    if interleaved:
        backing = torch.empty((pages, 2 * 32 * 144), dtype=torch.uint8, device=device)
        data = [
            backing.as_strided((pages, 32, 1, 128), (9216, 128, 128, 1), side * 4608)
            for side in range(2)
        ]
        sf_bytes = backing.view(torch.float8_e4m3fn)
        scales = [
            sf_bytes.as_strided(
                (pages, 32, 1, 16), (9216, 16, 16, 1), side * 4608 + 4096
            )
            for side in range(2)
        ]
    else:
        data = [
            torch.empty((pages, 32, 1, 128), dtype=torch.uint8, device=device)
            for _ in range(2)
        ]
        scales = [
            torch.empty((pages, 32, 1, 16), dtype=torch.float8_e4m3fn, device=device)
            for _ in range(2)
        ]
    for value in data:
        value.copy_(
            torch.randint(0, 256, value.shape, dtype=torch.uint8, device=device)
        )
    for value in scales:
        value.copy_(
            (torch.rand(value.shape, device=device) * 0.5 + 0.25).to(value.dtype)
        )
    table = torch.arange(pages, dtype=torch.int32, device=device).view(batch, width)
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    queries = [
        torch.randn((batch, 6, 256), device=device, dtype=torch.bfloat16) * 0.25
        for _ in range(2)
    ]
    return queries, data, scales, table, lens


def _reference(query, data, scales, table, lengths, bmm1, bmm2):
    # All FP4 codes, nontrivial E4M3 scales and the native BF16 operand rounding.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=query.device,
    )
    output = torch.zeros_like(query)
    for row, length in enumerate(lengths):
        if not length:
            continue
        ids = table[row, : (length + 31) // 32].long()
        operands = []
        for packed, sf in zip(data, scales, strict=True):
            values = packed[ids].view(-1, 128)[:length]
            codes = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2).long()
            block_scales = (
                sf.float()[ids].view(-1, 16)[:length].repeat_interleave(16, -1)
            )
            operands.append((lut[codes] * block_scales).to(torch.bfloat16).float())
        scores = (query[row].float() @ operands[0].T) * bmm1
        output[row] = (scores.softmax(-1) @ operands[1] * bmm2).to(output.dtype)
    return output


def _assert_output(actual, expected):
    assert torch.isfinite(actual).all()
    relative_rms = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-8)
    assert relative_rms.item() < 0.03


def _assert_plan(plan, lengths):
    if plan.direct:
        assert plan.ctas % len(lengths) == 0
        assert all(
            value.numel() == 0
            for value in (plan.tasks, plan.requests, plan.spans, plan.header)
        )
        return
    counts = [(n + 255) // 256 for n in lengths]
    limit = max(1, (sum(counts) + plan.ctas - 1) // plan.ctas)
    if plan.mode == "aligned":
        lo, hi = 1, max(1, max(counts))
        while lo < hi:
            mid = (lo + hi) // 2
            if sum((n + mid - 1) // mid for n in counts) <= plan.ctas:
                hi = mid
            else:
                lo = mid + 1
        limit = lo
    tasks, requests, prefix = [], [], 0
    spans = [[0, 0] for _ in range(plan.ctas)]
    for row, count in enumerate(counts):
        first = len(tasks)
        if plan.mode == "aligned":
            parts = (count + limit - 1) // limit
            tasks.extend(
                [row, i * count // parts, (i + 1) * count // parts, 1]
                for i in range(parts)
            )
        elif count:
            for bucket in range(prefix // limit, (prefix + count - 1) // limit + 1):
                if spans[bucket][0] == spans[bucket][1]:
                    spans[bucket][0] = len(tasks)
                tasks.append(
                    [
                        row,
                        max(prefix, bucket * limit) - prefix,
                        min(prefix + count, (bucket + 1) * limit) - prefix,
                        1,
                    ]
                )
                spans[bucket][1] = len(tasks)
        requests.append([first, len(tasks) - first])
        prefix += count
    used = (prefix + limit - 1) // limit
    if plan.mode == "aligned":
        spans[: len(tasks)] = [[i, i + 1] for i in range(len(tasks))]
        used = len(tasks)
    assert plan.header.cpu().tolist() == [
        limit,
        len(tasks),
        used,
        max(n for _, n in requests),
    ]
    assert plan.tasks[: len(tasks)].cpu().tolist() == tasks
    assert plan.requests.cpu().tolist() == requests
    assert plan.spans.cpu().tolist() == spans


def _check_changed_graph(
    mode, interleaved, *, direct=False, capacity_length=None, direct_batch=1
):
    _require_sm120()
    from flashinfer.xqa import create_xqa_decode_bucket_plan, xqa_decode_with_buckets
    from flashinfer.experimental.xqa_decode_buckets import backend

    initial = [0, 1, 31, 32, 33, 255, 256, 257, 511, 512, 513, 0, 8193]
    changed = [513, 0, 257, 1, 0, 512, 31, 256, 33, 255, 32, 8193, 0]
    replay_lengths = (initial, changed, [0] * len(initial), initial)
    if direct and direct_batch == 1:
        initial = [512]
        replay_lengths = [
            [n]
            for n in (0, 1, 255, 256, 257, 512, 8193, 65537, capacity_length, 0, 512)
            if n <= capacity_length
        ]
    elif direct:
        if direct_batch == 4:
            initial = [512, 257, 1, 0]
            replay_lengths = (
                initial,
                [0, 512, 256, 1],
                [255, 0, 257, 512],
                [512, 1, 0, 256],
                [0] * 4,
                initial,
            )
        elif direct_batch == 3:
            initial = [0, 257, 513]
            replay_lengths = (initial, [513, 0, 1], [1, 256, 0], [0] * 3, initial)
        else:
            assert direct_batch == 32
            initial = [0, 1, 255, 256] * 8
            replay_lengths = (initial, [256, 0, 1, 255] * 8, [0] * 32, initial)
    queries, data, scales, table, lens = _case(
        initial, interleaved=interleaved, capacity_length=capacity_length
    )
    plan = create_xqa_decode_bucket_plan(
        lens, page_table_width=table.shape[1], mode=mode
    )
    assert plan.direct == direct
    if direct and capacity_length == 512:
        assert plan.ctas == 2 * len(initial)
    print(dict(mode=mode, resources=plan.resources, ctas=plan.ctas), flush=True)
    workspaces = [plan.create_workspace() for _ in queries]
    assert len({workspace.scratch.data_ptr() for workspace in workspaces}) == len(
        workspaces
    )
    assert len({workspace.semaphores.data_ptr() for workspace in workspaces}) == len(
        workspaces
    )
    backings = [
        torch.full(
            (len(initial) + 1, 6, 256), 19, dtype=torch.bfloat16, device=lens.device
        )
        for _ in queries
    ]
    outputs = [value[:-1] for value in backings]
    bmm1, bmm2 = 0.7 / 16, 1.3

    def compute_layers():
        for query, workspace, out in zip(queries, workspaces, outputs, strict=True):
            xqa_decode_with_buckets(
                query,
                *data,
                *scales,
                table,
                plan,
                workspace,
                bmm1_scale=bmm1,
                bmm2_scale=bmm2,
                out=out,
            )

    # Compute must consume the plan, even when assignment is unavailable.
    with patch.object(
        backend, "_prepare", side_effect=AssertionError("implicit assignment")
    ):
        compute_layers()
    graph = torch.cuda.CUDAGraph()
    no_assignment = (
        patch.object(
            backend, "_prepare", side_effect=AssertionError("direct assignment")
        )
        if direct
        else nullcontext()
    )
    with no_assignment, torch.cuda.graph(graph):
        plan.prepare()
        compute_layers()
    pointers = [
        value.data_ptr()
        for value in (plan.tasks, plan.requests, plan.spans, plan.header)
    ]
    for lengths in replay_lengths:
        lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=lens.device))
        table.copy_(table.roll(1, dims=1))
        if direct and len(initial) > 1:
            table.copy_(table.roll(1, dims=0))
        for query in queries:
            query.normal_(0, 0.25)
        inputs = [*queries, *data, *scales, table, lens]
        snapshots = [value.clone() for value in inputs]
        graph.replay()
        torch.cuda.synchronize()
        _assert_plan(plan, lengths)
        assert pointers == [
            value.data_ptr()
            for value in (plan.tasks, plan.requests, plan.spans, plan.header)
        ]
        for query, out, backing, workspace in zip(
            queries, outputs, backings, workspaces, strict=True
        ):
            _assert_output(
                out, _reference(query, data, scales, table, lengths, bmm1, bmm2)
            )
            assert (backing[-1] == 19).all()
            assert (workspace.semaphores == 0).all()
            for row, length in enumerate(lengths):
                if not length:
                    assert (out[row] == 0).all()
        assert all(
            torch.equal(value, saved)
            for value, saved in zip(inputs, snapshots, strict=True)
        )


def _check_module_case(lengths, mode):
    _require_sm120()
    from flashinfer.xqa import create_xqa_decode_bucket_plan, xqa_decode_with_buckets

    queries, data, scales, table, lens = _case(lengths)
    plan = create_xqa_decode_bucket_plan(
        lens, page_table_width=table.shape[1], mode=mode
    )
    workspace = plan.create_workspace()
    actual = xqa_decode_with_buckets(
        queries[0],
        *data,
        *scales,
        table,
        plan,
        workspace,
        bmm1_scale=0.7 / 16,
        bmm2_scale=1.3,
    )
    _assert_plan(plan, lengths)
    _assert_output(
        actual, _reference(queries[0], data, scales, table, lengths, 0.7 / 16, 1.3)
    )
    assert (workspace.semaphores == 0).all()


class GeometryTests(unittest.TestCase):
    def test_static_capacity_and_validation(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "flashinfer/experimental/xqa_decode_buckets/geometry.py"
        )
        spec = importlib.util.spec_from_file_location("xqa_bucket_geometry", path)
        geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geometry)
        for batch, width, sms, expected in (
            (1, 1, 132, 1),
            (1, 16, 132, 2),
            (4, 16, 132, 8),
            (32, 4096, 132, 132),
            (32, 1, 4, 32),
        ):
            self.assertEqual(geometry.plan_geometry(batch, width, sms)[0], expected)
        for batch, width, sms, expected in (
            (1, 4096, 132, (132, True)),
            (4, 16, 8, (8, True)),
            (4, 16, 7, (7, False)),
            (32, 1, 4, (32, False)),
            (4, 264, 132, (132, True)),
            (4, 265, 132, (132, False)),
            (3, 17, 9, (9, True)),
            (3, 17, 8, (8, False)),
        ):
            self.assertEqual(geometry.plan_geometry(batch, width, sms), expected)
        for values in (
            (True, 16, 20),
            (0, 16, 20),
            (33, 16, 20),
            (1, True, 20),
            (1, 0, 20),
            (1, -1, 20),
            (1, geometry.MAX_TABLE_WIDTH + 1, 20),
            (1, 16, True),
            (1, 16, 0),
            (1, 16, 1.5),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                geometry.plan_geometry(*values)

    def test_aligned_boundaries_use_wide_intermediates(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_aligned_boundaries(Path(directory))


class BucketGraphTests(unittest.TestCase):
    def test_direct_replay(self):
        _require_sm120()
        for mode in ("persistent", "aligned"):
            for interleaved in (False, True):
                for capacity in (512, 131072):
                    with self.subTest(
                        mode=mode, interleaved=interleaved, capacity=capacity
                    ):
                        _check_changed_graph(
                            mode, interleaved, direct=True, capacity_length=capacity
                        )
                for batch, capacity in ((4, 512), (3, 513), (32, 256)):
                    with self.subTest(
                        mode=mode, interleaved=interleaved, direct_batch=batch
                    ):
                        _check_changed_graph(
                            mode,
                            interleaved,
                            direct=True,
                            capacity_length=capacity,
                            direct_batch=batch,
                        )

    def test_persistent_interleaved(self):
        _check_changed_graph("persistent", True)

    def test_persistent_separate(self):
        _check_changed_graph("persistent", False)

    def test_aligned_interleaved(self):
        _check_changed_graph("aligned", True)

    def test_aligned_separate(self):
        _check_changed_graph("aligned", False)


class BucketModuleTests(unittest.TestCase):
    def test_small_uniform_and_skew(self):
        _require_sm120()
        for mode in ("persistent", "aligned"):
            for lengths in ([512], [8064] * 32, [131072] + [4096] * 31):
                with self.subTest(mode=mode, lengths=lengths):
                    _check_module_case(lengths, mode)


if __name__ == "__main__":
    unittest.main()
