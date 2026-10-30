# Experimental SM120 NVFP4 Decode buckets

This backend uses the native XQA loader, BF16 QK/PV arithmetic, FP32 softmax and
accumulation, and normalized-partial last-CTA merge. A separately prepared GPU
work plan balances KV tiles across requests. For B=1, the compute kernel derives
its task interval directly from the live device length, without a separate
assignment launch. Native XQA behavior is unchanged.
This port changes task ownership, not the native MMA operand orientation; it does
not implement a separate KQ-transposed/V-transposed-P-transposed compute core.

Both the API and backend are experimental. Calling
`flashinfer.xqa.create_xqa_decode_bucket_plan` or
`flashinfer.xqa.xqa_decode_with_buckets` explicitly opts in and warns once. There
is no automatic native-versus-bucket routing, autotuning entry, AOT registration,
or environment gate.
The caller retains its native fallback. Proposed owner: Nicko-F. A tracking issue
and maintainer-agreed graduation target are required before PR submission; neither
an issue nor a target release is claimed by this preparation patch.

## Supported inputs

- SM120; one query token per request, `1 <= B <= 32`.
- Contiguous BF16 Q/output `[B,6,256]`; one local KV head.
- Packed NVFP4 K/V `[pages,32,1,128]`, E4M3 block scales `[pages,32,1,16]`, NHD.
  Native page/token byte strides are retained; the packed head is contiguous.
- Contiguous CUDA int32 page table `[B,width]` and sequence lengths `[B]`.
- Independent finite effective BMM1 (QK) and BMM2 (V/output) scales.
- No sinks, sliding windows, soft caps, speculative rows, or FP8 output.

Lengths must be nonnegative and no larger than `width * 32`. Every physical page
ID reached by a live length must address the supplied cache. These device-value
conditions are caller preconditions; the hot path does not synchronize to inspect
them. Zero-length rows produce zero output.

There is no model, TP degree, layer count, run-directory or configured 128K
requirement. Width is bounded by `(INT32_MAX - 255) // 32`: this keeps padded tile
ends and all 32 dense table rows representable. CTA count is
`max(B, min(SMs, B * ceil(width * 32 / 256)))`, with at least B CTAs for zero-row
owners. This uses static table capacity, not a device-to-host length read. A wide
reserved table can retain a full grid even when its live prefix is short.
Compact task capacity
is `CTAs + B - 1`, the intersection bound for two interval partitions. Native
scratch is sized using 64-bit host arithmetic and rejected if its byte extent
would exceed native `MemSegmenter`'s uint32 offsets. No fixed GPU SM-count limit
is imposed.

## Plan, compute and ownership

```python
from flashinfer.xqa import create_xqa_decode_bucket_plan, xqa_decode_with_buckets

# Cold allocation/JIT; source/package/cache paths use normal FlashInfer discovery.
plan = create_xqa_decode_bucket_plan(seq_lens, page_table_width=page_table.shape[1])
workspace = plan.create_workspace()

# Each subsequent forward updates the same seq_lens storage, then plans once.
plan.prepare()
for q, k, v, ks, vs, out in compatible_attention_layers:
    xqa_decode_with_buckets(
        q, k, v, ks, vs, page_table, plan, workspace,
        bmm1_scale=qk_scale, bmm2_scale=v_scale, out=out,
    )
```

For B>1, the initial factory call prepares one generation. `prepare()` refreshes
task contents without replacing pointers, and compute consumes those tables.
For B=1, `plan.direct` is true, task/request/span/header tensors are deliberately
empty, and `prepare()` is a no-op. Each active CTA derives one contiguous interval
from the current device length; no task table is materialized or read. The direct
specialization uses the same native XQA arithmetic and partial-merge protocol,
not a call back to the native public API. Zero length produces zero output.
Compatible layers
share request order and live lengths, but retain independent Q/cache/output.
`mode="persistent"` uses global interval buckets with multiple tasks per CTA;
`mode="aligned"` minimizes balanced request-local fanout and launches one task
per CTA. Both modes select the same direct specialization for B=1. For B>1 the
modes remain explicit controls, not a validated load-dependent routing policy.

Plans, workspace and referenced tensors must outlive every asynchronous consumer
and captured Graph. Sequential layers may share scratch/counters after the prior
compute/merge completes. Concurrent calls need independent workspaces and outputs.
Refresh a shared plan only after its previous consumers complete. On another
stream, use explicit event ordering for both plan preparation and workspace reuse.
Allocate/JIT before capture and supply `out`; capture `plan.prepare()` once plus
all compatible computes, or prepare on the current stream before replaying a
compute-only Graph. The latter includes an external planning launch and must be
timed as such.

The persistent specialization retains the audited 256-thread producer/consumer
participants, named PV-tail barriers, async-copy drains, and full-CTA task join.
Historical resource figures (207 registers, 70,272B shared, zero spill instructions)
are a prediction baseline, not measurements of this port. Recompile and inspect
`plan.resources` (registers, shared bytes, local bytes, resident CTAs) and compiler
logs before making a performance claim. Local bytes alone do not imply spills.

## Reproduction

Run the focused tests on an SM120 system with normal FlashInfer dependencies:

```bash
pytest tests/experimental/test_xqa_decode_buckets.py -q
python examples/xqa_decode_buckets.py --lengths 131072,4096,4096,4096 --layers 16 --benchmark
```

The example compares native and bucket windows with the same caller-owned inputs,
and counts one assignment per layer window for B>1, zero for B=1. Use
`--capacity-length` to distinguish reserved table capacity from live lengths.
It is not a model or HTTP benchmark.
The tests cover actual task ownership, changed input/length/page-table Graph
replays, zero/restore, packed native byte strides, independent workspaces/scales,
counters and canaries. Existing FlashInfer custom-op/fake decorators are
currently no-ops; the declarations here do not establish torch.compile/fullgraph
compatibility. Framework integration must validate its actual capture boundary.

## Initial module validation (2026-10-30, before the small-work fix)

Source import, JIT and all six focused tests passed on an RTX 5090 (SM120),
Torch 2.14.0+cu130 and CUDA 13. The tests include four changed-input Graph cases
and both modes at small, uniform and skewed shapes. This validates the source
port, not a vLLM integration or model/service speedup.

The following are medians in microseconds for **two attention layers**. The
candidate includes one plan preparation per two-layer window. Native is the mean
of its before/after medians on the same inputs/device. The example uses seed 0,
CUDA Graph timing, five warmups and twenty samples per arm, without CUPTI. Inputs
are reused: these are hot-cache measurements, not rotated/cold-cache results.

| Lengths | Mode | Native window (us) | Candidate window (us) | Native / candidate |
| --- | --- | ---: | ---: | ---: |
| 128K + 3 x 4K | persistent | 180.085 | 102.672 | 1.754 |
| 512 | persistent | 23.525 | 34.986 | 0.672 |
| 512 | aligned | 23.517 | 25.973 | 0.905 |
| 32 x 8064 | persistent | 123.075 | 133.443 | 0.922 |
| 32 x 8064 | aligned | 123.210 | 132.317 | 0.931 |
| 128K + 31 x 4K | persistent | 1378.960 | 142.813 | 9.656 |
| 128K + 31 x 4K | aligned | 1380.283 | 154.896 | 8.911 |

There are clear small/uniform regressions; these modes are not a universal native
replacement or a validated automatic dispatch policy. For B=1/L=512, the initial
persistent minimum budget of two tiles gives one active CTA, versus two native
CTAs. Both bucket modes still launch an SM-sized grid.

Runtime resources were 201 registers/thread for persistent and 186 for aligned,
with 70,272 shared bytes, 112 local bytes and one resident CTA/SM for both. Aligned
ptxas reports a 112-byte stack and zero spill loads/stores. Persistent compiler
spill output was not retained, so its spill status is unconfirmed; local bytes
alone are not evidence of spilling.

### Small-work diagnosis and correction

`--diagnose` times planning, two compute calls, and the complete window separately
on the same allocations. Before the correction, persistent compute alone took
32.454 us; aligned compute took 23.517 us, close to its bracketed native result.
The actual persistent plan assigned the two 256-token tiles to one CTA. Native
and aligned used two nonempty CTAs. The independent timings are not assumed to
add exactly: Graph dependencies and overlap affect the combined window.

The correction allows a one-tile bucket when work is smaller than the CTA budget.
It preserves the native arithmetic, barriers, SM-sized launch grid and large-work
budget. A two-owner regression assertion and all six focused tests passed.

| B=1, length=512, two layers | After correction (us) |
| --- | ---: |
| Native before / after | 23.718 / 23.718 |
| Plan only | 1.947 |
| Persistent compute only | 23.722 |
| Plan + persistent compute | 26.589 |

The compute-only regression is removed in this case. The complete window remains
12.1% slower than native; planning/dependency overhead remains a separate problem.
This is not an automatic-fallback or universal-speedup claim. The natural rebuild
also reports zero spill loads/stores for persistent, with its 201 registers and
112-byte stack unchanged. The transposed Compute proposal is still not included.
The B32 skew anchor retains the same 142.813 us candidate median; the large-work
budget is unchanged. This anchor does not replace a later full integration study.

## Direct B1 and static capacity bound (2026-10-30)

All eight focused test groups passed, including direct replay in both modes,
separate/interleaved caches, tight/wide table capacities, zero/restore, changed
page IDs and lengths up to 128K. The direct specialization uses 175 registers,
70,272 bytes dynamic shared memory, a 112-byte stack, one barrier and one resident
CTA per SM. Compiler reports show zero spill loads/stores for all three variants.

The same GPU0/NUMA0, seed0, two-layer hot-Graph protocol gives the following
complete windows. Each native pair brackets its candidate; units are microseconds.
Candidate B1 windows contain no assignment launch; B>1 windows contain one.

| Live lengths | Table capacity per request | Native before / after | Candidate |
| --- | ---: | ---: | ---: |
| 256 | 256 | 21.261 / 21.261 | 21.878 |
| 512 | 512 | 23.718 / 23.589 | 24.386 |
| 2048 | 2048 | 25.312 / 25.274 | 25.971 |
| 512 | 131072 | 70.822 / 70.826 | 23.517 |
| 4 x 512 | 512 | 23.518 / 23.718 | 27.818 |
| 131072 + 31 x 4096 | 131072 | 1380.878 / 1380.880 | 143.018 |

Tight B1 still regresses about 2.7-3.1%. Wide-table B1 is about 3.01x faster even
though both grids contain 170 CTAs: the direct path uses only two participating
partials, whereas native's split count follows table capacity. Grid/partial
counts here are source-derived and are not presented as profiler measurements.
The static-cap B4 case has eight launched CTAs on both sides but still regresses
17.8%; assignment and the persistent compute path remain costs. The skew anchor
retains about 9.66x, without a new service or general-load claim.

These results motivate testing direct geometry when all potential tiles fit the
device CTA budget, plus removing redundant boundary division when each active
CTA owns exactly one tile. Those extensions are not included in this checkpoint.
