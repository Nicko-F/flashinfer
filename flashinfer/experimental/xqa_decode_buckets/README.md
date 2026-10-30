# Experimental SM120 NVFP4 Decode buckets

This backend uses the native XQA loader, BF16 QK/PV arithmetic, FP32 softmax and
accumulation, and normalized-partial last-CTA merge. A separately prepared GPU
or caller-metadata host work plan balances KV tiles across requests. For B=1, or when every potential KV
tile fits the device CTA budget, the compute kernel derives its task interval
directly from live device lengths, without a separate assignment launch. Native
XQA behavior is unchanged.
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
- Contiguous BF16 Q/output `[B,H*R,256]`, with positive `H=num_kv_heads`
  (default 1) and `R=head_group_ratio` in `{4,6,8}` (default 6).
- Packed NVFP4 K/V `[pages,page_size,H,128]`, E4M3 block scales
  `[pages,page_size,H,16]`, NHD, with `page_size` in `{16,32,64}` (default 32).
  Native page/token byte strides are retained; the packed head is contiguous.
- Contiguous CUDA int32 page table `[B,width]` and sequence lengths `[B]`.
- Independent finite effective BMM1 (QK) and BMM2 (V/output) scales.
- No sinks, sliding windows, soft caps, speculative rows, or FP8 output.

Lengths must be nonnegative and no larger than `width * page_size`. Every physical page
ID reached by a live length must address the supplied cache. These device-value
conditions are caller preconditions; the hot path does not synchronize to inspect
them. Zero-length rows produce zero output.

There is no model, TP degree, layer count, run-directory or configured 128K
requirement. Width is bounded by `(INT32_MAX - 255) // page_size`: this keeps padded tile
ends and all 32 dense table rows representable. CTA count is
`C = max(B, min(ceil(SMs/H), B * ceil(width * page_size / 256)))` per head group,
with at least B CTAs for zero-row
owners. This uses static table capacity, not a device-to-host length read. A wide
reserved table can retain a full grid even when its live prefix is short.
Compact descriptor capacity is `C + B - 1`, the intersection bound for two
interval partitions, shared across heads. Native scratch capacity is at most
`H * (C + B - 1)` slots. Total slots/counters must fit int32, and native query
head/cache offsets must fit uint32. The factory checks the actual device grid-Y
limit once, outside capture, and retains it in the plan. Native
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

The factory selects `plan.direct` when `B == 1` or
`B * ceil(width * page_size / 256) <= ceil(SMs/H)`. This is a static capacity condition, not a
measured-length heuristic. Otherwise the initial factory call prepares one
generation and `prepare()` refreshes task contents without replacing pointers.

For direct plans, task/request/span/header tensors are deliberately empty and
`prepare()` is a no-op. The grid is `(C / B, H, B)`; its Z coordinate identifies
the request, with no task-table lookup. For B>1 each active CTA handles one tile,
and request-local partial ranges reserve slots from the static per-request cap.
For B=1, live splits can each cover multiple tiles for a long prefix; the merge
uses only its live partial count. One-tile intervals bypass general integer
boundary division. The direct specialization uses the same native XQA arithmetic
and partial-merge protocol, not a call back to the native public API. The first
CTA of an empty row clears its output; all excess owners return.
Compatible layers
share request order and live lengths, but retain independent Q/cache/output.
`mode="persistent"` uses global interval buckets with multiple tasks per CTA;
`mode="aligned"` minimizes balanced request-local fanout and launches one task
per CTA. Both modes select the same specialization when `plan.direct` is true.
For other shapes the modes remain explicit controls, not a validated
load-dependent routing policy.

The optional factory argument `page_size=32` is retained in `plan.page_size` and
selects a page-specific JIT/cache key and native `TOKENS_PER_PAGE` specialization.
Page sizes 16, 32 and 64 divide the unchanged 256-token CTA tile. The descriptor
ABI, ownership in token tiles, D256/q1 contract and native arithmetic are
unchanged. Cache views and the CPU host planner use the selected page capacity;
both planners and both ownership modes support all three page sizes.

The factory also accepts `num_kv_heads` and `head_group_ratio`. KV-head count is
runtime metadata and does not multiply descriptor storage or assignment work;
ratio selects its native JIT specialization. The prepared grid is `(C,H,1)`.
Each head group reads the same request plan and uses its own cache head stride,
Q/output head group and request counter `request * H + head`. Scratch slots are
compact across groups: prepared stride is live `header[1]`, direct B1 stride is
its live split count, and direct multi-request stride is C. The group offset is
added to both each slot and its request's partial base. The native scratch
segmenter receives the total live slot count across heads; allocation uses the
checked maximum. H=1 preserves the original slot layout. Multiple groups can
require more CTAs than SMs because each needs the existing zero-row owners.
The same 16-row padded MMA tile and 256-thread warp layout support ratios4/6/8;
resource reports must still be measured for each compiled specialization.

Plans, workspace and referenced tensors must outlive every asynchronous consumer
and captured Graph. Sequential layers may share scratch/counters after the prior
compute/merge completes. Concurrent calls need independent workspaces and outputs.
Refresh a shared plan only after its previous consumers complete. On another
stream, use explicit event ordering for both plan preparation and workspace reuse.
Allocate/JIT before capture and supply `out`; capture `plan.prepare()` once plus
all compatible computes, or prepare on the current stream before replaying a
compute-only Graph. The latter includes an external planning launch and must be
timed as such.

### Explicit host planner

The default `planner="gpu"` retains device Assign and its capture support.
`planner="host"` instead requires a caller-owned contiguous CPU int32
`host_seq_lens[B]`, with exactly the same lengths and request ordering as the
device `seq_lens`. The library does not copy device lengths back or infer them
from the page table. Values are checked against table capacity by the compiled
CPU planner; matching CPU/device metadata is the caller's responsibility.

```python
plan = create_xqa_decode_bucket_plan(
    seq_lens, page_table_width=page_table.shape[1],
    planner="host", host_seq_lens=host_seq_lens,
)
# Capture only the compatible compute calls, with plan/workspace kept alive.
# After updating both length buffers for the next forward:
plan.prepare()  # CPU fill + one H2D, outside capture
compute_graph.replay()
```

The C++ fill uses the same descriptor ABI, integer ownership math and compute
kernel as GPU Assign. One contiguous device descriptor backing has fixed views
for tasks, request ranges, CTA spans and header. One equally sized pinned CPU
backing is filled without Python per-descriptor loops, then copied by a single
asynchronous H2D. Storage is `24 * (CTAs + B)` bytes: 4,848 bytes for B32/170 CTAs.
Before refilling the staging backing, an event waits only if its previous H2D
is still in flight. This bounds staging storage and prevents DMA-source reuse;
the existing same-stream/event ordering requirement still protects device
descriptors from their prior compute consumers. Simultaneous host calls on the
same plan are unsupported. Direct plans retain empty descriptors and no-op
preparation for either planner.

Host preparation must run before each replay, outside capture; attempting to
capture a prepared host plan raises an error. CPU execution does not recur in a
CUDA Graph. Compute-only Graph time therefore excludes host fill, H2D submission
and dependency costs and must be labeled separately from completed wall time.

The bounded comparison example uses the same operand bank and compute path for
both planners and reports CPU C++/FFI fill, descriptor-copy completed wall,
compute-only GPU Graph and full completed-wall windows. GPU planning is measured
both inside its captured graph and before a compute-only graph. Metadata is
static during timing, so length uploads are excluded equally. Changed lengths,
row order, page IDs, zero/restore and staging reuse are covered by focused tests.

```bash
python examples/xqa_decode_buckets.py --compare-planners --layers 2 --output-json timings.json
```

Its balanced `32 x 4096` pattern adapts Tencent's B64 length pattern to this
backend's explicit B32 support; other anchors are `65536 + 7 x 4096`,
`131072 + 31 x 4096`, and a direct `512` control. Only the length patterns are
reused; dtype, backend and compute arithmetic remain this port's NVFP4 XQA.
This is a module comparison, not a vLLM integration or model/service claim.

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
# Select native page geometry explicitly; omit to retain page32.
python examples/xqa_decode_buckets.py --page-size 16 --lengths 65536,4096,4096,4096 --benchmark
```

The example compares native and bucket windows with the same caller-owned inputs,
and counts one assignment per layer window for prepared plans, zero for direct
plans. Use
`--capacity-length` to distinguish reserved table capacity from live lengths.
It is not a model or HTTP benchmark.
The tests cover actual task ownership, changed input/length/page-table Graph
replays, zero/restore, packed native byte strides, independent workspaces/scales,
counters and canaries. Existing FlashInfer custom-op/fake decorators are
currently no-ops; the declarations here do not establish torch.compile/fullgraph
compatibility. Framework integration must validate its actual capture boundary.

## Initial module validation (2026-10-30, before the small-work fix)

Historical measurements in the following sections use H=1/R=6 and page size32
unless another configuration is explicitly stated.

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

These results motivated direct geometry when all potential tiles fit the device
CTA budget, plus removing redundant boundary division when each active CTA owns
exactly one tile. Those extensions were not included in that checkpoint.

## Capacity-fitting direct tasks (2026-10-30)

The next change uses the static direct eligibility and `(tile, request)` grid
described above. All eight test groups passed again, including added B3, B4 and
B32 direct Graph replay with empty/mixed rows, non-power-of-two tile capacity,
page updates, both cache layouts and both mode labels. The direct kernel reports
186 registers, 70,272 bytes shared memory, 112 stack bytes, one barrier and one
resident CTA per SM, with zero spills. Prepared kernels retain their prior
resources. No native fallback or precision change is involved.

Four affected cases were retimed using the same paired two-layer Graph protocol.
Each complete candidate window now has zero assignment launches.

| Live lengths | Table capacity per request | Native before / after (us) | Complete candidate (us) |
| --- | ---: | ---: | ---: |
| 256 | 256 | 21.261 / 21.261 | 21.469 |
| 512 | 512 | 23.691 / 23.566 | 23.822 |
| 512 | 131072 | 70.822 / 70.822 | 23.310 |
| 4 x 512 | 512 | 23.718 / 23.581 | 23.874 |

The three tight cases still regress 0.8-1.0% against the mean of their native
controls. B4's earlier 17.8% gap is reduced to 0.95%; this is near parity in these
measurements, not a universal non-regression guarantee. Wide-table B1 is about
3.04x faster. Short-work geometry and multi-request direct dispatch changed
together, so their individual contributions are not causally isolated here.

Prepared skewed/uniform shapes were correctness-tested but not retimed in this
last change; their earlier timings remain explicitly earlier evidence. Wide
multi-request tables, vLLM metadata/capture integration and model/service timing
still require validation. In particular the wide-table B1 result is sensitive to
the actual capacity exposed by the framework and cannot be assumed for every
serving configuration. No transposed Compute implementation is claimed.

## Host versus GPU planning (2026-10-30)

All eleven focused test groups passed in 29.664 seconds, including both planner
choices and modes, changed-input host replay, descriptor equality over meaningful
slots, CPU validation, capture rejection and repeated staging reuse. Native
compute resource reports remained unchanged: persistent 201 registers, aligned
and direct 186; all use 70,272 shared bytes, 112 stack bytes and zero compiler
spill loads/stores.

One bounded persistent-mode invocation compared four patterns on RTX 5090
physical GPU0 with NUMA0 CPU affinity, seed0 and two layers. The same inputs were
reused with static exact CPU/device lengths. Length uploads are excluded equally;
these are hot-cache module windows. GPU Graph timing uses FlashInfer's existing
benchmark helper, five warmups, twenty samples, CUPTI off and cold-L2 handling
off. Completed wall samples include stream completion. The GPU captured wall is
the mean of medians measured before and after the host arm. GPU external wall
prepares on the CPU call path before its compute-only Graph, matching that
integration boundary of the host arm. Units below are microseconds.

| Lengths | GPU captured complete wall | GPU external complete wall | Host fill + H2D + Graph complete wall | CPU C++/FFI fill | H2D complete wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 x 4096 | 76.040 | 82.297 | 93.134 | 2.660 | 8.782 |
| 65536 + 7 x 4096 | 83.835 | 90.809 | 102.408 | 1.678 | 8.113 |
| 131072 + 31 x 4096 | 150.187 | 156.988 | 166.353 | 2.553 | 8.056 |
| 512, direct | 30.341 | 30.446 | 30.648 | no-op | no copy |

Compute-only GPU Graph medians were 63.248/74.922/137.027 us for GPU-planned
descriptors and 63.248/74.717/136.874 us for host-planned descriptors at the three
prepared patterns. Those windows omit host planning and H2D and do not establish
a host-planner gain. GPU Assign plus compute Graph medians were
69.648/77.584/144.093 us. Both direct compute-only medians were 23.722 us.
Component timings were measured independently and are not assumed additive.

The host option is slower for all three prepared patterns in completed wall
time, including against external GPU preparation. GPU remains the default;
host planning is retained as an explicit caller-metadata alternative. No tuning,
native compute change, vLLM integration or model/service speedup is claimed.
An initial timing harness attempt failed by capturing an existing Graph replay;
it produced no result row and was preserved. The corrected invocation captures
the original compute closures with the benchmark helper. The clean timing run
was not repeated.

## Page geometry expansion (2026-10-30 UTC)

The page16/32/64 slice passed all twelve focused test groups in 81.526 seconds.
The original default-page32 groups remain. Twenty-six added replay subcases
exercise page16/64 with both planners and modes, separate/interleaved cache
storage, page and tile tails, changed lengths/page IDs, zero/restore and direct
controls. CPU tests also cover each page size's maximum width and host capacity
rejection. Q6/KV1/D256/q1, task ownership and native compute remain unchanged.

All nine compiled page/mode variants report zero spill loads/stores, 70,272
shared bytes and one resident CTA per SM. Page specialization changes registers
and stack even though the source compute body is unchanged:

| Page size | Persistent registers | Aligned registers | Direct registers | Stack bytes |
| --- | ---: | ---: | ---: | ---: |
| 16 | 208 | 196 | 194 | 128 |
| 32 | 201 | 186 | 186 | 112 |
| 64 | 213 | 194 | 193 | 64 |

Three bounded matched GPU Graph measurements used RTX 5090 physical GPU0/NUMA0
CPU affinity, seed0, two layers and lengths `65536 + 7 x 4096`. Each page size
uses its corresponding native control and cache layout. These are hot-cache
GPU Graph medians in microseconds, five warmups and twenty samples per arm,
CUPTI off and cold-L2 handling off. The persistent candidate includes one GPU
Assign per two-layer window; native before/after brackets it. JIT and allocation
are outside timing.

| Page size | Native before | Native after | Persistent candidate |
| --- | ---: | ---: | ---: |
| 16 | 180.443 | 180.496 | 93.558 |
| 32 | 173.325 | 173.275 | 77.378 |
| 64 | 158.989 | 158.872 | 73.280 |

Each timing was run once after qualification. These anchors establish page
geometry support for this bounded q1 contract. Host-planner timing at the new
page sizes, larger head configurations, multiple query tokens and model/service
integration remain separate work.

## KV-head and ratio expansion (2026-10-30 UTC)

The shared-request-plan head expansion passed all fifteen focused test groups
in 178.518 seconds. The 128 new replay subcases cover H=1/2/4/8 and ratios4/6/8
at page32 with both planners, modes and cache layouts, plus direct and page16/64
anchors. Head-distinct scales, padded head strides, per-head 3% output tolerance,
changed lengths/page IDs, zero/restore and scratch/counter canaries all passed.
The original default and page regression groups remain. Multi-query is pending.

Nineteen compiled specializations reported zero spill loads/stores. Every
observed runtime resource tuple retained 70,272 shared bytes and one resident
CTA per SM. Page32 reports below include a 112-byte stack; they are measurements
of this shared-group implementation, separate from the earlier page-slice table.

| Ratio | Persistent registers | Aligned registers | Direct registers |
| --- | ---: | ---: | ---: |
| 4 | 206 | 191 | 180 |
| 6 | 204 | 183 | 183 |
| 8 | 206 | 191 | 191 |

Three bounded page32 persistent timing anchors used RTX 5090 physical GPU0/NUMA0
CPU affinity, seed0, two layers and lengths `65536 + 7 x 4096`. Inputs were shared
with their same-geometry native controls. Each row is one run with five warmups
and twenty samples, CUDA Graph timing, CUPTI off and cold-L2 handling off. Units
are microseconds. The candidate includes one GPU Assign per two-layer window;
allocation and JIT are outside timing. Native controls bracket each candidate.

| KV heads | Ratio | Native before | Native after | Persistent candidate |
| --- | ---: | ---: | ---: | ---: |
| 4 | 6 | 727.422 | 730.126 | 181.571 |
| 2 | 4 | 369.678 | 371.062 | 106.358 |
| 8 | 8 | 1879.414 | 1885.202 | 532.701 |

These are bounded q1 GPU Graph anchors, not full configuration curves or model
and serving latency. Host planning remains available and correctness-tested;
new-head completed-wall Host timing and the stage-final matrix remain separate
work. No per-change gain attribution or universal non-regression is implied.
