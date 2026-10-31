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

- SM120; uniform `1 <= T=q_len_per_req <= 5`, `1 <= B <= 32`.
- Contiguous BF16 Q/output `[B,H*R,256]` for T1 and `[B,T,H*R,256]` for T>1,
  with positive `H=num_kv_heads`
  (default 1) and `R=head_group_ratio` in `{4,6,8}` (default 6).
- Explicit `query_tile=16` (default) or 32 physical rows; M32 requires T>1.
- Packed NVFP4 K/V `[pages,page_size,H,128]`, E4M3 block scales
  `[pages,page_size,H,16]`, NHD, with `page_size` in `{16,32,64}` (default 32).
  Native page/token byte strides are retained; the packed head is contiguous.
- Contiguous CUDA int32 page table `[B,width]` and sequence lengths `[B]`.
- Independent finite effective BMM1 (QK) and BMM2 (V/output) scales.
- No sinks, sliding windows, soft caps, ragged queries, tree masks, or FP8 output.

Lengths include the full T-token suffix: each is zero for a padded request or
at least T, and no larger than `width * page_size`. Query row t sees keys before
`L - T + t + 1`. Every physical page
ID reached by a live length must address the supplied cache. These device-value
conditions are caller preconditions; the hot path does not synchronize to inspect
them. Zero-length rows produce zero output.

There is no model, TP degree, layer count, run-directory or configured 128K
requirement. Width is bounded by `(INT32_MAX - 255) // page_size`: this keeps padded tile
ends and all 32 dense table rows representable. CTA count is
`C = max(B, min(ceil(SMs/G), B * ceil(width * page_size / 256)))` per independent
row group, where `G = H * ceil(T*R/query_tile)`,
with at least B CTAs for zero-row
owners. This uses static table capacity, not a device-to-host length read. A wide
reserved table can retain a full grid even when its live prefix is short.
Compact descriptor capacity is `C + B - 1`, the intersection bound for two
interval partitions, shared across heads. Native scratch capacity is at most
`G * (C + B - 1)` slots. Total slots/counters must fit int32, and native query
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
`B * ceil(width * page_size / 256) <= ceil(SMs/G)`. This is a static capacity condition, not a
measured-length heuristic. Otherwise the initial factory call prepares one
generation and `prepare()` refreshes task contents without replacing pointers.

For direct plans, task/request/span/header tensors are deliberately empty and
`prepare()` is a no-op. The grid is `(C / B, G, B)`; its Z coordinate identifies
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
ABI, ownership in token tiles, D256 contract and native arithmetic are
unchanged. Cache views and the CPU host planner use the selected page capacity;
both planners and both ownership modes support all three page sizes.

The factory also accepts `num_kv_heads` and `head_group_ratio`. KV-head count is
runtime metadata and does not multiply descriptor storage or assignment work;
ratio selects its native JIT specialization. For T1 the prepared grid is `(C,H,1)`.
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

For uniform T>1, `q_len_per_req` selects the native `SPEC_DEC` path with
`M_TILESIZE=query_tile`, `IS_SPEC_DEC_TREE=0` and no `SPEC_Q_SEQ_LEN` swap
specialization. The explicit `query_tile` choice is retained in the plan and
JIT key; no automatic tile selection or native dispatcher change is involved.
Each KV head owns `ceil(T*R/query_tile)` row blocks; the prepared grid is `(C,G,1)`.
One request descriptor plan is reused across all G groups. Scratch group strides
retain the live compact layout above and counters use `request * G + group`.
Q/cache/output mapping follows the native head-token rules with explicit task
group and row offsets. Cache storage remains `[pages,P,H,...]` and is not copied
per query. The plan allocates its small packed uint16 `[B,T,2]` causal mask once
on the cold path; prepare and compute generate no masks and read no lengths back.

The task's end tile limits processing; the full logical L independently locates
the causal suffix for both the mask threshold and valid-column calculation.
Early partitions must not treat their endpoint as the start of a draft block.
Output-helper debug assertions also distinguish local tile row r from global
head-token row m, using actual T for bounds; output indexing and arithmetic are
unchanged. This matters for partial M16 groups at T3/R6 and T5/R4.

The default 16-row choice retains the original physical warp tile. M32 reuses
native arithmetic, barriers, causal masks and merge with more live rows. From
native SharedMem layout, persistent Q and X each grow by 8KiB, predicting
86,656 shared bytes versus 70,272 for M16. Row-stat padding and K/V buffers stay
unchanged. Warp accumulator storage doubles from 32 to 64 FP32 values per thread;
actual registers, spills and occupancy must be measured. M32 is an explicit
alternative, with no performance guarantee. Multiple row blocks can reread KV compared with native's default
tile selection, so support does not imply a universal multi-query speedup.
The public native control uses its own default dispatcher with flattened Q/output
views, `q_len_per_req=T` and the same causal mask. Ragged queries, arbitrary tree
masks and a model speculative-generation service are outside this module scope.

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
The original default and page regression groups remain. Multi-query was pending
at this checkpoint.

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

## Uniform multi-query expansion (2026-10-30 UTC)

The T=1-5 slice passed its one-case causal smoke in 11.564 seconds, then all
twelve focused qualification groups in 195.753 seconds, with no skips or
failures. New coverage includes 48 changed-input replay cases, 24 direct cases,
32 future-KV perturbation cases and three q1 compatibility anchors. Both
planners, modes and cache layouts are covered across representative head,
ratio and page configurations. An independent causal reference checks each
query's visible prefix. The L=257/T=5 cases exercise the tile boundary: changing
the final future V preserves preceding outputs and changes the final output.
Zero/restore, changed page IDs and lengths, per-group counters and scratch
canaries also passed. T3/R6 and T5/R4 exercise the second query-row block.

Twenty-one compiled variants, seventeen using SPEC, reported zero spill
loads/stores. All observed runtime tuples retained 70,272 shared bytes and one
resident CTA per SM. Registers range from 180 to 224 and stack allocations from
64 to 128 bytes. The persistent R6/page32 SPEC kernel uses 224 registers and a
112-byte stack; R8/page64 uses 221 registers and a 64-byte stack.

Two bounded timing anchors used RTX 5090 physical GPU0/NUMA0 CPU affinity,
seed0, two layers and lengths `65536 + 7 x 4096`. Each row was measured once
after qualification with five warmups and twenty samples using FlashInfer's
CUDA Graph benchmark helper, CUPTI off and cold-L2 handling off. Inputs and
exact length metadata were static; JIT and allocation are excluded. The
persistent candidate includes one GPU Assign per two-layer window. Native
before/after controls bracket it, use the same causal mask and flattened query
views, and retain native default dispatch. Units are microseconds.

| Query tokens | KV heads | Ratio | Page size | Native before | Native after | Persistent candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 4 | 6 | 32 | 961.347 | 965.152 | 421.136 |
| 5 | 8 | 8 | 64 | 2266.998 | 2265.053 | 1251.598 |

These are bounded module GPU Graph windows. Candidate SPEC uses 16 physical
rows; extra row groups can reread KV compared with native default tiling.
The results do not establish universal multi-query speedup, completed-wall host
planning performance or model/serving latency. The stage-final workload matrix
and framework integration remain separate work; no model MTP path is enabled.

## Stage-final single-attention matrix (2026-10-30 UTC)

The finite suite completed all 22 cases once, using the fixed multi-query
implementation published at `ec05ef44bb2683d3e6e28d0531f085db177150ad`.
Every case passed the existing 3% output tolerance before and after timing and
reset its counters. The largest measured relative RMS error was 0.004230.
No kernels or thresholds were tuned after measuring these cases.

The nine length distributions follow the Tencent HPC-Ops workload cited in
[`docs/benchmark_protocol.md`](../../../docs/benchmark_protocol.md); its three
B64 distributions are adapted to B32. Precision, dimension and configuration
follow this backend's contract. This is workload reuse, not a Tencent kernel
measurement. P/H/R/T below denotes page size, KV heads, Q/KV ratio and query
tokens per request. The protocol document spells out every pattern's lengths.

These are one-attention GPU Graph medians on RTX 5090 physical GPU0 with NUMA0
CPU affinity, seed0 and static exact device lengths. FlashInfer's existing
helper uses five warmups and twenty samples per arm, CUPTI off and cold-L2
handling off. This differs from Tencent's default fifty event-timed replay
samples. Allocation and JIT are excluded. Native retains its default dispatch
and receives the same operands, geometry and causal mask. Native controls
bracket compute-only and full candidate windows. Speedup divides the mean of
the native medians by the full candidate median; values below 1 are regressions.

| P/H/R/T | Pattern | Native before / after (us) | Compute only (us) | Assign + compute (us) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| P32/H1/R6/T1 | uniform512 | 12.048 / 12.048 | 12.253 | 12.253 | 0.98x |
| P32/H1/R6/T1 | uniform4096 | 32.323 / 32.323 | 31.914 | 38.467 | 0.84x |
| P32/H1/R6/T1 | mixed | 31.299 / 31.299 | 20.240 | 28.125 | 1.11x |
| P32/H1/R6/T1 | extreme | 48.912 / 49.014 | 21.878 | 26.179 | 1.87x |
| P32/H1/R6/T1 | one64k7 | 86.813 / 86.771 | 37.648 | 40.717 | 2.13x |
| P32/H1/R6/T1 | one64k15 | 163.035 / 163.008 | 38.058 | 42.478 | 3.84x |
| P32/H1/R6/T1 | one64k31 | 315.254 / 315.509 | 45.067 | 51.984 | 6.07x |
| P32/H1/R6/T1 | one128k31 | 618.355 / 618.563 | 57.306 | 64.474 | 9.59x |
| P32/H1/R6/T1 | two32k30 | 162.626 / 163.088 | 44.557 | 52.106 | 3.13x |
| P16/H2/R4/T2 | uniform512 | 19.011 / 19.011 | 15.120 | 15.120 | 1.26x |
| P16/H2/R4/T2 | uniform4096 | 81.680 / 81.885 | 78.506 | 80.451 | 1.02x |
| P16/H2/R4/T2 | extreme | 123.510 / 122.894 | 26.384 | 31.504 | 3.91x |
| P16/H2/R4/T2 | one64k7 | 236.150 / 236.611 | 63.690 | 63.614 | 3.72x |
| P32/H4/R6/T3 | uniform512 | 24.336 / 24.541 | 36.829 | 43.997 | 0.56x |
| P32/H4/R6/T3 | uniform4096 | 161.056 / 161.091 | 231.283 | 239.171 | 0.67x |
| P32/H4/R6/T3 | extreme | 278.030 / 278.338 | 38.813 | 43.178 | 6.44x |
| P32/H4/R6/T3 | one64k7 | 480.835 / 480.682 | 208.042 | 213.674 | 2.25x |
| P64/H8/R8/T5 | uniform512 | 89.770 / 89.870 | 86.186 | 92.925 | 0.97x |
| P64/H8/R8/T5 | uniform4096 | 610.627 / 611.699 | 589.840 | 605.043 | 1.01x |
| P64/H8/R8/T5 | extreme | 557.021 / 556.507 | 105.898 | 110.301 | 5.05x |
| P64/H8/R8/T5 | one64k7 | 1131.075 / 1131.280 | 623.274 | 625.117 | 1.81x |
| P32/H4/R4/T4 | one64k7 | 480.237 / 481.446 | 84.494 | 89.768 | 5.36x |

The uniform512 H1/T1 and H2/T2 cases use direct plans: compute-only and full
are the same measured operation, recorded once with no Assign. Every other
full window includes one GPU Assign. Independently timed components are not
assumed additive; small inversions between compute-only and full medians are
retained. Maximum native-before/after median drift was 0.84%.

Five full windows regress, including both balanced H4/R6/T3 cases. Gains span
0.56x-9.59x over this representative suite. Separate query-row groups can reread
KV; the fixed 16-row resource choice does not promise a gain on balanced work.
These measurements do not isolate each cause of a regression. All observed
candidate runtime resources retain 70,272 shared bytes and one CTA per SM;
unchanged kernels reuse the prior qualification's zero-spill compiler evidence.
Previous two-layer anchors above remain separate results. This suite establishes
neither Host completed-wall cost nor model, framework or serving speedup.

The balanced T3/R6 regressions already appear in compute: uniform512 takes
36.829 us versus the 24.438 us native mean, and uniform4096 takes 231.283 us
versus 161.074 us, before Assign is included. Eighteen logical query/head rows
use two candidate M16 row groups, while native generic M32 handles them in one.
The resulting extra KV reads and row-group overhead are likely contributors,
based on source mapping and timing; this is not a complete profiling attribution.
For q1 uniform4096, compute-only is 31.914 us versus native's 32.323 us, but the
38.467 us full window erases that small gain. Planning-window cost matters there.
All planned module cases are complete; these regressions rule out a universal
non-regression claim. Dispatch and tile tuning are not part of this stage.

Run the same finite suite with:

```bash
python examples/xqa_decode_buckets.py --suite tencent22 --layers 1 --output-json results.json
```

The JSON records raw samples, output errors, counters and resources after each
completed case using atomic replacement. Add `--resume` to skip completed cases
with matching suite configuration; use a new output file when configuration
changes. Failed case records are preserved in the failure list.

## Explicit M32 query tile (2026-10-30 UTC)

The M32 smoke passed in 11.871 seconds. Initial qualification ran eleven groups
in 102.454 seconds: ten passed, while one future-mask fixture asserted that its
plan must be prepared. M32 reduced that fixture to one row group, so its reserved
99 tiles correctly fit the 170-SM direct budget. The fixture's capacity now
derives from the current device SM count outside capture to guarantee prepared
ownership. Only that failed method was rerun; all sixteen future-mask cases
passed in 2.438 seconds. The failed run remains preserved. Kernel code did not
change after this fixture correction.

Combined qualification covers seven CPU groups and 58 GPU cases: 24 prepared
changed-input Graph replays, sixteen direct replays, sixteen causal future-KV
checks and two default-M16 compatibility anchors. Both planners, ownership modes,
cache layouts, zero/restore and partial row groups are covered. T3/R6 uses
eighteen rows; T5/R8 exercises a second M32 group with only eight active rows.

All twelve compiled variants, including nine M32 variants, report zero spill
loads/stores. Every observed M32 kernel uses 86,656 shared bytes, confirming the
resource prediction, and one resident CTA per SM. M32 registers range from 219
to 235; stack allocations are 64/112/128 bytes. The persistent R6/page32 M32
kernel uses 219 registers and a 112-byte stack, versus M16's 224 registers and
70,272 shared bytes. Persistent R8/page64 M32 uses 232 registers and a 64-byte
stack, versus M16's 221 registers. No register cap was added.

One bounded process measured the four planned single-attention cases once.
M16 and M32 share identical Q/KV/scales/page tables and native outputs, with
independent plans, candidate outputs and workspaces. Native before/after brackets
both candidates and retains its default dispatch. RTX 5090 physical GPU0/NUMA0
CPU affinity, seed0, static device lengths, five warmups and twenty hot CUDA Graph
samples per arm match the prior helper protocol; CUPTI and cold-L2 handling are
off. JIT/allocation are excluded. Every full candidate window includes one GPU
Assign, and output/counter checks pass before and after timing. Units are
microseconds; speedups use the full candidate windows and mean native controls.
Across both candidates in all four cases, maximum relative RMS error is
0.0041943 and maximum native-before/after median drift is 0.191%.

| P/H/R/T | Length pattern | Native before / after | M16 compute / full | M32 compute / full | M32 vs M16 | M32 vs native |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32/4/6/3 | 32 x 512 | 24.336 / 24.336 | 36.893 / 43.382 | 24.950 / 31.914 | 1.36x | 0.76x |
| 32/4/6/3 | 32 x 4096 | 160.835 / 161.142 | 231.286 / 239.259 | 224.323 / 232.259 | 1.03x | 0.69x |
| 32/4/6/3 | 65536 + 7 x 4096 | 482.032 / 482.574 | 208.144 / 213.059 | 107.997 / 113.219 | 1.88x | 4.26x |
| 64/8/8/5 | 65536 + 7 x 4096 | 1131.331 / 1133.482 | 625.834 / 627.370 | 601.870 / 605.590 | 1.04x | 1.87x |

M32 improves over M16 in these four measurements, but both balanced T3 cases
still regress against native in the full window. Short balanced compute is near
native before planning cost; long balanced compute remains slower. Removing one
row group therefore does not resolve every limitation, and these measurements
do not isolate the remaining cause. Independently timed components are not
assumed additive. M16 stays the default, with no automatic tile/shape policy or
universal non-regression claim. Broader curves, actual compile/opcheck support
and model/serving integration remain separate work.

Use `--query-tile 32 --q-len-per-req 3` for an explicit M32 example, or compare
the same four cases with `--suite query_tiles4 --layers 1 --output-json tiles.json`.
The comparison JSON retains both candidates' resource, error and timing records.

## Existing M32 ownership modes (2026-10-31 UTC)

One four-case diagnostic invocation compared existing persistent and aligned M32
modes with shared Q/KV/scales/page tables, the same constant mask tensor and
native outputs, plus independent candidate plans, outputs and workspaces.
Kernels, public API and defaults are unchanged; prior correctness and zero-spill
compiler qualification remain applicable. All four before/after output and
counter checks passed. Maximum relative RMS error across both modes was
0.0041943, and maximum native-before/after median drift was 0.302%.

GPU0-6 were busy at admission. This run used idle RTX 5090 physical GPU7 with
NUMA1 CPU affinity for every arm, with at least 128GiB host memory available.
Its current native controls define the comparisons below; absolute historical
GPU0 timings are separate evidence. The single-attention helper protocol is
unchanged: seed0, static device lengths, five warmups and twenty hot CUDA Graph
samples per arm, CUPTI off and cold-L2 handling off. JIT, allocation and descriptor
readback are excluded. Each full candidate window includes one GPU Assign.
Units are microseconds. Native speedup uses the mean native medians; mode
speedup uses the persistent full median, each divided by aligned's full median.
Independently measured components are not assumed additive.

| P/H/R/T | Length pattern | Native before / after | Persistent compute / full | Aligned compute / full | Aligned vs persistent full | Aligned vs native full |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32/4/6/3 | 32 x 512 | 24.131 / 24.131 | 24.746 / 31.094 | 24.541 / 30.376 | 1.02x | 0.79x |
| 32/4/6/3 | 32 x 4096 | 158.790 / 158.787 | 221.099 / 229.034 | 161.245 / 168.824 | 1.36x | 0.94x |
| 32/4/6/3 | 65536 + 7 x 4096 | 474.640 / 476.074 | 106.768 / 111.582 | 167.187 / 173.432 | 0.64x | 2.74x |
| 64/8/8/5 | 65536 + 7 x 4096 | 1118.224 / 1118.226 | 596.293 / 599.416 | 577.194 / 578.730 | 1.04x | 1.93x |

Actual device descriptors confirm the uniform4096 ownership prediction. Per
query/head group, persistent uses a twelve-tile budget, 64 tasks and 43 active
owners; 21 owners process two tasks, and every request has two partials. Across
four groups, all 172 launched CTAs are active. Aligned uses a sixteen-tile budget
and 32 whole-request tasks, with no inter-CTA request merge: 128 active CTAs in
the same 172-CTA capacity grid. The generic native SM120 source derives one split
and grid `(1,4,32)`, or 128 CTAs with no inter-CTA merge. Native counts are
source-derived; bucket counts come from meaningful descriptors copied outside
timing. Neither is presented as a profiler launch measurement.

Aligned removes most of the balanced4K compute gap in this comparison, while
its 168.824 us full window remains 6.32% slower than the 158.789 us native mean.
Ownership, task restarts,
barriers and generated code differ together, so this does not isolate merge cost.
Aligned is substantially slower on the T3 skew case despite both modes using a
nine-tile budget and 29 partials for the long request; partial counts alone do
not explain that result. Both modes remain explicit alternatives, with no
automatic length threshold, default switch or universal performance claim.

Runtime resources retain 86,656 shared bytes and one resident CTA per SM. R6/P32
persistent/aligned report 219/228 registers and a 112-byte stack; R8/P64 report
232/223 registers and a 64-byte stack. These unchanged specializations reuse the
prior zero-spill compiler evidence. No additional correctness matrix, broader
curve or service run was performed.

Use `--suite query_modes4 --layers 1 --output-json ownership.json` to reproduce
the finite comparison. Its JSON includes raw samples, output errors, resources,
meaningful descriptors and owner/partial histograms. The native launch report
uses the generic SM120 M32 mapping with its source basis explicitly recorded.
