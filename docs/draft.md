# SM120 NVFP4 decode bucket module design

The contribution adapts task ownership around the native XQA loader, arithmetic
and last-CTA merge. Native XQA stays the default. An explicit experimental API
owns planning and workspace; it contains no model identity, experiment paths,
runtime source rewriting or fixed layer count.

## Dataflow

One plan owns stable device lengths, compact request/task/bucket tables and four
header words. Refresh the plan once per forward and reuse it across compatible
sequential attention layers. Layer Q, KV and output pointers remain independent.
A workspace owns native BF16 partial outputs, FP32 max/sum and completion
counters. Concurrent calls require separate workspaces. Refresh and consumption
must be ordered on the same stream or with explicit events.

GPU assignment and explicit Host planning share the descriptor ABI. Host planning
uses caller-provided exact CPU lengths and uploads once outside Graph capture;
it does not read lengths back from the GPU. Direct geometry skips descriptors
when the static tile bound fits the allocated CTA budget, or for one request.
Live lengths can still change on Graph replay.

## Configuration expansion

Page16/32/64 support is validated. The implementation also supports local KV head count,
Q/KV ratios4/6/8 and uniform query lengths1–5 while retaining head dimension256,
BF16 Q/output, packed NVFP4 cache and E4M3 block scales, with batch1–32.
The experimental README records focused GPU validation, bounded timing anchors
and the completed22-case representative matrix. Five full-window regressions
remain, so support for a configuration does not imply a speedup.

Let G be the number of independent head/query-row groups. For q1, G equals the
KV head count. Multi-query uses G=KV_heads*ceil(query_length*ratio/query_tile).
The default query tile is16; an explicit32-row alternative for multi-query
reuses the native SPEC_DEC core. Tile selection is cold plan metadata and a
JIT specialization, not a runtime scan or change to native XQA dispatch. A group
receives ceil(SMs/G) CTAs, bounded below by the batch and above by static tile
capacity. All groups reuse the request task table but have disjoint scratch and
request/group completion counters. This avoids assigning a full SM-sized grid
independently to every head. Grid and allocation bounds are checked during plan
creation; live metadata stays on device.

Multi-query reuses the native SPEC_DEC causal suffix path. The KV length includes
all query suffix tokens; query row t sees keys before L-T+t+1. Full logical L must
remain distinct from a bucket's clipped processing endpoint. The cold plan builds
the fixed causal mask; no per-step mask kernel or model-level MTP integration is
part of this work.

## Evidence and limits

Real source-package import, GPU compilation, changed-input Graph replay and
module timing have run. Initial transport failures are historical, not a current
blocker. Published page-size validation is commit
`e2898e54ecac4aece0c5e94f85685a17a64c25ec`; the experimental README records its
paired measurements. The native core uses256threads and70,272shared bytes in
the tested q1 configurations, one resident CTA/SM and zero compiler spills.
Registers vary with page size and mode; remeasure new specializations.

The subsequent M32 ownership comparison keeps the core and both modes unchanged.
For B32/T3/R6 uniform4096, persistent ownership splits every request and requires
two-task owners; aligned ownership keeps complete requests. Actual descriptors
confirm this distinction. Matched compute medians are221.099us persistent,
161.245us aligned and158.789us native. Planning-inclusive aligned is168.824us,
still slower than native. Mixed T3 requests instead favor persistent111.582us
over aligned173.432us full windows. Thus routing needs workload evidence; neither
mode is a universally faster backend. The report retains all four comparisons.

The public API now has an actual local torch.library dispatcher boundary for
Inductor, with declared plan/workspace/output mutations and fake implementations.
Direct and prepared calls, Host-prepared consumers and two independent layers
sharing one plan passed real fullgraph execution and changed-input CUDA Graph
replay. Generated code retains one plan and two compute calls for that shared
window, without unconditional workspace cloning. This is compile compatibility
and metadata reuse, not fusion of the two attention kernels. The exact FP8
schema-checker limitation and compiler input-realignment behavior are documented
in the module report; complete opcheck success is not claimed.

One matched B32 x 4096 two-layer M32/aligned window measured 337.066 us with
per-layer planning, 330.102 us with shared planning and 331.997 us with compiled
shared planning. Native before/after 317.920/322.730 us brackets the comparison.
The observed plan-reuse reduction is 2.07%; the shared window still regresses
against native. A 1.50% native bracket drift is larger than the 0.57% eager versus
compiled difference, so no compiler speedup is claimed.

These are module results. vLLM runtime integration, service gains and upstream PR
acceptance remain pending. Retain source facts, predictions and measurements as
separate claims. See [the execution plan](plan.md) for the next stage.
