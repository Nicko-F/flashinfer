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

Page16/32/64 support is validated. The active work expands local KV head count,
Q/KV ratios4/6/8 and uniform query lengths1–5 while retaining head dimension256,
BF16 Q/output, packed NVFP4 cache and E4M3 block scales, with batch1–32.
The experimental README is authoritative for which API features have actually
landed; this section describes the complete authorized stage.

Let G be the number of independent head/query-row groups. For q1, G equals the
KV head count. Multi-query uses G=KV_heads*ceil(query_length*ratio/16). A group
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

These are module results. vLLM runtime integration, service gains and upstream PR
acceptance remain pending. Retain source facts, predictions and measurements as
separate claims. See [the execution plan](plan.md) for the next stage.
