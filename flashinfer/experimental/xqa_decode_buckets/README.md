# Experimental SM120 NVFP4 Decode buckets

This backend uses the native XQA loader, BF16 QK/PV arithmetic, FP32 softmax and
accumulation, and normalized-partial last-CTA merge. A separately prepared GPU
work plan balances KV tiles across requests. Native XQA behavior is unchanged.
This port changes task ownership, not the native MMA operand orientation; it does
not implement a separate KQ-transposed/V-transposed-P-transposed compute core.

Both the API and backend are experimental. Calling
`flashinfer.xqa.create_xqa_decode_bucket_plan` or
`flashinfer.xqa.xqa_decode_with_buckets` explicitly opts in and warns once. There
is no automatic routing, autotuning entry, AOT registration, or environment gate.
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
ends and all 32 dense table rows representable. CTA count is derived from the
device SM count, with at least B CTAs for zero-row owners. Compact task capacity
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

The initial factory call prepares one generation. `prepare()` refreshes task
contents without replacing pointers. Compute never replans. Compatible layers
share request order and live lengths, but retain independent Q/cache/output.
`mode="persistent"` uses global interval buckets with multiple tasks per CTA;
`mode="aligned"` minimizes balanced request-local fanout and launches one task
per CTA. The modes are explicit controls, not a validated routing policy.

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
and counts one assignment per layer window. It is not a model or HTTP benchmark.
The tests cover actual task ownership, changed input/length/page-table Graph
replays, zero/restore, packed native byte strides, independent workspaces/scales,
counters and canaries. Existing FlashInfer custom-op/fake decorators are
currently no-ops; the declarations here do not establish torch.compile/fullgraph
compatibility. Framework integration must validate its actual capture boundary.

## Module validation (2026-10-30)

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
replacement or a validated automatic dispatch policy. For B=1/L=512, the current
persistent minimum budget of two tiles gives one active CTA, versus two native
CTAs. Both bucket modes still launch an SM-sized grid. Separate planning/compute
diagnostics are being collected before changing this policy.

Runtime resources were 201 registers/thread for persistent and 186 for aligned,
with 70,272 shared bytes, 112 local bytes and one resident CTA/SM for both. Aligned
ptxas reports a 112-byte stack and zero spill loads/stores. Persistent compiler
spill output was not retained, so its spill status is unconfirmed; local bytes
alone are not evidence of spilling.

