# Multi-configuration module benchmark protocol

This stage measures one attention operator, not model forward or serving.
Use the existing FlashInfer `bench_gpu_time` helper and the runnable
`examples/xqa_decode_buckets.py`; no separate benchmark framework is required.

## Reference and adaptations

Length distributions come from
[Tencent HPC-Ops decode benchmark](https://github.com/Tencent/hpc-ops/blob/2a2e26562433a8ba4b504858f1c938eb7612c901/benchmark/attention_decode/bench_attention_decode_bf16.py).
Its default measures a single BF16 attention operator with Q8/KV1, dimension128,
page64 and q1. This contribution retains BF16 Q/output, NVFP4 KV, dimension256
and the declared configuration below. This is reuse of workload distributions,
not execution of Tencent's BF16 kernel on SM120.

The three 64-request patterns are reduced to32 requests, retaining their length
proportions. The other six original patterns are unchanged:

| Pattern | Lengths in this stage |
| --- | --- |
| uniform512 | 32 x 512 |
| uniform4096 | 32 x 4096 |
| mixed | 16 x 128 + 16 x 4096 |
| extreme | 15 x 64 + 16384 |
| one64k7 | 65536 + 7 x 4096 |
| one64k15 | 65536 + 15 x 4096 |
| one64k31 | 65536 + 31 x 4096 |
| one128k31 | 131072 + 31 x 4096 |
| two32k30 | 2 x 32768 + 30 x 4096 |

## Representative configuration matrix

| Page | KV heads | Q/KV ratio | Query tokens/request | Patterns |
| ---: | ---: | ---: | ---: | --- |
| 32 | 1 | 6 | 1 | All nine |
| 16 | 2 | 4 | 2 | uniform512, uniform4096, extreme, one64k7 |
| 32 | 4 | 6 | 3 | uniform512, uniform4096, extreme, one64k7 |
| 64 | 8 | 8 | 5 | uniform512, uniform4096, extreme, one64k7 |
| 32 | 4 | 4 | 4 | one64k7 |

These22 cases are representative performance coverage, not a Cartesian sweep of
all supported configurations. Correctness coverage is separate. Per-case lengths
include the causal query suffix. Both native and candidate receive the same mask,
cache, query and page-table geometry. Keep native's default dispatch/tile choice;
do not force it to match the candidate's16-row tile.

## Timing and reporting

- One physical GPU and identical CPU affinity for paired controls/candidate.
- One layer per window; preserve previous two-layer measurements separately.
- Record native-before, candidate-compute-only, candidate-plan-plus-compute and
  native-after medians and samples. Direct plans have no Assign launch.
- Reuse existing FlashInfer hot CUDA Graph timing: five warmups, twenty samples,
  CUPTI off, cold-L2 handling off. Allocation and JIT are outside measurement.
  Tencent's original helper uses fifty event-timed replay samples by default;
  do not present our helper or sampling count as identical to theirs.
- GPU planning is included in the complete candidate window. Host completed-wall
  results are a separate measurement and cannot be inferred from GPU times.
- Check outputs and reset counters outside timing. Record resource usage and
  retain failures/partial runs. Report regressions, without retuning thresholds
  on each measured case or claiming universal non-regression.
- Persist one result per completed case so interrupted work can resume without
  rerunning completed cases. The existing bounded experiment launcher controls
  timeout, heartbeat, memory floor and ownership.

Status: completed once in `flashinfer-buckets-suite-20261030T170455Z`, with all22
case output/counter checks passing and no retries or post-result kernel tuning.
Seventeen full-window medians are faster, five regress; the largest relative RMS
error is0.004230 and maximum native bracket drift is0.84%. Full results and
limitations are in the [module report](../flashinfer/experimental/xqa_decode_buckets/README.md#stage-final-single-attention-matrix-2026-10-30-utc).

## Focused query-tile follow-up

`--suite query_tiles4` compares native, M16 and explicit M32 over the same
operand bank, with independent candidate plans/workspaces. It selects T3/R6/H4/P32
uniform512, uniform4096 and one64k7, plus T5/R8/H8/P64 one64k7. Native brackets
both candidates; each candidate retains separate compute-only and complete
planning-inclusive windows. The same helper, precision and sampling settings
above apply. This diagnoses the M16 row-group split; it does not replace the
22-case historical matrix or constitute a new full performance qualification.

`--suite query_modes4` subsequently compares existing persistent/aligned M32
ownership on those same four workloads. Both arms share Q/KV/scales/page tables
and the constant causal mask, with independent outputs/plans/workspaces. Actual
device descriptors and owner/partial counts are copied outside timing. Generic
SM120 native grids are explicitly marked source-derived, not profiler-observed.
One native bracket surrounds both modes. Compare within the matched physical
GPU/NUMA run; historical timings on another GPU are context, not direct controls.

## Compile and shared-plan integration anchor

The bounded compile follow-up uses the public bucket API with default Inductor
and `fullgraph=True`, then CUDA Graph replay with changed inputs. A separate
Dynamo explanation checks operator counts; it does not replace actual Inductor
execution. Plan construction, JIT, workspace allocation and Host descriptor
uploads stay outside compilation/capture. The GPU planner may be captured once
before multiple compatible attention consumers.

The timing anchor is two independent attention layers sharing request metadata:
B32, uniform4096, P32, H4, R6, T3, M32, aligned mode. Compare native before/after,
eager per-layer preparation, eager once-per-window preparation and compiled
once-per-window preparation on identical operands and one GPU/CPU placement.
Each layer has its own output and workspace. Use the existing hot CUDA Graph
helper and record all samples; count actual plan/compute operations and inspect
Inductor output for additional copies or kernels. This measures a two-attention
module window, not a whole transformer block or vLLM serving latency. It does
not claim that `torch.compile` itself fuses the opaque attention kernels.
