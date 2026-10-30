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

Status: planned stage-final protocol; results must be added only after execution.
