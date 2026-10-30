# SM120 NVFP4 decode bucket module plan

Current stage: expand module geometry before framework integration. Native XQA
remains the default; the experimental API is opt-in. Git tracks implementation
history on `feat/sm120-nvfp4-decode-bucketing`.

## Completed

- Source-level native task hooks, package JIT/FFI, planner/workspace API and
  changed-input CUDA Graph replay tests.
- Remote source import and real GPU validation. The initial transfer failures
  were resolved; they no longer block execution.
- Compact GPU assignment, persistent/aligned compute, direct geometry for small
  static capacities, and an optional Host planner with one descriptor upload.
- Page sizes 16/32/64: twelve test groups passed, zero compiler spills. Three
  matched mixed-length module measurements are in the experimental README.
- Multiple KV heads and ratios4/6/8: fifteen test groups passed, including128
  new replay cases. Three paired mixed-length anchors are recorded.
- Uniform query lengths1–5: causal reference and future-KV perturbation checks
  passed, followed by twelve focused groups and two paired timing anchors.
  The compiled configurations retain zero spills.
- Stage-final representative matrix: all22 single-attention cases completed with
  output/counter checks. Seventeen full-window medians are faster and five
  regress; speedup ranges0.56x–9.59x. All results are in the experimental README.

## First-stage conclusion

The authorized geometry expansion and representative module measurements are
complete. The [protocol](benchmark_protocol.md) defines the matrix. This is not
universal non-regression: balanced T3/R6 work regresses in compute itself, and
assignment erases the small q1 uniform4096 compute gain. Keep the native default.
No per-case threshold tuning or extra GPU reruns were performed.

For each slice: finish the module, review the diff, run bounded correctness and
timing, record compiler resources, then commit. Run the broader representative
matrix at stage completion, not after every patch. No service or TP4 run belongs
to this stage.

## Subsequent contribution work

### Completed bounded slice: native M32 query tile

Added and evaluated an explicit `query_tile=32` alternative to the retained M16
path; the experimental default and native XQA baseline remain unchanged.
It reuses native `mha.cu` SPEC_DEC arithmetic, causal mask, barriers and last-CTA
merge. Row-group mapping, workspace sizing and JIT keys are parameterized together:
`groups = H * ceil(T*R/query_tile)`. Request interval descriptors remain shared.
No new attention implementation or shape-fitted benchmark threshold is needed.

Resource model from native SharedMem: D256 BF16 persistent Q grows from8KiB to
16KiB; four X tiles grow from8KiB to16KiB. K/V buffers, scales and barriers stay
unchanged; row-stat storage is padded to32 rows in either case. Expected shared
memory is86656B versus70272B, still one CTA/SM. Per-thread accumulator storage
grows from32 to64 FP32 elements for a32x64 tile, so register pressure is the
main risk. Measure actual ptxas registers/spills and occupancy before claiming
benefit; do not force a register cap or redistribute warp registers speculatively.

Tests cover M32 causal suffixes, partial row groups, changed-input Graph replay,
direct/prepared ownership and Host/GPU planner compatibility. Focused timing is
native/M16/M32 at T3/R6/H4/P32: uniform512, uniform4096 and one64k+7x4k, plus
one T5/R8 partial-group anchor. Keep compute-only and planning-inclusive windows
separate. Review this slice before broader curves or real compile/opcheck work.

Measured: the explicit M32 path passed58 focused GPU cases plus7 CPU groups;
the first future-mask fixture needed a capacity correction because M32 correctly
selected the direct path. Passed cases were retained and only that failed method
was rerun. Nine M32 compiled variants have zero spills,219–235 registers/thread,
86656B shared memory and one resident CTA/SM.

The four matched full-window M16 -> M32 medians are43.382 ->31.914us,
239.259 ->232.259us,213.059 ->113.219us and627.370 ->605.590us. M32 improves
each measured M16 point, but both balanced T3 cases still regress against native.
Short balanced compute is24.950us versus24.336us native; uniform4096 compute
remains224.323us versus160.989us native. Thus query tile choice explains part,
not all, of the regression. Keep explicit selection/defaults unchanged.

Next performance investigation: inspect native split/grid selection against
bucket CTA ownership and merge work on uniform4096 before changing scheduling.
Do not assume that removing Assign resolves a compute-only regression. The
compile/interface qualification below is a separate prerequisite for framework
integration. Do not expand the service matrix to diagnose this module issue.

- Resolve the remaining balanced multi-query dispatch tradeoff using resource and
  dataflow analysis. Both query tile options are now measured; M32 improves the
  tested points but does not close uniform4096. Do not attribute all regressions to assignment or publish
  a blanket faster-backend claim.
- Integrate at the framework metadata boundary, reusing one plan across compatible
  layers and preserving changed-input Graph replay.
- Add the relevant FlashInfer/vLLM native tests, real compile/opcheck checks and benchmark
  integration. Module measurements alone do not establish serving gains.
  Source audit: this checkout's `flashinfer/utils.py` registration decorators
  currently return the original function; decorator presence in the bucket
  backend is not evidence of dispatcher registration or compile support. Verify
  the actual compiled public call and its mutation contract in a separate slice;
  do not change shared FlashInfer decorators as a local workaround.
- Prepare maintainer-facing scope, limitations and evidence before an upstream
  PR. No upstream acceptance or submitted PR is claimed.
