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

- Resolve the balanced multi-query tile/dispatch tradeoff using resource and
  dataflow analysis. Eighteen rows currently need two M16 groups versus one
  native M32 group. Do not attribute all regressions to assignment or publish
  a blanket faster-backend claim.
- Integrate at the framework metadata boundary, reusing one plan across compatible
  layers and preserving changed-input Graph replay.
- Add the relevant FlashInfer/vLLM native tests, real compile/opcheck checks and benchmark
  integration. Module measurements alone do not establish serving gains.
- Prepare maintainer-facing scope, limitations and evidence before an upstream
  PR. No upstream acceptance or submitted PR is claimed.
