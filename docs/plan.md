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

## First-stage remaining work

1. Complete representative Tencent length-pattern performance coverage. Retain
   NVFP4 cache geometry, explicitly label batch-size adaptations, and report
   native, compute-only and planning-inclusive windows. No universal speedup is
   assumed and a one-percent short-case difference is not a tuning objective.
2. Publish the complete supported scope, measurement limits and next framework
   integration tasks. The matrix is defined in [the protocol](benchmark_protocol.md).

For each slice: finish the module, review the diff, run bounded correctness and
timing, record compiler resources, then commit. Run the broader representative
matrix at stage completion, not after every patch. No service or TP4 run belongs
to this stage.

## Subsequent contribution work

- Integrate at the framework metadata boundary, reusing one plan across compatible
  layers and preserving changed-input Graph replay.
- Add the relevant FlashInfer/vLLM native tests, build checks and benchmark
  integration. Module measurements alone do not establish serving gains.
- Prepare maintainer-facing scope, limitations and evidence before an upstream
  PR. No upstream acceptance or submitted PR is claimed.
