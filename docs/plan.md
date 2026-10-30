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

## First-stage geometry expansion (in progress)

1. Multiple KV heads and Q/KV ratios 4/6/8. Share one request plan; partition the
   CTA budget among independent head groups. Isolate scratch and completion
   counters, and test distinct head data, layouts and changed Graph inputs.
2. Uniform query lengths 1–5, including multi-query in this stage. Reuse native
   causal suffix attention with 16-row tiles. Separate logical sequence length
   from task processing boundaries so partitioning does not alter the mask.
3. Complete representative Tencent length-pattern performance coverage. Retain
   NVFP4 cache geometry, explicitly label batch-size adaptations, and report
   native, compute-only and planning-inclusive windows. No universal speedup is
   assumed and a one-percent short-case difference is not a tuning objective.

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
