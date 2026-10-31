# SM120 NVFP4 decode bucket module plan

Current stage: compilation and shared-plan reuse qualified; framework integration
is next. Native XQA remains the default; the experimental API is opt-in. Git tracks implementation
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

### Completed slice: real compile contract and shared-plan layer window

Starting source audit: shared FlashInfer registration decorators are identity functions;
the previous bucket public run called data_ptr/storage pointers, hardware lookup
and JIT/FFI Python paths. Native test_batch_decode_kernels.py uses a module-level
torch.library boundary; experimental b12x also registers real custom ops with
mutated outputs and fake implementations. Reuse that PyTorch mechanism locally,
without editing global decorators, inventing a handle registry or hiding mutations.

Register actual bucket plan/compute ops at backend import. Keep the existing
eager direct-FFI implementation available; compiled calls cross an opaque op
boundary sharing the same implementation. Declare tasks/requests/spans/header
and scratch/semaphores/output mutations correctly. Move storage-pointer alignment
and output-alias checks inside the real implementation, not into fake tracing;
retain traceable metadata checks. Resolve default PDL/hardware state in cold plan
creation. Plan construction/JIT/workspace allocation stay outside compile/capture.
Host planning/upload stays outside compiled graphs; GPU prepare can be compiled
once before multiple consumers. Preserve causal math, layouts, kernels, explicit
modes, output-buffer semantics and all existing defaults.

Evidence: actual torch.library.opcheck where applicable, default Inductor
torch.compile(fullgraph=True) of the public call (no suppress_errors/eager-only
backend/private-call substitution), changed Q/cache/lengths including zero/restore,
and unchanged numerical tolerances. Cover a direct q1 and prepared M32 modes;
host prepare outside a compiled consumer gets a bounded anchor. A two-layer
call prepares once and consumes independent layer operands/workspaces; inspect
captured operators to verify one plan and two computes. Do not label this a
whole-model/vLLM integration or serving result.

Budget remains the qualified native core; binding changes must not allocate a
per-token workspace or synchronize/read GPU lengths to CPU. Mutation handling
may introduce compiler copies: inspect graph and targeted timings, and diagnose
any material overhead rather than dropping mutation declarations. Use the existing
test/example/runner with one selected idle GPU,128GiB memory floor,threads2,
bounded heartbeat/timeouts. Finish coherent interface changes and focused checks,
then one two-layer reuse timing anchor with current native/eager/compiled controls;
no full22-case/service rerun. Material redesign returns to lead; no blind retries.

Qualification now exercises actual default Inductor fullgraph execution on
PyTorch 2.14.0+cu130. Direct q1, both prepared M32 modes, Host-prepared compiled
consumption and shared-plan two-layer changed-input Graph replay passed. The
two-layer generated code contains one plan op and two compute ops; inspection
found no unconditional capacity-workspace copies or extra generated math kernels.
Read-only inputs have Inductor's conditional `copy_if_misaligned` guards.

Two test-tool expectations were corrected without changing the library math:
PyTorch schema checking calls unsupported FP8 `allclose`/`mul_cuda`, so the exact
observed limitation is reported, while fake checks and read-only input byte
immutability are still verified. Plan schema checking passes. Also, a compiled
consumer may realign an unaligned read-only input and succeed, whereas eager
must reject it. Do not require identical rejection behavior after compiler input
normalization. Preserve both original failed runs and their explanations.

The corrected storage-behavior test passed without library changes: compiled
aliasing was rejected, and misaligned query/output inputs were normalized and
passed numerical checks. One matched B32 x 4096, H4/R6/P32/T3/M32 aligned
two-layer timing invocation measured native before/after 317.920/322.730 us,
eager per-layer preparation 337.066 us, eager shared preparation 330.102 us and
compiled shared preparation 331.997 us. Sharing the plan reduces this observed
window by 2.07%; eager shared remains 3.05% slower than the native paired mean 320.325 us.
Native bracket drift is 1.50%, so the 0.57% compiled-versus-eager difference is not
a stable compiler performance claim. No extra curve or timing retry was run.

Lead conclusion: the compile and shared-plan interface prerequisite is complete
for these bounded cases. Kernel math/defaults remain unchanged. The next scoped
work is the existing vLLM metadata lifecycle and native fallback integration,
not another binding microbenchmark or per-shape threshold search. Small/uniform
regressions remain visible and must not become a universal faster-backend claim.

### Completed slice: balanced-load task ownership diagnosis

Starting point: T3/R6/H4/P32, B32 uniform4096 M32 compute224.323us,
full232.259us versus native160.989us. Native `launchMHAFlashInfer` chooses
floor(SMs/(B*H)) splits, bounded by cache tiles. On the measured170-SM GPU this
is one split,128 active CTAs and no inter-CTA merge. The persistent planner
has43 buckets/group,512 KV tiles/group and a12-tile budget:64 task fragments
per group,43 active owners, with21 owners processing two tasks. All32 requests
have two partials. These are source-derived counts, to verify against descriptors.

Existing aligned planning finds a16-tile budget:32 whole-request tasks/group,
128 active CTAs and no inter-CTA merge. Grid capacity remains172 with44 idle
CTAs, so it is not identical to native. Compare this already-implemented path
before creating a new scheduler. Persistent barriers/task restart and fragment
merge are coupled differences; do not attribute their total cost individually.

Reuse the native math, M32 specialization, existing planners, examples and timing
helper. No production kernel/API/default change in this diagnosis. Extend the
existing example to compare persistent/aligned M32 over one shared operand bank,
with independent workspaces; collect descriptors outside timing and retain
native-before/after, compute-only and plan-plus-compute windows. The same four
fixed query_tiles4 workloads provide balanced-short, balanced4K and skewed anchors.
Run once on one admitted GPU with the standard memory/thread limits. No full
service/22-case rerun. Existing correctness qualification remains applicable;
check outputs/counters for these diagnostic calls and record resources.

Budget is unchanged: M32 shared86656B,256 threads, one resident CTA/SM; only
work assignment differs. If aligned closes the gap, review a metadata-derived
dispatch/assignment policy rather than fitting length thresholds. If it does not,
inspect the measured grid/resources before a narrowly targeted profile. Return
to lead judgment after this bounded comparison; no unplanned tuning sweep.

Measured once on matched GPU7/NUMA1 (GPU0–6 occupied): all four output/counter
checks passed. Actual uniform4096 descriptors match the predicted counts above.
Native paired mean158.789us; persistent compute/full221.099/229.034us; aligned
compute/full161.245/168.824us. Aligned cuts full-window latency26.29% relative
to persistent, but remains6.32% slower than native. This isolates most of the
balanced compute regression to the ownership/specialization path as a coupled
change, not to M32 arithmetic or Assign alone. It does not separately measure
restart barriers, merges or compiler differences.

Do not select aligned globally: T3 mixed-load full time is111.582us persistent
versus173.432us aligned. Both use budget9 and29 partials for the long request,
so task counts alone do not explain that difference. T5 mixed is599.416us
persistent versus578.730us aligned. Uniform512 compute is near native in both
modes, with the complete windows still paying a roughly6us planning cost.
Resource occupancy remains one CTA/SM at86656B; no kernel/API/default changed.

Lead decision: retain both explicit modes and all regressions. No new length
threshold, global default switch or automatic dispatch is justified by four
points. Next contribution slice should establish real compile/public-interface
compatibility and a shared-plan multi-layer integration anchor; metadata already
available to the caller can later drive routing without a GPU length readback.
Any future uniform-fast-path policy must preserve the skewed counterexample.
No extra profiler or full service rerun was launched for this diagnosis.

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
- Add relevant vLLM metadata/lifecycle tests and benchmark integration. The
  bounded real compile qualification above is complete, with the FP8 schema
  checker limitation explicitly recorded. Module measurements alone do not
  establish serving gains. Keep the actual local dispatcher registration;
  shared FlashInfer identity decorators were not changed as a workaround.
- Prepare maintainer-facing scope, limitations and evidence before an upstream
  PR. No upstream acceptance or submitted PR is claimed.
