// SPDX-License-Identifier: Apache-2.0
// Compact interval ownership around the native Apache-licensed XQA core.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "task_geometry.h"

struct XQABucketTask { int request, begin, end, stride; };
struct XQABucketRange { int begin, count; };
struct XQABucketSpan { int begin, end; };
struct XQATaskContext {
  uint32_t request, slot, begin_tile, end_tile;
  uint32_t partial_begin, partial_count, total_slots;
  float qk_scale, v_scale;
  uint32_t reuse_barriers;
};

#if FLASHINFER_XQA_PERSISTENT_TASKS
#define FLASHINFER_XQA_PV_SYNC() namedBarSync(1, 128)
#endif
// Source-level task hooks retain the native loader, arithmetic and merge.
// The ordinary native launchers are excluded only for this JIT specialization.
#include "mha.cu"

namespace flashinfer::xqa_buckets {
constexpr int MaxBatch = 32;
constexpr int Tile = 256;
constexpr int MaxTableWidth = (0x7fffffff - (Tile - 1)) / 32;
struct PlanParams {
  const int* lengths;
  XQABucketTask* tasks;
  XQABucketRange* requests;
  XQABucketSpan* buckets;
  int* header;
  int batch, ctas, aligned;
};
struct ComputeParams {
  PlanParams plan;
  const __nv_bfloat16* q;
  const uint8_t *k, *v, *ks, *vs;
  const int* table;
  int* semaphores;
  uint8_t* scratch;
  __nv_bfloat16* output;
  int table_width;
  int64_t kv_page, kv_token, sf_page, sf_token;
  float qscale, vscale;
};

static_assert(ctaSize == 256 && ctaShapeInWarps.x == 4 && ctaShapeInWarps.y == 1 &&
              ctaShapeInWarps.z == 2 && grpLoadV && nbVBuffers == 2 &&
              gemm1NbWarpGrps == 1 && nbHeadSplits == 1 && headElems == 256 &&
              headGrpSize == 6 && SPEC_DEC == 0 && BEAM_WIDTH == 1 && USE_CUSTOM_BARRIER == 1,
              "Task/barrier ownership supports only the declared q1/Q6/KV1/D256 core");
static_assert(ctaTile.x == Tile);
static_assert(!FLASHINFER_XQA_DIRECT_TASKS || !FLASHINFER_XQA_PERSISTENT_TASKS,
              "Direct B1 uses the nonpersistent one-task core");
static_assert(sizeof(SharedMem::Barrier) == 8 &&
              (sizeof(SharedMem::xBarriers) / sizeof(CtaBarrierPair)) * 2 +
                  ctaShapeInWarps.y + SharedMem::nbOtherBarriers == 12);

__device__ __forceinline__ int ceildiv(int a, int b) { return (a + b - 1) / b; }

__global__ void assign(PlanParams p) {
  __shared__ int prefix[MaxBatch + 1], base[MaxBatch + 1], first[MaxBatch];
  __shared__ int budget, slots;
  if (threadIdx.x == 0) {
    prefix[0] = base[0] = 0;
    for (int r = 0; r < p.batch; ++r)
      prefix[r + 1] = prefix[r] + ceildiv(p.lengths[r], Tile);
    budget = max(1, ceildiv(prefix[p.batch], p.ctas));
    if (p.aligned) {
      int lo = 1, hi = 1;
      for (int r = 0; r < p.batch; ++r) hi = max(hi, prefix[r + 1] - prefix[r]);
      while (lo < hi) {
        int mid = (lo + hi) / 2, count = 0;
        for (int r = 0; r < p.batch; ++r)
          count += ceildiv(prefix[r + 1] - prefix[r], mid);
        if (count <= p.ctas) hi = mid; else lo = mid + 1;
      }
      budget = lo;
    }
    int max_parts = 0;
    for (int r = 0; r < p.batch; ++r) {
      int n = prefix[r + 1] - prefix[r];
      first[r] = prefix[r] / budget;
      int count = p.aligned ? ceildiv(n, budget) :
          (n ? (prefix[r + 1] - 1) / budget - first[r] + 1 : 0);
      p.requests[r] = {base[r], count};
      base[r + 1] = base[r] + count;
      max_parts = max(max_parts, count);
    }
    slots = base[p.batch];
    p.header[0] = budget;
    p.header[1] = slots;
    p.header[2] = p.aligned ? slots : ceildiv(prefix[p.batch], budget);
    p.header[3] = max_parts;
  }
  __syncthreads();
  for (int slot = threadIdx.x; slot < slots; slot += blockDim.x) {
    int r = 0;
    while (r + 1 < p.batch && base[r + 1] <= slot) ++r;
    int ordinal = slot - base[r], n = prefix[r + 1] - prefix[r];
    if (p.aligned) {
      int count = base[r + 1] - base[r];
      p.tasks[slot] = {r, aligned_boundary(ordinal, n, count),
                     aligned_boundary(ordinal + 1, n, count), 1};
    } else {
      int bucket = first[r] + ordinal;
      p.tasks[slot] = {r, max(prefix[r], bucket * budget) - prefix[r],
                     min(prefix[r + 1], (bucket + 1) * budget) - prefix[r], 1};
    }
  }
  // Compact slots are monotonic in global bucket, so these spans partition them.
  for (int bucket = threadIdx.x; bucket < p.ctas; bucket += blockDim.x) {
    if (p.aligned) {
      p.buckets[bucket] = bucket < slots ? XQABucketSpan{bucket, bucket + 1} : XQABucketSpan{0, 0};
      continue;
    }
    int lo = bucket * budget, hi = min(prefix[p.batch], lo + budget);
    if (lo >= hi) { p.buckets[bucket] = {0, 0}; continue; }
    int first_req = 0;
    while (first_req + 1 < p.batch && prefix[first_req + 1] <= lo) ++first_req;
    int last_req = first_req;
    while (last_req + 1 < p.batch && prefix[last_req + 1] < hi) ++last_req;
    p.buckets[bucket] = {base[first_req] + bucket - first[first_req],
                        base[last_req] + bucket - first[last_req] + 1};
  }
}

__global__ __launch_bounds__(256, 1) void compute(ComputeParams p) {
#if ENABLE_PDL
  preExit();
  acqBulk();  // Before every task, header or live-length read.
#endif
  auto const& plan = p.plan;
  int tid = threadIdx.x + 128 * threadIdx.z, owner = blockIdx.x;
  if (owner < plan.batch && plan.lengths[owner] == 0) {
    for (int i = tid; i < 6 * 256; i += 256) p.output[owner * 6 * 256 + i] = __float2bfloat16_rn(0.f);
    if (tid == 0) p.semaphores[owner] = 0;
  }
#if FLASHINFER_XQA_DIRECT_TASKS
  // Every active CTA derives the same live scratch layout. Descriptors are
  // deliberately absent; changing length during replay cannot strand metadata.
  int tiles = direct_tile_count(plan.lengths[0]);
  int total_slots = direct_split_count(plan.lengths[0], plan.ctas);
  if (owner >= total_slots) return;  // Includes zero length, before division.
  int slot = owner;
  XQABucketTask task{0, aligned_boundary(owner, tiles, total_slots),
                      aligned_boundary(owner + 1, tiles, total_slots), 1};
  XQABucketRange range{0, total_slots};
#elif FLASHINFER_XQA_PERSISTENT_TASKS
  XQABucketSpan span = plan.buckets[owner];
  bool reused = false;
  for (int slot = span.begin; slot < span.end; ++slot) {
#else
  int slot = owner;
  if (slot >= plan.header[1]) return;
#endif
#if !FLASHINFER_XQA_DIRECT_TASKS
    XQABucketTask task = plan.tasks[slot];
    XQABucketRange range = plan.requests[task.request];
    int total_slots = plan.header[1];
#endif
    XQATaskContext context{uint32_t(task.request), uint32_t(slot), uint32_t(task.begin),
        uint32_t(task.end), uint32_t(range.begin), uint32_t(range.count),
        uint32_t(total_slots), p.qscale, p.vscale,
#if FLASHINFER_XQA_PERSISTENT_TASKS
        uint32_t(reused)
#else
        0
#endif
    };
    KVCacheList<true> cache{reinterpret_cast<GMemCacheHead*>(const_cast<uint8_t*>(p.k)),
        reinterpret_cast<GMemCacheHead*>(const_cast<uint8_t*>(p.v)),
        reinterpret_cast<GMemCacheHeadSf*>(const_cast<uint8_t*>(p.ks)),
        reinterpret_cast<GMemCacheHeadSf*>(const_cast<uint8_t*>(p.vs)),
        reinterpret_cast<const KVCachePageIndex*>(p.table),
        reinterpret_cast<const uint32_t*>(plan.lengths), uint32_t(p.table_width)};
    kernel_mha_impl(context, 1, 1.f, nullptr, reinterpret_cast<OutputHead*>(p.output),
        reinterpret_cast<const IOHead*>(p.q), nullptr, cache, plan.batch, 1.f, nullptr,
        uint32_t(p.kv_page / sizeof(GMemCacheHead)), uint32_t(p.kv_token / sizeof(GMemCacheHead)), 1,
        uint32_t(p.sf_page / sizeof(GMemCacheHeadSf)), uint32_t(p.sf_token / sizeof(GMemCacheHeadSf)), 1,
        reinterpret_cast<uint32_t*>(p.semaphores), p.scratch);
#if FLASHINFER_XQA_PERSISTENT_TASKS
    // Join every producer/consumer before the next task reinitializes barriers.
    ldgsts::commitGroup();
    ldgsts::waitGroup<0>();
    __syncthreads();
    reused = true;
  }
#endif
}

inline uint64_t scratch_bytes(int capacity) {
  // Native MemSegmenter uses uint32 byte offsets. Size in uint64 first so the
  // binding can reject an overflow before any workspace allocation.
  Segmenter<uint64_t> layout;
  layout.newSeg<SMemWarpRowMax>(capacity, sizeof(SMemWarpRowMax));
  layout.newSeg<SMemWarpRowMax>(capacity, sizeof(SMemWarpRowMax));
  using ScratchBuf = Array2D<LdGrain, nbValidRows, SharedMem::XSmemBuffer::cols>;
  using Slices = Vec<ScratchBuf, gemm1WarpsPerGrp * nbHeadSplits>;
  layout.newSeg<Slices>(capacity, sizeof(Slices));
  return layout.getEndOffset();
}
}  // namespace flashinfer::xqa_buckets
