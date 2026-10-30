// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define XQA_BUCKET_HOST_DEVICE __host__ __device__
#else
#define XQA_BUCKET_HOST_DEVICE
#endif

namespace flashinfer::xqa_buckets {
XQA_BUCKET_HOST_DEVICE constexpr int aligned_boundary(int ordinal, int tiles, int count) {
  // Table capacity is independent of live length; ordinal*tiles can exceed int32.
  return int(int64_t(ordinal) * tiles / count);
}
}  // namespace flashinfer::xqa_buckets
#undef XQA_BUCKET_HOST_DEVICE
