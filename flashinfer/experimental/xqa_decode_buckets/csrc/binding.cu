// SPDX-License-Identifier: Apache-2.0
#include "tvm_ffi_utils.h"
#include "bucket.cuh"

namespace buckets = flashinfer::xqa_buckets;

namespace {
void check_tensor(TensorView t, DLDataType dtype, DLDevice device, int64_t elements) {
  CHECK_CUDA(t);
  CHECK_CONTIGUOUS(t);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(t.dtype()), encode_dlpack_dtype(dtype));
  TVM_FFI_ICHECK_EQ(t.device().device_id, device.device_id);
  TVM_FFI_ICHECK_EQ(t.numel(), elements);
}

buckets::PlanParams plan_params(TensorView lengths, TensorView tasks, TensorView requests,
                               TensorView spans, TensorView header, int64_t ctas, bool aligned) {
  int64_t batch = lengths.numel();
  TVM_FFI_ICHECK(batch >= 1 && batch <= buckets::MaxBatch && ctas >= batch &&
                 ctas <= INT32_MAX - buckets::MaxBatch);
  int64_t capacity = ctas + batch - 1;
  auto device = lengths.device();
  check_tensor(lengths, dl_int32, device, batch);
  check_tensor(tasks, dl_int32, device, capacity * 4);
  check_tensor(requests, dl_int32, device, batch * 2);
  check_tensor(spans, dl_int32, device, ctas * 2);
  check_tensor(header, dl_int32, device, 4);
#if FLASHINFER_XQA_PERSISTENT_TASKS
  TVM_FFI_ICHECK(!aligned);
#endif
  return {static_cast<const int*>(lengths.data_ptr()),
          static_cast<XQABucketTask*>(tasks.data_ptr()),
          static_cast<XQABucketRange*>(requests.data_ptr()),
          static_cast<XQABucketSpan*>(spans.data_ptr()), static_cast<int*>(header.data_ptr()),
          int(batch), int(ctas), int(aligned)};
}
}  // namespace

void bucket_plan(TensorView lengths, TensorView tasks, TensorView requests, TensorView spans,
                 TensorView header, int64_t ctas, bool aligned) {
  auto p = plan_params(lengths, tasks, requests, spans, header, ctas, aligned);
  ffi::CUDADeviceGuard device_guard(lengths.device().device_id);
  buckets::assign<<<1, 128, 0, get_stream(lengths.device())>>>(p);
  checkCuda(cudaPeekAtLastError());
}

int64_t bucket_scratch_bytes(int64_t capacity) {
  TVM_FFI_ICHECK(capacity >= 1 && capacity <= INT32_MAX);
  uint64_t size = buckets::scratch_bytes(int(capacity));
  TVM_FFI_ICHECK_LE(size, UINT32_MAX) << "Native XQA scratch byte offsets must fit uint32";
  return int64_t(size);
}

void bucket_configure(TensorView resources) {
  CHECK_CPU(resources);
  CHECK_CONTIGUOUS(resources);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(resources.dtype()), int64_code);
  TVM_FFI_ICHECK_EQ(resources.numel(), 4);
  checkCuda(cudaFuncSetAttribute(buckets::compute, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                sizeof(SharedMem)));
  cudaFuncAttributes attributes;
  checkCuda(cudaFuncGetAttributes(&attributes, buckets::compute));
  int resident = 0;
  checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, buckets::compute, 256,
                                                       sizeof(SharedMem)));
  auto* out = static_cast<int64_t*>(resources.data_ptr());
  out[0] = attributes.numRegs;
  out[1] = sizeof(SharedMem);
  out[2] = attributes.localSizeBytes;
  out[3] = resident;
}

void bucket_compute(TensorView q, TensorView k, TensorView v, TensorView ks, TensorView vs,
                    TensorView table, TensorView lengths, TensorView tasks, TensorView requests,
                    TensorView spans, TensorView header, TensorView semaphores, TensorView scratch,
                    TensorView output, int64_t ctas, bool aligned, double bmm1_scale,
                    double bmm2_scale, bool enable_pdl) {
  auto plan = plan_params(lengths, tasks, requests, spans, header, ctas, aligned);
  auto device = q.device();
  ffi::CUDADeviceGuard device_guard(device.device_id);
  check_tensor(q, dl_bfloat16, device, plan.batch * 6 * 256);
  check_tensor(output, dl_bfloat16, device, q.numel());
  check_tensor(table, dl_int32, device, table.numel());
  check_tensor(semaphores, dl_int32, device, plan.batch);
  check_tensor(scratch, dl_uint8, device, bucket_scratch_bytes(ctas + plan.batch - 1));
  TVM_FFI_ICHECK_EQ(lengths.device().device_id, device.device_id);
  CHECK_DIM(2, table);
  TVM_FFI_ICHECK_EQ(table.size(0), plan.batch);
  TVM_FFI_ICHECK(table.size(1) >= 1 && table.size(1) <= buckets::MaxTableWidth);
  for (auto t : {k, v, ks, vs}) {
    CHECK_CUDA(t);
    CHECK_LAST_DIM_CONTIGUOUS(t);
    CHECK_DIM(4, t);
    TVM_FFI_ICHECK_EQ(t.device().device_id, device.device_id);
    TVM_FFI_ICHECK_EQ(t.size(0), k.size(0));
    TVM_FFI_ICHECK_EQ(t.size(1), 32);
    TVM_FFI_ICHECK_EQ(t.size(2), 1);
  }
  CHECK_INPUT_TYPE(k, dl_uint8);
  CHECK_INPUT_TYPE(v, dl_uint8);
  CHECK_INPUT_TYPE(ks, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(vs, dl_float8_e4m3fn);
  TVM_FFI_ICHECK(k.size(3) == 128 && v.size(3) == 128 && ks.size(3) == 16 && vs.size(3) == 16);
  TVM_FFI_ICHECK(k.stride(0) == v.stride(0) && k.stride(1) == v.stride(1) &&
                 ks.stride(0) == vs.stride(0) && ks.stride(1) == vs.stride(1));
  TVM_FFI_ICHECK(k.stride(0) % sizeof(GMemCacheHead) == 0 &&
                 k.stride(1) % sizeof(GMemCacheHead) == 0 &&
                 ks.stride(0) % sizeof(GMemCacheHeadSf) == 0 &&
                 ks.stride(1) % sizeof(GMemCacheHeadSf) == 0);
  TVM_FFI_ICHECK(std::isfinite(bmm1_scale) && std::isfinite(bmm2_scale));
  buckets::ComputeParams p{plan, static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<const uint8_t*>(k.data_ptr()), static_cast<const uint8_t*>(v.data_ptr()),
      static_cast<const uint8_t*>(ks.data_ptr()), static_cast<const uint8_t*>(vs.data_ptr()),
      static_cast<const int*>(table.data_ptr()), static_cast<int*>(semaphores.data_ptr()),
      static_cast<uint8_t*>(scratch.data_ptr()), static_cast<__nv_bfloat16*>(output.data_ptr()),
      int(table.size(1)), k.stride(0), k.stride(1), ks.stride(0), ks.stride(1),
      float(bmm1_scale), float(bmm2_scale)};
  auto config = makeLaunchConfig(dim3(ctas), dim3(128, 1, 2), sizeof(SharedMem),
                                 get_stream(device), enable_pdl);
  checkCuda(cudaLaunchKernelEx(&config, buckets::compute, p));
  checkCuda(cudaPeekAtLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, bucket_plan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, bucket_compute);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(scratch_bytes, bucket_scratch_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure, bucket_configure);
