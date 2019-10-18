// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_cache_config.dp.cpp

#include <stdio.h>

__global__ void simple_kernel(float *d_array) {
  return;
}

int checkCudaError(const cudaError_t err, const char* cmd)
{
    if(err) {
      exit(-1);
    }
    return err;
}

#define CHKERR(cmd) checkCudaError(cmd, #cmd)

int main(int argc, char **argv) {
  int size = 360;
  float *d_array;
  float h_array[360];

  // CHECK: dpct::dpct_malloc((void **)&d_array, sizeof(float) * size);
  cudaMalloc((void **)&d_array, sizeof(float) * size);
  // CHECK: CHKERR(0);
  CHKERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferNone));
  // CHECK: CHKERR(0);
  CHKERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  // CHECK: 0;
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  // CHECK: CHKERR(0);
  CHKERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual));

  // CHECK: if(CHKERR(0)) {
  // CHECK-NEXT:    return 0;
  // CHECK-NEXT: }
  if(CHKERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual))) {
    return 0;
  }

  // CHECK: if(0){
  // CHECK-NEXT:    return 0;
  // CHECK-NEXT:  }
  if(cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual)){
    return 0;
  }

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> d_array_buf_ct0 = dpct::get_buffer_and_offset(d_array);
  // CHECK-NEXT:   size_t d_array_offset_ct0 = d_array_buf_ct0.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto d_array_acc_ct0 = d_array_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(size / 64, 1, 1) * cl::sycl::range<3>(64, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(64, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           float *d_array_ct0 = (float *)(&d_array_acc_ct0[0] + d_array_offset_ct0);
  // CHECK-NEXT:           simple_kernel(d_array_ct0);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  simple_kernel<<<size / 64, 64>>>(d_array);

  cudaFuncCache pCacheConfig;
  // CHECK: CHKERR(0);
  CHKERR(cudaDeviceGetCacheConfig(&pCacheConfig));
  // CHECK: dpct::dpct_free(d_array);
  cudaFree(d_array);
  return 0;
}
