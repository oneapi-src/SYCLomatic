// RUN: dpct --format-range=none --usm-level=none -out-root %T/cuda_cache_config %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_cache_config/cuda_cache_config.dp.cpp

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

  // CHECK: d_array = (float *)dpct::dpct_malloc(sizeof(float) * size);
  cudaMalloc((void **)&d_array, sizeof(float) * size);
  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaDeviceSetCacheConfig was replaced with 0 because SYCL currently does not support setting cache config on devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: CHKERR(0);
  CHKERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferNone));

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaDeviceSetCacheConfig was replaced with 0 because SYCL currently does not support setting cache config on devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: CHKERR(0);
  CHKERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaDeviceSetCacheConfig was removed because SYCL currently does not support setting cache config on devices.
  // CHECK-NEXT: */
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaDeviceSetCacheConfig was replaced with 0 because SYCL currently does not support setting cache config on devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: CHKERR(0);
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

  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     auto d_array_acc_ct0 = dpct::get_access(d_array, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, size / 64) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         simple_kernel((float *)(&d_array_acc_ct0[0]));
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  simple_kernel<<<size / 64, 64>>>(d_array);

  cudaFuncCache pCacheConfig;
  // CHECK: CHKERR(0);
  CHKERR(cudaDeviceGetCacheConfig(&pCacheConfig));
  // CHECK: dpct::dpct_free(d_array);
  cudaFree(d_array);
  return 0;
}

