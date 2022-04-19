// RUN: c2s --format-range=none --usm-level=none -out-root %T/cuda_kernel_include %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -I ./
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_kernel_include/cuda_kernel_include.dp.cpp

// CHECK:#include <CL/sycl.hpp>
// CHECK-NEXT:#include <c2s/c2s.hpp>
#include <stdio.h>

// CHECK:#include "simple_kernel.dp.hpp"
#include "simple_kernel.cuh"

// CHECK: void hello(sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:  int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
// CHECK-NEXT:  int tmp = sycl::min((unsigned int)((item_ct1.get_group(2)+1)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2)), (unsigned int)(sycl::max(index, 45)));
// CHECK-NEXT:  int num = sycl::max((unsigned int)((item_ct1.get_group(2)+1)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2)), (unsigned int)(sycl::min(tmp, 45)));
// CHECK-NEXT:  sycl::log((double)(item_ct1.get_group(2) + item_ct1.get_local_range(2) + item_ct1.get_local_id(2)));
// CHECK-NEXT:  sycl::log((double)(item_ct1.get_local_range(2) + item_ct1.get_local_range(1) + item_ct1.get_local_range(0)));
// CHECK-NEXT:}
__global__ void hello() {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp = min((blockIdx.x+1)*blockDim.x+threadIdx.x, max(index, 45));
  int num = max((blockIdx.x+1)*blockDim.x+threadIdx.x, min(tmp, 45));
  log(blockIdx.x + blockDim.x + threadIdx.x);
  log(blockDim.x + blockDim.y + blockDim.z);
}

int main(int argc, char **argv) {
  int size = 360;
  float *d_array;
  float h_array[360];

  // CHECK: d_array = (float *)c2s::c2s_malloc(sizeof(float) * size);
  cudaMalloc((void **)&d_array, sizeof(float) * size);

  // CHECK: c2s::c2s_memset(d_array, 0, sizeof(float) * size);
  cudaMemset(d_array, 0, sizeof(float) * size);

  // CHECK: c2s::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     auto d_array_acc_ct0 = c2s::get_access(d_array, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<c2s_kernel_name<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, size / 64) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         simple_kernel((float *)(&d_array_acc_ct0[0]), item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  simple_kernel<<<size / 64, 64>>>(d_array);

  // CHECK:  c2s::c2s_memcpy(h_array, d_array, 360 * sizeof(float), c2s::device_to_host);
  cudaMemcpy(h_array, d_array, 360 * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 1; i < 360; i++) {
    if (fabs(h_array[i] - 10.0) > 1e-5) {
      exit(-1);
    }
  }

  cudaFree(d_array);

  printf("Test Passed!\n");
  return 0;
}

