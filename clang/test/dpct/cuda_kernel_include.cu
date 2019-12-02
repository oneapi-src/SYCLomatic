// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only  -I ./
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_kernel_include.dp.cpp

// CHECK:#include <CL/sycl.hpp>
// CHECK-NEXT:#include <dpct/dpct.hpp>
#include <stdio.h>

// CHECK:#include "simple_kernel.dp.hpp"
#include "simple_kernel.cuh"

int main(int argc, char **argv) {
  int size = 360;
  float *d_array;
  float h_array[360];

  // CHECK: dpct::dpct_malloc((void **)&d_array, sizeof(float) * size);
  cudaMalloc((void **)&d_array, sizeof(float) * size);

  // CHECK: dpct::dpct_memset(d_array, 0, sizeof(float) * size);
  cudaMemset(d_array, 0, sizeof(float) * size);

  // CHECK: {
  // CHECK-NEXT:   dpct::buffer_t d_array_buf_ct0 = dpct::get_buffer(d_array);
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto d_array_acc_ct0 = d_array_buf_ct0.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, size / 64) * cl::sycl::range<3>(1, 1, 64), cl::sycl::range<3>(1, 1, 64)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           simple_kernel((float *)(&d_array_acc_ct0[0]), item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  simple_kernel<<<size / 64, 64>>>(d_array);

  // CHECK:  dpct::dpct_memcpy(h_array, d_array, 360 * sizeof(float), dpct::device_to_host);
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
