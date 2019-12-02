// RUN: dpct --usm-level=none -in-root %S -out-root %T %s %S/multi-files-device.cuh --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/multi-files-main.dp.cpp
// RUN: FileCheck %S/multi-files-device.cuh --match-full-lines --input-file %T/multi-files-device.dp.hpp
// RUN: FileCheck %S/multi-files-kernel.cuh --match-full-lines --input-file %T/multi-files-kernel.dp.hpp

#include "multi-files-kernel.cuh"

int main() {
  unsigned *i_array;
  cudaMalloc((void **)&i_array, sizeof(unsigned) * 360);

  // CHECK: {
  // CHECK-NEXT:   dpct::buffer_t i_array_buf_ct0 = dpct::get_buffer(i_array);
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto i_array_acc_ct0 = i_array_buf_ct0.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 16) * cl::sycl::range<3>(1, 1, 16), cl::sycl::range<3>(1, 1, 16)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           simple_kernel((unsigned int *)(&i_array_acc_ct0[0]), item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  simple_kernel<<<16, 16>>>(i_array);

  cudaFree(i_array);

  sgemm();
}
