// RUN: dpct -in-root %S -out-root %T %s %S/multi-files-device.cuh -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/multi-files-main.dp.cpp
// RUN: FileCheck %S/multi-files-device.cuh --match-full-lines --input-file %T/multi-files-device.dp.hpp
// RUN: FileCheck %S/multi-files-kernel.cuh --match-full-lines --input-file %T/multi-files-kernel.dp.hpp

#include "multi-files-kernel.cuh"

int main() {
  unsigned *i_array;
  cudaMalloc((void **)&i_array, sizeof(unsigned) * 360);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(i_array);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(16, 1, 1)), cl::sycl::range<3>(16, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           unsigned int *arg_ct0 = (unsigned int *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           simple_kernel(arg_ct0, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  simple_kernel<<<16, 16>>>(i_array);

  cudaFree(i_array);

  sgemm();
}
