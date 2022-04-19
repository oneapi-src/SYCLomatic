// RUN: c2s --format-range=none --usm-level=none -in-root %S -out-root %T/multi-files-main %s %S/multi-files-device.cuh --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/multi-files-main/multi-files-main.dp.cpp
// RUN: FileCheck %S/multi-files-device.cuh --match-full-lines --input-file %T/multi-files-main/multi-files-device.dp.hpp

#include "multi-files-kernel.cuh"

int main() {
  unsigned *i_array;
  cudaMalloc((void **)&i_array, sizeof(unsigned) * 360);

  // CHECK: c2s::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     auto i_array_acc_ct0 = c2s::get_access(i_array, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<c2s_kernel_name<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 16), sycl::range<3>(1, 1, 16)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         simple_kernel((unsigned int *)(&i_array_acc_ct0[0]), item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  simple_kernel<<<16, 16>>>(i_array);

  cudaFree(i_array);

  sgemm();
  randomGen();
}
