// RUN: syclct -in-root %S -out-root %T %s %S/mf-kernel.cu %S/mf-kernel.cuh -- -std=c++14 -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/mf-test.sycl.cpp
// RUN: FileCheck %S/mf-kernel.cuh --match-full-lines --input-file %T/mf-kernel.sycl.hpp

#include "mf-kernel.cuh"

void test() {
  // CHECK:     syclct::get_default_queue().submit(
  // CHECK-NEXT:       [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:         extern syclct::device_memory<volatile int, 0> g_mutex;
  // CHECK-NEXT:         auto g_mutex_acc_ct1 = g_mutex.get_access(cgh);
  // CHECK-NEXT:         cgh.parallel_for<syclct_kernel_name<class Reset_kernel_parameters_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             Reset_kernel_parameters(syclct::syclct_accessor<volatile int, syclct::device, 0>(g_mutex_acc_ct1));
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  Reset_kernel_parameters<<<1,1>>>();
}
