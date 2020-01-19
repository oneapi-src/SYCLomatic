// RUN: dpct --format-range=none --usm-level=none -in-root %S -out-root %T %s %S/mf-kernel.cu -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/mf-test.dp.cpp
// RUN: FileCheck %S/mf-kernel.cuh --match-full-lines --input-file %T/mf-kernel.dp.hpp

#include "mf-kernel.cuh"

__global__ void cuda_hello(){
    test_foo();
}

void test() {
  // CHECK:     dpct::get_default_queue().submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         extern dpct::device_memory<volatile int, 0> g_mutex;
  // CHECK-NEXT:         auto g_mutex_acc_ct1 = g_mutex.get_access(cgh);
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class Reset_kernel_parameters_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1) * sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             Reset_kernel_parameters(g_mutex_acc_ct1.get_pointer());
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  Reset_kernel_parameters<<<1,1>>>();
  cuda_hello<<<2,2>>>();
}
