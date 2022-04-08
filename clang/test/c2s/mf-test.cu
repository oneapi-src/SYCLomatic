// RUN: c2s --format-range=none --usm-level=none -in-root %S -out-root %T/mf-test %S/mf-kernel.cu %S/mf-func-included.cu %s -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/mf-test/mf-test.dp.cpp
// RUN: FileCheck %S/mf-kernel.cu --match-full-lines --input-file %T/mf-test/mf-kernel.dp.cpp
// RUN: FileCheck %S/mf-kernel.cuh --match-full-lines --input-file %T/mf-test/mf-kernel.dp.hpp
// RUN: FileCheck %S/mf-extern.cuh --match-full-lines --input-file %T/mf-test/mf-extern.dp.hpp
// RUN: FileCheck %S/mf-func-included.cu --match-full-lines --input-file %T/mf-test/mf-func-included.dp.cpp
// RUN: FileCheck %S/mf-func-included-angled.cu --match-full-lines --input-file %T/mf-test/mf-func-included-angled.dp.cpp
// RUN: FileCheck %S/mf-func-mid-included.cu --match-full-lines --input-file %T/mf-test/mf-func-mid-included.dp.cpp

// CHECK: #include "mf-kernel.dp.hpp"
// CHECK-NEXT#include "mf-extern.dp.hpp"
// CHECK-NEXT#include "mf-func-included.dp.cpp"
// CHECK-NEXT#include <mf-func-included-angled.dp.cpp>
// CHECK-NEXT#include "mf-mid.dp.cpp"
#include "mf-kernel.cuh"
#include "mf-extern.cuh"
#include "mf-func-included.cu"
#include <mf-func-included-angled.cu>
#include "mf-mid.cu"

__global__ void cuda_hello(){
    test_foo();
}

void test() {
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();

  // CHECK:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         extern c2s::global_memory<volatile int, 0> g_mutex;
  // CHECK-EMPTY:
  // CHECK-NEXT:         g_mutex.init();
  // CHECK-EMPTY:
  // CHECK-NEXT:         auto g_mutex_acc_ct1 = g_mutex.get_access(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<c2s_kernel_name<class Reset_kernel_parameters_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             Reset_kernel_parameters(g_mutex_acc_ct1.get_pointer());
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  Reset_kernel_parameters<<<1,1>>>();
  // CHECK: q_ct1.parallel_for<c2s_kernel_name<class cuda_hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:        cuda_hello();
  // CHECK-NEXT:      });
  cuda_hello<<<2,2>>>();

  // CHECK:          q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(360), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<c2s_kernel_name<class kernel_extern_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             kernel_extern(item_ct1, a_acc_ct1.get_pointer());
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  kernel_extern<<<1,1>>>();
}


void call_constAdd(float *C, int size);

// CHECK: float A[3 * 3] = {0.0625f, 0.125f,  0.0625f, 0.1250f, 0.250f,
// CHECK-NEXT:                 0.1250f, 0.0625f, 0.125f,  0.0625f};
float A[3 * 3] = {0.0625f, 0.125f,  0.0625f, 0.1250f, 0.250f,
                  0.1250f, 0.0625f, 0.125f,  0.0625f};

// CHECK: float A1, A4, A5;
float A1, A4, A5;

int main(void) {
  int size = 3 * 3 * sizeof(float);

  float *h_C = (float *)malloc(size);
  call_constAdd(h_C, size);

  // CHECK:  q_ct1.parallel_for<c2s_kernel_name<class static_func_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:          static_func();
  // CHECK-NEXT:        });
  static_func<<<1, 1>>>();

  // CHECK:  q_ct1.parallel_for<c2s_kernel_name<class static_func_in_anonymous_namespace_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:          static_func_in_anonymous_namespace();
  // CHECK-NEXT:        });
  static_func_in_anonymous_namespace<<<1, 1>>>();

  return 0;
}
