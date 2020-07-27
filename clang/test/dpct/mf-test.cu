// RUN: dpct --format-range=none --usm-level=none -in-root %S -out-root %T %s %S/mf-kernel.cu -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/mf-test.dp.cpp
// RUN: FileCheck %S/mf-kernel.cuh --match-full-lines --input-file %T/mf-kernel.dp.hpp

#include "mf-kernel.cuh"

__global__ void cuda_hello(){
    test_foo();
}

void test() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK:          q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         extern dpct::device_memory<volatile int, 0> g_mutex;
  // CHECK-EMPTY:
  // CHECK-NEXT:         auto g_mutex_acc_ct1 = g_mutex.get_access(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class Reset_kernel_parameters_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             Reset_kernel_parameters(g_mutex_acc_ct1.get_pointer());
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  Reset_kernel_parameters<<<1,1>>>();
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:  [&](sycl::handler &cgh) {
  // CHECK-NEXT:    cgh.parallel_for<dpct_kernel_name<class cuda_hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:        cuda_hello();
  // CHECK-NEXT:      });
  // CHECK-NEXT:  });
  cuda_hello<<<2,2>>>();
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
  return 0;
}
