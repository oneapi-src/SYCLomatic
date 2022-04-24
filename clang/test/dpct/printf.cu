// RUN: dpct --format-range=none --usm-level=none -out-root %T/printf %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/printf/printf.dp.cpp

#include <stdio.h>

// CHECK: void device_test(const sycl::stream &[[STREAM:stream_ct1]]) {
// CHECK-NEXT: [[STREAM]] << "print test\n";
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1015:0: Output needs adjustment.
// CHECK-NEXT: */
// CHECK-NEXT: [[STREAM]] << "print %s\n";
// CHECK-NEXT: }
__device__ void device_test() {
  printf("print test\n");
  printf("print %s\n", "test");
}

// CHECK: void kernel_test(const sycl::stream &[[STREAM:stream_ct1]]) {
// CHECK-NEXT: device_test([[STREAM]]);
// CHECK-NEXT: [[STREAM]] << "kernel test\n";
// CHECK-NEXT: }
__global__ void kernel_test() {
  device_test();
  printf("kernel test\n");
}

// CHECK: void host_test() {
// CHECK-NEXT:  printf("host test\n");
// CHECK-NEXT: }
void host_test() {
  printf("host test\n");
}

int main() {
  // CHECK:  dpct::get_default_queue().submit(
  // CHECK-NEXT:    [&](sycl::handler &cgh) {
  // CHECK-NEXT:      sycl::stream [[STREAM:stream_ct1]](64 * 1024, 80, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_test_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:        [=](sycl::nd_item<3> [[ITEM:item_ct1]]) {
  // CHECK-NEXT:           kernel_test([[STREAM]]);
  // CHECK-NEXT:      });
  // CHECK-NEXT:    });
  kernel_test<<<1, 1>>>();
  host_test();
}

