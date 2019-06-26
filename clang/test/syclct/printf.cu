// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/printf.sycl.cpp

#include <stdio.h>

// CHECK: void device_test(cl::sycl::stream [[STREAM:stream_[a-f0-9]+]]) {
// CHECK-NEXT: /*
// CHECK-NEXT: SYCLCT1015:0: Please adjust the code manually.
// CHECK-NEXT: */
// CHECK-NEXT: [[STREAM]] << "TODO - output needs update" << cl::sycl::endl;
// CHECK-NEXT: }
__device__ void device_test() {
  printf("print test\n");
}

// CHECK: void kernel_test(cl::sycl::stream [[STREAM:stream_[a-f0-9]+]]) {
// CHECK-NEXT: device_test([[STREAM]]);
// CHECK-NEXT: /*
// CHECK-NEXT: SYCLCT1015:1: Please adjust the code manually.
// CHECK-NEXT: */
// CHECK-NEXT: [[STREAM]] << "TODO - output needs update" << cl::sycl::endl;
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
  // CHECK: {
  // CHECK-NEXT:  syclct::get_default_queue().submit(
  // CHECK-NEXT:    [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:      cl::sycl::stream [[STREAM:stream_[a-f0-9]+]](64 * 1024, 80, cgh);
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class kernel_test_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:        cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:        [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:           kernel_test([[STREAM]]);
  // CHECK-NEXT:      });
  // CHECK-NEXT:    });
  // CHECK-NEXT:}
  kernel_test<<<1, 1>>>();
  host_test();
}
