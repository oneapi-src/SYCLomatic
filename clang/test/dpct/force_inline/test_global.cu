// RUN: dpct --format-range=none -in-root %S -out-root %T/force_inline %S/test_device.cu %S/test_global.cu --optimize-migration --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/force_inline/test_global.dp.cpp --match-full-lines %s
#include <cuda.h>
#include <stdio.h>
// CHECK: SYCL_EXTERNAL extern void test_device(const sycl::stream &stream_ct1);
extern __device__ void test_device();
// CHECK: __dpct_inline__ void test_global(const sycl::nd_item<3> &item_ct1,
__global__ void test_global() {
    test_device();
    printf("%d\n", blockIdx.x);

}

// CHECK: __dpct_inline__ void test_global_inline(const sycl::nd_item<3> &item_ct1,
__global__ void test_global_inline() {
    printf("%d\n", blockIdx.x);

}

int main () {
    test_global<<<1,1>>>();
    test_global_inline<<<1,1>>>();
    return 0;
}
