// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1
// RUN: dpct --format-range=none -out-root %T/compat_nvcc %s --cuda-include-path="%cuda-path/include" --stop-on-parse-err
// RUN: FileCheck %s --match-full-lines --input-file %T/compat_nvcc/compat_nvcc.dp.cpp

#include "cuda_fp16.h"
#include <cstdint>

// CHECK: inline void foo1(sycl::half2 *array, sycl::half a) {
// CHECK-NEXT:   array[10] = {a, a};
// CHECK-NEXT: }
__device__ inline void foo1(__half2 *array, __half a) {
  array[10] = {a, a};
}

// CHECK: void foo2(int a, int b) {
// CHECK-NEXT:   sycl::range<3> block{1, 1, dpct::min(512, uint32_t(a * b))};
// CHECK-NEXT: }
void foo2(int a, int b) {
  dim3 block{min(512, uint32_t(a * b))};
}
