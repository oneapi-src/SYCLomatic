// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1
// RUN: dpct --format-range=none -out-root %T/compat_with_clang %s --cuda-include-path="%cuda-path/include" --stop-on-parse-err
// RUN: FileCheck %s --match-full-lines --input-file %T/compat_with_clang/compat_with_clang.dp.cpp

#include "cuda_fp16.h"

// CHECK: inline void foo1(sycl::half2 *array, sycl::half a) {
// CHECK-NEXT:   array[dpct::reverse_bits<unsigned int>(123)] = {a, sycl::vec<float, 1>{2.3f}.convert<sycl::half, sycl::rounding_mode::automatic>()[0]};
// CHECK-NEXT: }
__device__ inline void foo1(__half2 *array, __half a) {
  array[__brev(123)] = {a, __float2half(2.3f)};
}
