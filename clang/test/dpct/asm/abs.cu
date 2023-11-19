// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/abs %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/abs/abs.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void abs() {
  int x = 1;
  int16_t i16;
  int32_t i32;
  int64_t i64;

  // CHECK: i16 = sycl::abs(x);
  asm("abs.s16 %0, %1;" : "=r"(i16) : "r"(x));

  // CHECK: i32 = sycl::abs(x);
  asm("abs.s32 %0, %1;" : "=r"(i32) : "r"(x));

  // CHECK: i64 = sycl::abs(x);
  asm("abs.s64 %0, %1;" : "=r"(i64) : "r"(x));
}

// clang-format on
