// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/mul %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/mul/mul.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void mul() {
  int x = 1, y = 2;
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;

  // CHECK: i16 = sycl::mul_hi((int16_t)x, (int16_t)y);
  asm("mul.hi.s16 %0, %1, %2;" : "=r"(i16) : "r"(x), "r"(y));

  // CHECK: u16 = sycl::mul_hi((uint16_t)x, (uint16_t)y);
  asm("mul.hi.u16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: i32 = sycl::mul_hi((int32_t)x, (int32_t)y);
  asm("mul.hi.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = sycl::mul_hi((uint32_t)x, (uint32_t)y);
  asm("mul.hi.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: i64 = sycl::mul_hi((int64_t)x, (int64_t)y);
  asm("mul.hi.s64 %0, %1, %2;" : "=r"(i64) : "r"(x), "r"(y));

  // CHECK: u64 = sycl::mul_hi((uint64_t)x, (uint64_t)y);
  asm("mul.hi.u64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));

  // CHECK: i16 = (int32_t)x * (int32_t)y;
  asm("mul.wide.s16 %0, %1, %2;" : "=r"(i16) : "r"(x), "r"(y));

  // CHECK: u16 = (uint32_t)x * (uint32_t)y;
  asm("mul.wide.u16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: i32 = (int64_t)x * (int64_t)y;
  asm("mul.wide.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = (uint64_t)x * (uint64_t)y;
  asm("mul.wide.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mul.wide.s64 %0, %1, %2;" : "=r"(i64) : "r"(x), "r"(y));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mul.wide.u64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}

// clang-format on
