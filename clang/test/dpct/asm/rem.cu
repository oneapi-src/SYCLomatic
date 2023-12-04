// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/rem %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/rem/rem.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void rem() {
  int x = 1, y = 2;
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;

  // CHECK: i16 = x % y;
  asm("rem.s16 %0, %1, %2;" : "=r"(i16) : "r"(x), "r"(y));

  // CHECK: u16 = x % y;
  asm("rem.u16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: i32 = x % y;
  asm("rem.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = x % y;
  asm("rem.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: i64 = x % y;
  asm("rem.s64 %0, %1, %2;" : "=r"(i64) : "r"(x), "r"(y));

  // CHECK: u64 = x % y;
  asm("rem.u64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}


// clang-format on
