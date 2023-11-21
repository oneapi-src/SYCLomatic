// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/shl %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/shl/shl.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void shl() {
  int x = 1, y = 2;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  // CHECK: u16 = x << y;
  asm("shl.b16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: u32 = x << y;
  asm("shl.b32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: u64 = x << y;
  asm("shl.b64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}

// clang-format on
