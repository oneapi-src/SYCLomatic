// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/or %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/or/or.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/or/or.dp.cpp -o %T/or/or.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void bitwise_or() {
  int x = 1, y = 2;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  // CHECK: u8 = x | y;
  asm("or.pred %0, %1, %2;" : "=r"(u8) : "r"(x), "r"(y));

  // CHECK: u16 = x | y;
  asm("or.b16 %0, %1, %2;" : "=r"(u16) : "r"(x), "r"(y));

  // CHECK: u32 = x | y;
  asm("or.b32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: u64 = x | y;
  asm("or.b64 %0, %1, %2;" : "=r"(u64) : "r"(x), "r"(y));
}

// clang-format on
