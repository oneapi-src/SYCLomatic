// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/sad %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sad/sad.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/sad/sad.dp.cpp -o %T/sad/sad.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void sad() {
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;

  // CHECK: i16 = sycl::abs_diff(i16, i16) + i16;
  asm("sad.s16 %0, %1, %2, %3;" : "=h"(i16) : "h"(i16), "h"(i16), "h"(i16));
  
  // CHECK: u16 = sycl::abs_diff(u16, u16) + u16;
  asm("sad.u16 %0, %1, %2, %3;" : "=h"(u16) : "h"(u16), "h"(u16), "h"(u16));
  
  // CHECK: i32 = sycl::abs_diff(i32, i32) + i32;
  asm("sad.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(i32), "r"(i32), "r"(i32));
  
  // CHECK: u32 = sycl::abs_diff(u32, u32) + u32;
  asm("sad.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(u32), "r"(u32), "r"(u32));
  
  // CHECK: i64 = sycl::abs_diff(i64, i64) + i64;
  asm("sad.s64 %0, %1, %2, %3;" : "=l"(i64) : "l"(i64), "l"(i64), "l"(i64));
  
  // CHECK: u64 = sycl::abs_diff(u64, u64) + u64;
  asm("sad.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(u64), "l"(u64), "l"(u64));
}

// clang-format on
