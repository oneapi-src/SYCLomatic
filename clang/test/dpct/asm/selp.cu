// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/selp %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/selp/selp.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/selp/selp.dp.cpp -o %T/selp/selp.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void selp() {
  int16_t i16;
  uint16_t u16;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;
  float f32;
  double f64;

  // CHECK: i16 = i16 == 1 ? i16 : i16;
  asm("selp.s16 %0, %1, %2, %3;" : "=h"(i16) : "h"(i16), "h"(i16), "h"(i16));
  
  // CHECK: u16 = u16 == 1 ? u16 : u16;
  asm("selp.u16 %0, %1, %2, %3;" : "=h"(u16) : "h"(u16), "h"(u16), "h"(u16));

  // CHECK: u16 = u16 == 1 ? u16 : u16;
  asm("selp.b16 %0, %1, %2, %3;" : "=h"(u16) : "h"(u16), "h"(u16), "h"(u16));
  
  // CHECK: i32 = i32 == 1 ? i32 : i32;
  asm("selp.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(i32), "r"(i32), "r"(i32));
  
  // CHECK: u32 = u32 == 1 ? u32 : u32;
  asm("selp.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(u32), "r"(u32), "r"(u32));

  // CHECK: u32 = u32 == 1 ? u32 : u32;
  asm("selp.b32 %0, %1, %2, %3;" : "=r"(u32) : "r"(u32), "r"(u32), "r"(u32));
  
  // CHECK: i64 = i64 == 1 ? i64 : i64;
  asm("selp.s64 %0, %1, %2, %3;" : "=l"(i64) : "l"(i64), "l"(i64), "l"(i64));
  
  // CHECK: u64 = u64 == 1 ? u64 : u64;
  asm("selp.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(u64), "l"(u64), "l"(u64));

  // CHECK: u64 = u64 == 1 ? u64 : u64;
  asm("selp.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(u64), "l"(u64), "l"(u64));

  // CHECK: f32 = f32 == 1 ? f32 : f32;
  asm("selp.f32 %0, %1, %2, %3;" : "=f"(f32) : "f"(f32), "f"(f32), "f"(f32));

  // CHECK: f64 = f64 == 1 ? f64 : f64;
  asm("selp.f64 %0, %1, %2, %3;" : "=d"(f64) : "d"(f64), "d"(f64), "d"(f64));
}

// clang-format on
