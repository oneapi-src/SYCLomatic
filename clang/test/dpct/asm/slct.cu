// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/slct %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/slct/slct.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/slct/slct.dp.cpp -o %T/slct/slct.dp.o %}

// clang-format off
#include <cstdint>
#include <cuda_runtime.h>

__global__ void slct() {
  uint16_t u16, b16;
  uint32_t u32, b32;
  uint64_t u64, b64;
  int16_t s16;
  int32_t s32;
  int64_t s64;
  float f32;
  double f64;

  // Test slct.{b16|b32|b64|s16|s32|s64|u16|u32|u64|f32|f64}.s32
  // Test slct.{b16|b32|b64|s16|s32|s64|u16|u32|u64|f32|f64}.s32
  // CHECK: b16 = (s32 >= 0) ? b16 : u16;
  // CHECK: b32 = (s32 >= 0) ? b32 : u32;
  // CHECK: b64 = (s32 >= 0) ? b64 : u64;
  // CHECK: s16 = (s32 >= 0) ? s16 : u16;
  // CHECK: s32 = (s32 >= 0) ? s32 : u32;
  // CHECK: s64 = (s32 >= 0) ? s64 : u64;
  // CHECK: u16 = (s32 >= 0) ? u16 : u16;
  // CHECK: u32 = (s32 >= 0) ? u32 : u32;
  // CHECK: u64 = (s32 >= 0) ? u64 : u64;
  // CHECK: f32 = (s32 >= 0) ? f32 : f32;
  // CHECK: f64 = (s32 >= 0) ? f64 : f64;
  asm volatile ("slct.b16.s32 %0, %1, %2, %3;" : "=h"(b16) : "h"(b16), "h"(u16), "r"(s32));
  asm volatile ("slct.b32.s32 %0, %1, %2, %3;" : "=r"(b32) : "r"(b32), "r"(u32), "r"(s32));
  asm volatile ("slct.b64.s32 %0, %1, %2, %3;" : "=l"(b64) : "l"(b64), "l"(u64), "r"(s32));
  asm volatile ("slct.s16.s32 %0, %1, %2, %3;" : "=h"(s16) : "h"(s16), "h"(u16), "r"(s32));
  asm volatile ("slct.s32.s32 %0, %1, %2, %3;" : "=r"(s32) : "r"(s32), "r"(u32), "r"(s32));
  asm volatile ("slct.s64.s32 %0, %1, %2, %3;" : "=l"(s64) : "l"(s64), "l"(u64), "r"(s32));
  asm volatile ("slct.u16.s32 %0, %1, %2, %3;" : "=h"(u16) : "h"(u16), "h"(u16), "r"(s32));
  asm volatile ("slct.u32.s32 %0, %1, %2, %3;" : "=r"(u32) : "r"(u32), "r"(u32), "r"(s32));
  asm volatile ("slct.u64.s32 %0, %1, %2, %3;" : "=l"(u64) : "l"(u64), "l"(u64), "r"(s32));
  asm volatile ("slct.f32.s32 %0, %1, %2, %3;" : "=f"(f32) : "f"(f32), "f"(f32), "r"(s32));
  asm volatile ("slct.f64.s32 %0, %1, %2, %3;" : "=d"(f64) : "d"(f64), "d"(f64), "r"(s32));


  // Test slct.{b16|b32|b64|s16|s32|s64|u16|u32|u64|f32|f64}.f32
  // CHECK: b16 = (f32 >= 0.0f) ? b16 : u16;
  // CHECK: b32 = (f32 >= 0.0f) ? b32 : u32;
  // CHECK: b64 = (f32 >= 0.0f) ? b64 : u64;
  // CHECK: s16 = (f32 >= 0.0f) ? s16 : u16;
  // CHECK: s32 = (f32 >= 0.0f) ? s32 : u32;
  // CHECK: s64 = (f32 >= 0.0f) ? s64 : u64;
  // CHECK: u16 = (f32 >= 0.0f) ? u16 : u16;
  // CHECK: u32 = (f32 >= 0.0f) ? u32 : u32;
  // CHECK: u64 = (f32 >= 0.0f) ? u64 : u64;
  // CHECK: f32 = (f32 >= 0.0f) ? f32 : f32;
  // CHECK: f64 = (f32 >= 0.0f) ? f64 : f64;
  asm volatile ("slct.b16.f32 %0, %1, %2, %3;" : "=h"(b16) : "h"(b16), "h"(u16), "f"(f32));
  asm volatile ("slct.b32.f32 %0, %1, %2, %3;" : "=r"(b32) : "r"(b32), "r"(u32), "f"(f32));
  asm volatile ("slct.b64.f32 %0, %1, %2, %3;" : "=l"(b64) : "l"(b64), "l"(u64), "f"(f32));
  asm volatile ("slct.s16.f32 %0, %1, %2, %3;" : "=h"(s16) : "h"(s16), "h"(u16), "f"(f32));
  asm volatile ("slct.s32.f32 %0, %1, %2, %3;" : "=r"(s32) : "r"(s32), "r"(u32), "f"(f32));
  asm volatile ("slct.s64.f32 %0, %1, %2, %3;" : "=l"(s64) : "l"(s64), "l"(u64), "f"(f32));
  asm volatile ("slct.u16.f32 %0, %1, %2, %3;" : "=h"(u16) : "h"(u16), "h"(u16), "f"(f32));
  asm volatile ("slct.u32.f32 %0, %1, %2, %3;" : "=r"(u32) : "r"(u32), "r"(u32), "f"(f32));
  asm volatile ("slct.u64.f32 %0, %1, %2, %3;" : "=l"(u64) : "l"(u64), "l"(u64), "f"(f32));
  asm volatile ("slct.f32.f32 %0, %1, %2, %3;" : "=f"(f32) : "f"(f32), "f"(f32), "f"(f32));
  asm volatile ("slct.f64.f32 %0, %1, %2, %3;" : "=d"(f64) : "d"(f64), "d"(f64), "f"(f32));
}

// clang-format on
