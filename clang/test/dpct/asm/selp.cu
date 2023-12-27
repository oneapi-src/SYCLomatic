// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/selp %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/selp/selp.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/selp/selp.dp.cpp -o %T/selp/selp.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

// CHECK: void selp() {
// CHECK-NEXT:   int16_t i16, i161, i162;
// CHECK-NEXT:   uint16_t u16, u161, u162;
// CHECK-NEXT:   int32_t i32, i321, i322;
// CHECK-NEXT:   uint32_t u32, u321, u322;
// CHECK-NEXT:   int64_t i64, i641, i642;
// CHECK-NEXT:   uint64_t u64, u641, u642;
// CHECK-NEXT:   float f32, f321, f322;
// CHECK-NEXT:   double f64, f641, f642;
// CHECK-NEXT:   i16 = i162 == 1 ? i16 : i161;
// CHECK-NEXT:   u16 = u162 == 1 ? u16 : u161;
// CHECK-NEXT:   u16 = u162 == 1 ? u16 : u161;
// CHECK-NEXT:   i32 = i322 == 1 ? i32 : i321;
// CHECK-NEXT:   u32 = u322 == 1 ? u32 : u321;
// CHECK-NEXT:   u32 = u322 == 1 ? u32 : u321;
// CHECK-NEXT:   i64 = i642 == 1 ? i64 : i641;
// CHECK-NEXT:   u64 = u642 == 1 ? u64 : u641;
// CHECK-NEXT:   u64 = u642 == 1 ? u64 : u641;
// CHECK-NEXT:   f32 = f322 == 1 ? f32 : f321;
// CHECK-NEXT:   f64 = f642 == 1 ? f64 : f641;
// CHECK-NEXT: }
__global__ void selp() {
  int16_t i16, i161, i162;
  uint16_t u16, u161, u162;
  int32_t i32, i321, i322;
  uint32_t u32, u321, u322;
  int64_t i64, i641, i642;
  uint64_t u64, u641, u642;
  float f32, f321, f322;
  double f64, f641, f642;
  asm("selp.s16 %0, %1, %2, %3;" : "=h"(i16) : "h"(i16), "h"(i161), "h"(i162));
  asm("selp.u16 %0, %1, %2, %3;" : "=h"(u16) : "h"(u16), "h"(u161), "h"(u162));
  asm("selp.b16 %0, %1, %2, %3;" : "=h"(u16) : "h"(u16), "h"(u161), "h"(u162));
  asm("selp.s32 %0, %1, %2, %3;" : "=r"(i32) : "r"(i32), "r"(i321), "r"(i322));
  asm("selp.u32 %0, %1, %2, %3;" : "=r"(u32) : "r"(u32), "r"(u321), "r"(u322));
  asm("selp.b32 %0, %1, %2, %3;" : "=r"(u32) : "r"(u32), "r"(u321), "r"(u322));
  asm("selp.s64 %0, %1, %2, %3;" : "=l"(i64) : "l"(i64), "l"(i641), "l"(i642));
  asm("selp.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(u64), "l"(u641), "l"(u642));
  asm("selp.u64 %0, %1, %2, %3;" : "=l"(u64) : "l"(u64), "l"(u641), "l"(u642));
  asm("selp.f32 %0, %1, %2, %3;" : "=f"(f32) : "f"(f32), "f"(f321), "f"(f322));
  asm("selp.f64 %0, %1, %2, %3;" : "=d"(f64) : "d"(f64), "d"(f641), "d"(f642));
}

// clang-format on
