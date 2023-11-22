// RUN: dpct -out-root %T/vmin %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/vmin/vmin.dp.cpp


// clang-format off
#include <cstdint>

__global__ void vmin() {
  int a, b, c, d;

  // CHECK: a = dpct::extend_min<int32_t>(b, c);
  asm("vmin.s32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_min<uint32_t>(b, c);
  asm("vmin.u32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_min_sat<int32_t>(b, c);
  asm("vmin.s32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_min_sat<uint32_t>(b, c);
  asm("vmin.u32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_min_sat<int32_t>(b, c, d, sycl::plus<>());
  asm("vmin.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
  
  // CHECK: a = dpct::extend_min_sat<int32_t>(b, c, d, sycl::minimum<>());
  asm("vmin.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));

  // CHECK: a = dpct::extend_min_sat<int32_t>(b, c, d, sycl::maximum<>());
  asm("vmin.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
}


// clang-format on
