// RUN: dpct -out-root %T/vmax %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/vmax/vmax.dp.cpp


// clang-format off
#include <cstdint>

__global__ void vmax() {
  int a, b, c, d;

  // CHECK: a = dpct::extend_max<int32_t>(b, c);
  asm("vmax.s32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_max<uint32_t>(b, c);
  asm("vmax.u32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_max_sat<int32_t>(b, c);
  asm("vmax.s32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_max_sat<uint32_t>(b, c);
  asm("vmax.u32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_max_sat<int32_t>(b, c, d, sycl::plus<>());
  asm("vmax.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
  
  // CHECK: a = dpct::extend_max_sat<int32_t>(b, c, d, sycl::minimum<>());
  asm("vmax.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));

  // CHECK: a = dpct::extend_max_sat<int32_t>(b, c, d, sycl::maximum<>());
  asm("vmax.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
}

// clang-format on
