// RUN: dpct -out-root %T/vadd %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/vadd/vadd.dp.cpp


// clang-format off
#include <cstdint>

__global__ void vadd() {
  int a, b, c, d;

  // CHECK: a = dpct::extend_add<int32_t>(b, c);
  asm("vadd.s32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_add<uint32_t>(b, c);
  asm("vadd.u32.u32.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_add_sat<int32_t>(b, c);
  asm("vadd.s32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_add_sat<uint32_t>(b, c);
  asm("vadd.u32.u32.s32.sat %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));

  // CHECK: a = dpct::extend_add_sat<int32_t>(b, c, d, sycl::plus<>());
  asm("vadd.s32.u32.s32.sat.add %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
  
  // CHECK: a = dpct::extend_add_sat<int32_t>(b, c, d, sycl::minimum<>());
  asm("vadd.s32.u32.s32.sat.min %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));

  // CHECK: a = dpct::extend_add_sat<int32_t>(b, c, d, sycl::maximum<>());
  asm("vadd.s32.u32.s32.sat.max %0, %1, %2, %3;" : "=r"(a) : "r"(b), "r"(c), "r"(d));
}

// clang-format on
