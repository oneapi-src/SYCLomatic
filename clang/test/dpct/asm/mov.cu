// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/mov %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/mov/mov.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

// CHECK:void mov() {
// CHECK-NEXT: unsigned p;
// CHECK-NEXT: double d;
// CHECK-NEXT: float f;
// CHECK-NEXT: p = 123 * 123U + 456 * ((4 ^ 7) + 2 ^ 3) | 777 & 128U == 2 != 3 > 4 < 5 <= 3 >= 5 >> 1 << 2 && 1 || 7 && !0;
// CHECK: f = sycl::bit_cast<float>(uint32_t(0x3f800000U));
// CHECK: f = sycl::bit_cast<float>(uint32_t(0x3f800000U));
// CHECK: d = sycl::bit_cast<double>(uint64_t(0x40091EB851EB851FULL));
// CHECK: d = sycl::bit_cast<double>(uint64_t(0x40091EB851EB851FULL));
// CHECK: }
__global__ void mov() {
  unsigned p;
  double d;
  float f;
  asm ("mov.s32 %0, 123 * 123U + 456 * ((4 ^7) + 2 ^ 3) | 777 & 128U == 2 != 3 > 4 < 5 <= 3 >= 5 >> 1 << 2 && 1 || 7 && !0;" : "=r"(p) );
  asm ("mov.f32 %0, 0F3f800000;" : "=f"(f));
  asm ("mov.f32 %0, 0f3f800000;" : "=f"(f));
  asm ("mov.f64 %0, 0D40091EB851EB851F;" : "=d"(d));
  asm ("mov.f64 %0, 0d40091EB851EB851F;" : "=d"(d));
  return;
}

// clang-format on
