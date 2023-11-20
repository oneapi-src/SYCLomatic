// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/mul24 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/mul24/mul24.dp.cpp

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void mul24() {
  int x = 1, y = 2;
  int32_t i32;
  uint32_t u32;

  // CHECK: i32 = sycl::mul24((int32_t)x, (int32_t)y);
  asm("mul24.lo.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));

  // CHECK: u32 = sycl::mul24((uint32_t)x, (uint32_t)y);
  asm("mul24.lo.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mul24.hi.s32 %0, %1, %2;" : "=r"(i32) : "r"(x), "r"(y));
  
  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm("mul24.hi.u32 %0, %1, %2;" : "=r"(u32) : "r"(x), "r"(y));
}

// clang-format on
