// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/copysign %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/copysign/copysign.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/copysign/copysign.dp.cpp -o %T/copysign/copysign.dp.o %}

// clang-format off
// CHECK: #include <cmath>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void copysign() {
  float f32;
  double f64;
  // CHECK: f32 = std::copysign(10.0f, -100.0f);
  asm("copysign.f32 %0, %1, %2;" : "=f"(f32) : "f"(-100.0f), "f"(10.0f));
  
  // CHECK: f64 = std::copysign(10.0f, -100.0f);
  asm("copysign.f64 %0, %1, %2;" : "=l"(f64) : "f"(-100.0f), "f"(10.0f));
}

// clang-format on
