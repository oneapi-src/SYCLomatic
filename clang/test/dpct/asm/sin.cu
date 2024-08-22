// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/sin %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sin/sin.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/sin/sin.dp.cpp -o %T/sin/sin.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void sin() {
  float f32;

  // CHECK: f32 = sycl::sin(1.0f);
  asm("sin.approx.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  
  // CHECK: f32 = sycl::sin(1.0f);
  asm("sin.approx.ftz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
}

// clang-format on
