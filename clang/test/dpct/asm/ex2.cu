// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/ex2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/ex2/ex2.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/ex2/ex2.dp.cpp -o %T/ex2/ex2.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void ex2() {
  float f32;
  // CHECK: f32 = sycl::pow(2, 3.4f);
  asm("ex2.approx.f32 %0, %1;" : "=f"(f32) : "f"(3.4f));
  
  // CHECK: f32 = sycl::pow(2, 3.4f);
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(f32) : "f"(3.4f));
}

// clang-format on
