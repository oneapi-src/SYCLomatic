// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/tanh %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/tanh/tanh.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/tanh/tanh.dp.cpp -o %T/tanh/tanh.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void tanh() {
  float f32;

  // CHECK: f32 = sycl::tanh(1.0f);
  asm("tanh.approx.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
}

__global__ void f(unsigned *out) {
  unsigned const in = 0;
  // CHECK:  *out = sycl::tanh(sycl::vec<uint32_t, 1>(in).template as<sycl::half2>()).template as<sycl::vec<uint32_t, 1>>().x();
  asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(*out) : "r"(in));
}

// clang-format on
