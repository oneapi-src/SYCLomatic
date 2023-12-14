// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/testp %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/testp/testp.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/testp/testp.dp.cpp -o %T/testp/testp.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void testp() {
  int val;

  // CHECK: val = !sycl::isinf(1.0f);
  asm("testp.finite.f32 %0, %1;" : "=r"(val) : "f"(1.0f));

  // CHECK: val = sycl::isinf(1.0f);
  asm("testp.infinite.f32 %0, %1;" : "=r"(val) : "f"(1.0f));

  // CHECK: val = !sycl::isnan(1.0f);
  asm("testp.number.f32 %0, %1;" : "=r"(val) : "f"(1.0f));

  // CHECK: val = sycl::isnan(1.0f);
  asm("testp.notanumber.f32 %0, %1;" : "=r"(val) : "f"(1.0f));

  // CHECK: val = sycl::isnormal(1.0f);
  asm("testp.normal.f32 %0, %1;" : "=r"(val) : "f"(1.0f));

  // CHECK: val = !sycl::isnormal(1.0f);
  asm("testp.subnormal.f32 %0, %1;" : "=r"(val) : "f"(1.0f));

  // CHECK: val = !sycl::isinf(1.0);
  asm("testp.finite.f64 %0, %1;" : "=r"(val) : "d"(1.0));

  // CHECK: val = sycl::isinf(1.0);
  asm("testp.infinite.f64 %0, %1;" : "=r"(val) : "d"(1.0));

  // CHECK: val = !sycl::isnan(1.0);
  asm("testp.number.f64 %0, %1;" : "=r"(val) : "d"(1.0));

  // CHECK: val = sycl::isnan(1.0);
  asm("testp.notanumber.f64 %0, %1;" : "=r"(val) : "d"(1.0));

  // CHECK: val = sycl::isnormal(1.0);
  asm("testp.normal.f64 %0, %1;" : "=r"(val) : "d"(1.0));

  // CHECK: val = !sycl::isnormal(1.0);
  asm("testp.subnormal.f64 %0, %1;" : "=r"(val) : "d"(1.0));
}

// clang-format on
