// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/sqrt %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sqrt/sqrt.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/sqrt/sqrt.dp.cpp -o %T/sqrt/sqrt.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void sqrt() {
  float f32;
  double f64;
  // CHECK: f32 = sycl::sqrt<float>(1.0f);
  asm("sqrt.approx.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));

  // CHECK: f32 = dpct::flush_denormal_to_zero(sycl::sqrt<float>(1.0f));
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  
  // CHECK: f32 = sycl::vec<float, 1>(sycl::sqrt<float>(1.0f)).convert<float, sycl::rounding_mode::rte>()[0];
  asm("sqrt.rn.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  
  // CHECK: f32 = sycl::vec<float, 1>(sycl::sqrt<float>(1.0f)).convert<float, sycl::rounding_mode::rtz>()[0];
  asm("sqrt.rz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  
  // CHECK: f32 = sycl::vec<float, 1>(sycl::sqrt<float>(1.0f)).convert<float, sycl::rounding_mode::rtn>()[0];
  asm("sqrt.rm.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  
  // CHECK: f32 = sycl::vec<float, 1>(sycl::sqrt<float>(1.0f)).convert<float, sycl::rounding_mode::rtp>()[0];
  asm("sqrt.rp.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  
  // CHECK: f32 = dpct::flush_denormal_to_zero(sycl::vec<float, 1>(sycl::sqrt<float>(1.0f)).convert<float, sycl::rounding_mode::rte>()[0]);
  asm("sqrt.rn.ftz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  
  // CHECK: f32 = dpct::flush_denormal_to_zero(sycl::vec<float, 1>(sycl::sqrt<float>(1.0f)).convert<float, sycl::rounding_mode::rtz>()[0]);
  asm("sqrt.rz.ftz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  
  // CHECK: f32 = dpct::flush_denormal_to_zero(sycl::vec<float, 1>(sycl::sqrt<float>(1.0f)).convert<float, sycl::rounding_mode::rtn>()[0]);
  asm("sqrt.rm.ftz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  
  // CHECK: f32 = dpct::flush_denormal_to_zero(sycl::vec<float, 1>(sycl::sqrt<float>(1.0f)).convert<float, sycl::rounding_mode::rtp>()[0]);
  asm("sqrt.rp.ftz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  
  // CHECK: f64 = sycl::vec<double, 1>(sycl::sqrt<double>(1.0)).convert<double, sycl::rounding_mode::rte>()[0];
  asm("sqrt.rn.f64 %0, %1;" : "=d"(f64) : "d"(1.0));
  
  // CHECK: f64 = sycl::vec<double, 1>(sycl::sqrt<double>(1.0)).convert<double, sycl::rounding_mode::rtz>()[0];
  asm("sqrt.rz.f64 %0, %1;" : "=d"(f64) : "d"(1.0));
  
  // CHECK: f64 = sycl::vec<double, 1>(sycl::sqrt<double>(1.0)).convert<double, sycl::rounding_mode::rtn>()[0];
  asm("sqrt.rm.f64 %0, %1;" : "=d"(f64) : "d"(1.0));
  
  // CHECK: f64 = sycl::vec<double, 1>(sycl::sqrt<double>(1.0)).convert<double, sycl::rounding_mode::rtp>()[0];
  asm("sqrt.rp.f64 %0, %1;" : "=d"(f64) : "d"(1.0));
}

// clang-format on
