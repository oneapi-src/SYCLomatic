// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/fma %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/fma/fma.dp.cpp --check-prefix=CHECK
// RUN: %if build_lit %{icpx -c -fsycl %T/fma/fma.dp.cpp -o %T/fma/fma.dp.o %}
// RUN: rm -rf %T/fma/
// RUN: dpct --format-range=none --use-dpcpp-extensions=intel_device_math -out-root %T/fma %s --cuda-include-path="%cuda-path/include" --extra-arg="-DUSE_INTEL_DEVICE_MATH" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/fma/fma.dp.cpp --check-prefix=CHECK-INTEL-EXT
// RUN: %if build_lit %{icpx -c -fsycl -DUSE_INTEL_DEVICE_MATH %T/fma/fma.dp.cpp -o %T/fma/fma.dp.o %}

// clang-format off
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef USE_INTEL_DEVICE_MATH
__global__ void fma() {
  uint32_t u32a, u32b, u32c, u32d;
  uint16_t u16a, u16b, u16c, u16d;
  half f16a, f16b, f16c, f16d;
  float f32;
  double f64;

  // Test fma.{rn|rz|rm|rp}.f32
  // CHECK: f32 = sycl::fma(2.0f, 3.0f, 4.0f);
  // CHECK: f32 = sycl::fma(2.0f, 3.0f, 4.0f);
  // CHECK: f32 = sycl::fma(2.0f, 3.0f, 4.0f);
  // CHECK: f32 = sycl::fma(2.0f, 3.0f, 4.0f);
  asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(f32) : "f"(2.0f), "f"(3.0f), "f"(4.0f));
  asm volatile("fma.rz.f32 %0, %1, %2, %3;" : "=f"(f32) : "f"(2.0f), "f"(3.0f), "f"(4.0f));
  asm volatile("fma.rm.f32 %0, %1, %2, %3;" : "=f"(f32) : "f"(2.0f), "f"(3.0f), "f"(4.0f));
  asm volatile("fma.rp.f32 %0, %1, %2, %3;" : "=f"(f32) : "f"(2.0f), "f"(3.0f), "f"(4.0f));
  
  // Test fma.{rn|rz|rm|rp}.f64
  // CHECK: f64 = sycl::fma(2.0, 3.0, 4.0);
  // CHECK: f64 = sycl::fma(2.0, 3.0, 4.0);
  // CHECK: f64 = sycl::fma(2.0, 3.0, 4.0);
  // CHECK: f64 = sycl::fma(2.0, 3.0, 4.0);
  asm volatile("fma.rn.f64 %0, %1, %2, %3;" : "=d"(f64) : "d"(2.0), "d"(3.0), "d"(4.0));
  asm volatile("fma.rz.f64 %0, %1, %2, %3;" : "=d"(f64) : "d"(2.0), "d"(3.0), "d"(4.0));
  asm volatile("fma.rm.f64 %0, %1, %2, %3;" : "=d"(f64) : "d"(2.0), "d"(3.0), "d"(4.0));
  asm volatile("fma.rp.f64 %0, %1, %2, %3;" : "=d"(f64) : "d"(2.0), "d"(3.0), "d"(4.0));

  // Test fma.rn.{f16|bf16}
  // CHECK: u16d = sycl::fma(sycl::vec<uint16_t, 1>(u16a).template as<sycl::vec<sycl::half, 1>>(), sycl::vec<uint16_t, 1>(u16b).template as<sycl::vec<sycl::half, 1>>(), sycl::vec<uint16_t, 1>(u16c).template as<sycl::vec<sycl::half, 1>>()).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: f16d = sycl::fma(f16a, f16b, f16c);
  asm volatile("fma.rn.f16 %0, %1, %2, %3;" : "=h"(u16d) : "h"(u16a), "h"(u16b), "h"(u16c));
  asm volatile("fma.rn.f16 %0, %1, %2, %3;" : "=h"(f16d) : "h"(f16a), "h"(f16b), "h"(f16c));
  // asm volatile("fma.rn.bf16 %0, %1, %2, %3;" : "=h"(h) : "h"(e), "h"(f), "h"(g));
  
  // Test fma.rn.{f16x2|bf16x2}
  // CHECK: u32d = sycl::fma(sycl::vec<uint32_t, 1>(u32a).template as<sycl::half2>(), sycl::vec<uint32_t, 1>(u32b).template as<sycl::half2>(), sycl::vec<uint32_t, 1>(u32c).template as<sycl::half2>()).template as<sycl::vec<uint32_t, 1>>().x();
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(u32d) : "r"(u32a), "r"(u32b), "r"(u32c));
  // asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));

  // Test fma.rn.relu.{f16|bf16}
  // CHECK: u16d = sycl::fma(sycl::vec<uint16_t, 1>(u16a).template as<sycl::vec<sycl::half, 1>>(), sycl::vec<uint16_t, 1>(u16b).template as<sycl::vec<sycl::half, 1>>(), sycl::vec<uint16_t, 1>(u16c).template as<sycl::vec<sycl::half, 1>>()).template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile("fma.rn.relu.f16 %0, %1, %2, %3;" : "=h"(u16d) : "h"(u16a), "h"(u16b), "h"(u16c));
  // asm volatile("fma.rn.relu.bf16 %0, %1, %2, %3;" : "=h"(h) : "h"(e), "h"(f), "h"(g));

  // Test fma.rn.relu.{f16x2|bf16x2}
  // CHECK: u32d = sycl::fma(sycl::vec<uint32_t, 1>(u32a).template as<sycl::half2>(), sycl::vec<uint32_t, 1>(u32b).template as<sycl::half2>(), sycl::vec<uint32_t, 1>(u32c).template as<sycl::half2>()).template as<sycl::vec<uint32_t, 1>>().x();
  asm volatile("fma.rn.relu.f16x2 %0, %1, %2, %3;" : "=r"(u32d) : "r"(u32a), "r"(u32b), "r"(u32c));
  // asm volatile("fma.rn.relu.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));

  // Test fma.rn.sat.{f16|f16x2}
  // CHECK: u16d = sycl::fma(sycl::vec<uint16_t, 1>(u16a).template as<sycl::vec<sycl::half, 1>>(), sycl::vec<uint16_t, 1>(u16b).template as<sycl::vec<sycl::half, 1>>(), sycl::vec<uint16_t, 1>(u16c).template as<sycl::vec<sycl::half, 1>>()).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u32d = sycl::fma(sycl::vec<uint32_t, 1>(u32a).template as<sycl::half2>(), sycl::vec<uint32_t, 1>(u32b).template as<sycl::half2>(), sycl::vec<uint32_t, 1>(u32c).template as<sycl::half2>()).template as<sycl::vec<uint32_t, 1>>().x();
  asm volatile("fma.rn.sat.f16 %0, %1, %2, %3;" : "=h"(u16d) : "h"(u16a), "h"(u16b), "h"(u16c));
  asm volatile("fma.rn.sat.f16x2 %0, %1, %2, %3;" : "=r"(u32d) : "r"(u32a), "r"(u32b), "r"(u32c));
}
#else
__global__ void fma() {
  float f32;
  double f64;
  // Test fma.{rn|rz|rm|rp}.f32
  // CHECK-INTEL-EXT: f32 = sycl::ext::intel::math::fmaf_rn(2.0f, 3.0f, 4.0f);
  // CHECK-INTEL-EXT: f32 = sycl::ext::intel::math::fmaf_rz(2.0f, 3.0f, 4.0f);
  // CHECK-INTEL-EXT: f32 = sycl::ext::intel::math::fmaf_rd(2.0f, 3.0f, 4.0f);
  // CHECK-INTEL-EXT: f32 = sycl::ext::intel::math::fmaf_ru(2.0f, 3.0f, 4.0f);
  asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(f32) : "f"(2.0f), "f"(3.0f), "f"(4.0f));
  asm volatile("fma.rz.f32 %0, %1, %2, %3;" : "=f"(f32) : "f"(2.0f), "f"(3.0f), "f"(4.0f));
  asm volatile("fma.rm.f32 %0, %1, %2, %3;" : "=f"(f32) : "f"(2.0f), "f"(3.0f), "f"(4.0f));
  asm volatile("fma.rp.f32 %0, %1, %2, %3;" : "=f"(f32) : "f"(2.0f), "f"(3.0f), "f"(4.0f));
  
  // Test fma.{rn|rz|rm|rp}.f64
  // CHECK-INTEL-EXT: f64 = sycl::ext::intel::math::fma_rn(2.0, 3.0, 4.0);
  // CHECK-INTEL-EXT: f64 = sycl::ext::intel::math::fma_rz(2.0, 3.0, 4.0);
  // CHECK-INTEL-EXT: f64 = sycl::ext::intel::math::fma_rd(2.0, 3.0, 4.0);
  // CHECK-INTEL-EXT: f64 = sycl::ext::intel::math::fma_ru(2.0, 3.0, 4.0);
  asm volatile("fma.rn.f64 %0, %1, %2, %3;" : "=d"(f64) : "d"(2.0), "d"(3.0), "d"(4.0));
  asm volatile("fma.rz.f64 %0, %1, %2, %3;" : "=d"(f64) : "d"(2.0), "d"(3.0), "d"(4.0));
  asm volatile("fma.rm.f64 %0, %1, %2, %3;" : "=d"(f64) : "d"(2.0), "d"(3.0), "d"(4.0));
  asm volatile("fma.rp.f64 %0, %1, %2, %3;" : "=d"(f64) : "d"(2.0), "d"(3.0), "d"(4.0));
}
#endif

// clang-format on
