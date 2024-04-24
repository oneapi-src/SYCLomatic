// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/rcp %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/rcp/rcp.dp.cpp --check-prefix=CHECK
// RUN: %if build_lit %{icpx -c -fsycl %T/rcp/rcp.dp.cpp -o %T/rcp/rcp.dp.o %}
// RUN: rm -rf %T/rcp/
// RUN: dpct --format-range=none --use-dpcpp-extensions=intel_device_math -out-root %T/rcp %s --cuda-include-path="%cuda-path/include" --extra-arg="-DUSE_INTEL_DEVICE_MATH" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/rcp/rcp.dp.cpp --check-prefix=CHECK-INTEL-EXT
// RUN: %if build_lit %{icpx -c -fsycl -DUSE_INTEL_DEVICE_MATH %T/rcp/rcp.dp.cpp -o %T/rcp/rcp.dp.o %}

// clang-format off
#include <cuda_runtime.h>

#ifndef USE_INTEL_DEVICE_MATH
__global__ void rcp() {
  float f;
  double d;
  // CHECK: f = 1 / 2.0f;
  // CHECK: f = 1 / 2.0f;
  // CHECK: f = 1 / 2.0f;
  // CHECK: f = 1 / 2.0f;
  // CHECK: f = 1 / 2.0f;
  // CHECK: d = 1 / 2.0f;
  // CHECK: d = 1 / 2.0f;
  // CHECK: d = 1 / 2.0f;
  // CHECK: d = 1 / 2.0f;
  asm volatile ("rcp.approx.f32 %0, %1;" : "=f"(f) : "f"(2.0f));
  asm volatile ("rcp.rn.f32 %0, %1;" : "=f"(f) : "f"(2.0f));
  asm volatile ("rcp.rz.f32 %0, %1;" : "=f"(f) : "f"(2.0f));
  asm volatile ("rcp.rm.f32 %0, %1;" : "=f"(f) : "f"(2.0f));
  asm volatile ("rcp.rp.f32 %0, %1;" : "=f"(f) : "f"(2.0f));
  asm volatile ("rcp.rn.f64 %0, %1;" : "=d"(d) : "d"(2.0f));
  asm volatile ("rcp.rz.f64 %0, %1;" : "=d"(d) : "d"(2.0f));
  asm volatile ("rcp.rm.f64 %0, %1;" : "=d"(d) : "d"(2.0f));
  asm volatile ("rcp.rp.f64 %0, %1;" : "=d"(d) : "d"(2.0f));
}
#else
__global__ void rcp() {
  float f;
  double d;
  // CHECK-INTEL-EXT: f = sycl::ext::intel::math::frcp_rn(2.0f);
  // CHECK-INTEL-EXT: f = sycl::ext::intel::math::frcp_rz(2.0f);
  // CHECK-INTEL-EXT: f = sycl::ext::intel::math::frcp_rd(2.0f);
  // CHECK-INTEL-EXT: f = sycl::ext::intel::math::frcp_ru(2.0f);
  // CHECK-INTEL-EXT: d = sycl::ext::intel::math::drcp_rn(2.0f);
  // CHECK-INTEL-EXT: d = sycl::ext::intel::math::drcp_rz(2.0f);
  // CHECK-INTEL-EXT: d = sycl::ext::intel::math::drcp_rd(2.0f);
  // CHECK-INTEL-EXT: d = sycl::ext::intel::math::drcp_ru(2.0f);
  asm volatile ("rcp.rn.f32 %0, %1;" : "=f"(f) : "f"(2.0f));
  asm volatile ("rcp.rz.f32 %0, %1;" : "=f"(f) : "f"(2.0f));
  asm volatile ("rcp.rm.f32 %0, %1;" : "=f"(f) : "f"(2.0f));
  asm volatile ("rcp.rp.f32 %0, %1;" : "=f"(f) : "f"(2.0f));
  asm volatile ("rcp.rn.f64 %0, %1;" : "=d"(d) : "d"(2.0f));
  asm volatile ("rcp.rz.f64 %0, %1;" : "=d"(d) : "d"(2.0f));
  asm volatile ("rcp.rm.f64 %0, %1;" : "=d"(d) : "d"(2.0f));
  asm volatile ("rcp.rp.f64 %0, %1;" : "=d"(d) : "d"(2.0f));
}
#endif

// clang-format on
