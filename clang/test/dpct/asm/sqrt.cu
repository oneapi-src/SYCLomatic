// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/sqrt %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sqrt/sqrt.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/sqrt/sqrt.dp.cpp -o %T/sqrt/sqrt.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

// CHECK: void sqrt() {
// CHECK-NEXT:   float f32;
// CHECK-NEXT:   double f64;
// CHECK-NEXT:   f32 = sycl::sqrt(1.0f);
// CHECK-NEXT:   f32 = sycl::sqrt(1.0f);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f32 = sycl::sqrt(1.0f);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f32 = sycl::sqrt(1.0f);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f32 = sycl::sqrt(1.0f);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f32 = sycl::sqrt(1.0f);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f32 = sycl::sqrt(1.0f);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f32 = sycl::sqrt(1.0f);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f32 = sycl::sqrt(1.0f);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f32 = sycl::sqrt(1.0f);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f64 = sycl::sqrt(1.0);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f64 = sycl::sqrt(1.0);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f64 = sycl::sqrt(1.0);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1013:{{.*}}: The rounding mode could not be specified and the generated code may have different accuracy than the original code. Verify the correctness. SYCL math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
// CHECK-NEXT:   */
// CHECK-NEXT:   f64 = sycl::sqrt(1.0);
// CHECK-NEXT: }
__global__ void sqrt() {
  float f32;
  double f64;
  asm("sqrt.approx.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  asm("sqrt.rn.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  asm("sqrt.rz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  asm("sqrt.rm.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  asm("sqrt.rp.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  asm("sqrt.rn.ftz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  asm("sqrt.rz.ftz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  asm("sqrt.rm.ftz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  asm("sqrt.rp.ftz.f32 %0, %1;" : "=f"(f32) : "f"(1.0f));
  asm("sqrt.rn.f64 %0, %1;" : "=d"(f64) : "d"(1.0));
  asm("sqrt.rz.f64 %0, %1;" : "=d"(f64) : "d"(1.0));
  asm("sqrt.rm.f64 %0, %1;" : "=d"(f64) : "d"(1.0));
  asm("sqrt.rp.f64 %0, %1;" : "=d"(f64) : "d"(1.0));
}

// clang-format on
