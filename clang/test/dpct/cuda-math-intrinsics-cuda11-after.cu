// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/cuda-math-intrinsics-cuda11-after %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cuda-math-intrinsics-cuda11-after/cuda-math-intrinsics-cuda11-after.dp.cpp --match-full-lines %s

#include "cuda_fp16.h"

using namespace std;

__global__ void kernelFuncHalf(__half *deviceArrayHalf) {
  __half h, h_1, h_2;
  __half2 h2, h2_1, h2_2;

  // Half Precision Conversion and Data Movement

  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = *deviceArrayHalf;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldca call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h2_2 = h2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = *deviceArrayHalf;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h2_2 = h2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = *deviceArrayHalf;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h2_2 = h2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = *deviceArrayHalf;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldcv call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h2_2 = h2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = *deviceArrayHalf;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h2_2 = h2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = *deviceArrayHalf;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldlu call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h2_2 = h2;
  h_2 = __ldca(deviceArrayHalf);
  h_2 = __ldca(&h);
  h2_2 = __ldca(&h2);
  h_2 = __ldcg(deviceArrayHalf);
  h_2 = __ldcg(&h);
  h2_2 = __ldcg(&h2);
  h_2 = __ldcs(deviceArrayHalf);
  h_2 = __ldcs(&h);
  h2_2 = __ldcs(&h2);
  h_2 = __ldcv(deviceArrayHalf);
  h_2 = __ldcv(&h);
  h2_2 = __ldcv(&h2);
  h_2 = __ldg(deviceArrayHalf);
  h_2 = __ldg(&h);
  h2_2 = __ldg(&h2);
  h_2 = __ldlu(deviceArrayHalf);
  h_2 = __ldlu(&h);
  h2_2 = __ldlu(&h2);
}

int main() { return 0; }
