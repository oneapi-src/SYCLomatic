// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5
// RUN: dpct --format-range=none -out-root %T/math/half/half-cuda11.6-after %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/half/half-cuda11.6-after/half-cuda11.6-after.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/math/half/half-cuda11.6-after/half-cuda11.6-after.dp.cpp -o %T/math/half/half-cuda11.6-after/half-cuda11.6-after.dp.o %}

#include "cuda_fp16.h"

using namespace std;

__global__ void kernelFuncHalf(__half *deviceArrayHalf) {
  __half h, h_1, h_2;
  __half2 h2, h2_1, h2_2;
  double d;

  // Half Arithmetic Functions

  //  CHECK: h_2 = h + h_1;
  h_2 = __hadd_rn(h, h_1);
  // CHECK: h_2 = dpct::relu(sycl::fma(h, h_1, h_2));
  h_2 = __hfma_relu(h, h_1, h_2);
  // CHECK: h_2 = h * h_1;
  h_2 = __hmul_rn(h, h_1);
  // CHECK: h_2 = h - h_1;
  h_2 = __hsub_rn(h, h_1);
#ifndef NO_BUILD_TEST
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of half version of atomicAdd is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: atomicAdd(&h, h_1);
  atomicAdd(&h, h_1);
#endif

  // Half2 Arithmetic Functions

  // CHECK: h2_2 = h2 + h2_1;
  h2_2 = __hadd2_rn(h2, h2_1);
  // CHECK: h2_2 = dpct::complex_mul_add(h2, h2_1, h2_2);
  h2_2 = __hcmadd(h2, h2_1, h2_2);
  // CHECK: h2_2 = dpct::relu(sycl::fma(h2, h2_1, h2_2));
  h2_2 = __hfma2_relu(h2, h2_1, h2_2);
  // CHECK: h2_2 = h2 * h2_1;
  h2_2 = __hmul2_rn(h2, h2_1);
  // CHECK: h2_2 = h2 - h2_1;
  h2_2 = __hsub2_rn(h2, h2_1);
  // CHECK: dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&h2, h2_1);
  atomicAdd(&h2, h2_1);

  // Half Comparison Functions

  // CHECK: h_2 = sycl::fmax(h, h_1);
  h_2 = __hmax(h, h_1);
  // CHECK: h_2 = dpct::fmax_nan(h, h_1);
  h_2 = __hmax_nan(h, h_1);
  // CHECK: h_2 = sycl::fmin(h, h_1);
  h_2 = __hmin(h, h_1);
  // CHECK: h_2 = dpct::fmin_nan(h, h_1);
  h_2 = __hmin_nan(h, h_1);

  // Half2 Comparison Functions

  // CHECK: h2_2 = sycl::half2(sycl::fmax(h2[0], h2_1[0]), sycl::fmax(h2[1], h2_1[1]));
  h2_2 = __hmax2(h2, h2_1);
  // CHECK: h2_2 = dpct::fmax_nan(h2, h2_1);
  h2_2 = __hmax2_nan(h2, h2_1);
  // CHECK: h2_2 = sycl::half2(sycl::fmin(h2[0], h2_1[0]), sycl::fmin(h2[1], h2_1[1]));
  h2_2 = __hmin2(h2, h2_1);
  // CHECK: h2_2 = dpct::fmin_nan(h2, h2_1);
  h2_2 = __hmin2_nan(h2, h2_1);

  // Half Precision Conversion and Data Movement

  // CHECK: h_2 = sycl::half(d);
  h_2 = __double2half(d);
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

  // CHECK: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(deviceArrayHalf + 1) = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h2_2 = h2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *deviceArrayHalf = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stcs call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h2_2 = h2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(deviceArrayHalf + 1) = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwb call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h2_2 = h2;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: *deviceArrayHalf = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_2 = h;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '=' expression is used instead of the __stwt call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
  // CHECK-NEXT: */
  // CHECK-NEXT: h2_2 = h2;
  __stcg(deviceArrayHalf + 1, h);
  __stcg(&h_2, h);
  __stcg(&h2_2, h2);
  __stcs(deviceArrayHalf, h);
  __stcs(&h_2, h);
  __stcs(&h2_2, h2);
  __stwb(deviceArrayHalf + 1, h);
  __stwb(&h_2, h);
  __stwb(&h2_2, h2);
  __stwt(deviceArrayHalf, h);
  __stwt(&h_2, h);
  __stwt(&h2_2, h2);
}

int main() { return 0; }
