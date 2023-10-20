// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

/// Half Arithmetic Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__habs | FileCheck %s -check-prefix=HABS
// HABS: CUDA API:
// HABS-NEXT:   __habs(h /*__half*/);
// HABS-NEXT:   __habs(b /*__nv_bfloat16*/);
// HABS-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HABS-NEXT:   sycl::ext::intel::math::habs(h);
// HABS-NEXT:   sycl::ext::oneapi::experimental::fabs(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hadd | FileCheck %s -check-prefix=__HADD
// __HADD: CUDA API:
// __HADD-NEXT:   __hadd(h1 /*__half*/, h2 /*__half*/);
// __HADD-NEXT:   __hadd(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HADD-NEXT:   __hadd(i1 /*int*/, i2 /*int*/);
// __HADD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HADD-NEXT:   sycl::ext::intel::math::hadd(h1, h2);
// __HADD-NEXT:   b1 + b2;
// __HADD-NEXT:   sycl::hadd(i1, i2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hadd_sat | FileCheck %s -check-prefix=HADD_SAT
// HADD_SAT: CUDA API:
// HADD_SAT-NEXT:   __hadd_sat(h1 /*__half*/, h2 /*__half*/);
// HADD_SAT-NEXT:   __hadd_sat(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// HADD_SAT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HADD_SAT-NEXT:   sycl::ext::intel::math::hadd_sat(h1, h2);
// HADD_SAT-NEXT:   dpct::clamp<sycl::ext::oneapi::bfloat16>(b1 + b2, 0.f, 1.0f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hdiv | FileCheck %s -check-prefix=HDIV
// HDIV: CUDA API:
// HDIV-NEXT:   __hdiv(h1 /*__half*/, h2 /*__half*/);
// HDIV-NEXT:   __hdiv(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// HDIV-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HDIV-NEXT:   sycl::ext::intel::math::hdiv(h1, h2);
// HDIV-NEXT:   b1 / b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hfma | FileCheck %s -check-prefix=HFMA
// HFMA: CUDA API:
// HFMA-NEXT:   __hfma(h1 /*__half*/, h2 /*__half*/, h3 /*__half*/);
// HFMA-NEXT:   __hfma(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/, b3 /*__nv_bfloat16*/);
// HFMA-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HFMA-NEXT:   sycl::ext::intel::math::hfma(h1, h2);
// HFMA-NEXT:   sycl::ext::oneapi::experimental::fma(b1, b2, b3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hfma_relu | FileCheck %s -check-prefix=HFMA_RELU
// HFMA_RELU: CUDA API:
// HFMA_RELU-NEXT:   __hfma_relu(h1 /*__half*/, h2 /*__half*/, h3 /*__half*/);
// HFMA_RELU-NEXT:   __hfma_relu(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/, b3 /*__nv_bfloat16*/);
// HFMA_RELU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HFMA_RELU-NEXT:   sycl::ext::intel::math::hfma_relu(h1, h2, h3);
// HFMA_RELU-NEXT:   dpct::relu(sycl::ext::oneapi::experimental::fma(b1, b2, b3));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hfma_sat | FileCheck %s -check-prefix=HFMA_SAT
// HFMA_SAT: CUDA API:
// HFMA_SAT-NEXT:   __hfma_sat(h1 /*__half*/, h2 /*__half*/, h3 /*__half*/);
// HFMA_SAT-NEXT:   __hfma_sat(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/, b3 /*__nv_bfloat16*/);
// HFMA_SAT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HFMA_SAT-NEXT:   sycl::ext::intel::math::hfma_sat(h1, h2, h3);
// HFMA_SAT-NEXT:   dpct::clamp<sycl::ext::oneapi::bfloat16>(sycl::ext::oneapi::experimental::fma(b1, b2, b3), 0.f, 1.0f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmul | FileCheck %s -check-prefix=HMUL
// HMUL: CUDA API:
// HMUL-NEXT:   __hmul(h1 /*__half*/, h2 /*__half*/);
// HMUL-NEXT:   __hmul(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// HMUL-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HMUL-NEXT:   sycl::ext::intel::math::hmul(h1, h2);
// HMUL-NEXT:   b1 * b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmul_sat | FileCheck %s -check-prefix=HMUL_SAT
// HMUL_SAT: CUDA API:
// HMUL_SAT-NEXT:   __hmul_sat(h1 /*__half*/, h2 /*__half*/);
// HMUL_SAT-NEXT:   __hmul_sat(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// HMUL_SAT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HMUL_SAT-NEXT:   sycl::ext::intel::math::hmul_sat(h1, h2);
// HMUL_SAT-NEXT:   dpct::clamp<sycl::ext::oneapi::bfloat16>(b1 * b2, 0.f, 1.0f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hneg | FileCheck %s -check-prefix=HNEG
// HNEG: CUDA API:
// HNEG-NEXT:   __hneg(h /*__half*/);
// HNEG-NEXT:   __hneg(b /*__nv_bfloat16*/);
// HNEG-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HNEG-NEXT:   sycl::ext::intel::math::hneg(h);
// HNEG-NEXT:   -b;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hsub | FileCheck %s -check-prefix=HSUB
// HSUB: CUDA API:
// HSUB-NEXT:   __hsub(h1 /*__half*/, h2 /*__half*/);
// HSUB-NEXT:   __hsub(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// HSUB-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HSUB-NEXT:   sycl::ext::intel::math::hsub(h1, h2);
// HSUB-NEXT:   b1 - b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hsub_sat | FileCheck %s -check-prefix=HSUB_SAT
// HSUB_SAT: CUDA API:
// HSUB_SAT-NEXT:   __hsub_sat(h1 /*__half*/, h2 /*__half*/);
// HSUB_SAT-NEXT:   __hsub_sat(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// HSUB_SAT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HSUB_SAT-NEXT:   sycl::ext::intel::math::hsub_sat(h1, h2);
// HSUB_SAT-NEXT:   dpct::clamp<sycl::ext::oneapi::bfloat16>(b1 - b2, 0.f, 1.0f);

/// Half2 Arithmetic Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__h2div | FileCheck %s -check-prefix=H2DIV
// H2DIV: CUDA API:
// H2DIV-NEXT:   __h2div(h1 /*__half2*/, h2 /*__half2*/);
// H2DIV-NEXT:   __h2div(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// H2DIV-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// H2DIV-NEXT:   sycl::ext::intel::math::h2div(h1, h2);
// H2DIV-NEXT:   b1 / b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__habs2 | FileCheck %s -check-prefix=HABS2
// HABS2: CUDA API:
// HABS2-NEXT:   __habs2(h /*__half2*/);
// HABS2-NEXT:   __habs2(b /*__nv_bfloat162*/);
// HABS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HABS2-NEXT:   sycl::ext::intel::math::habs2(h);
// HABS2-NEXT:   sycl::ext::oneapi::experimental::fabs(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hadd2 | FileCheck %s -check-prefix=HADD2
// HADD2: CUDA API:
// HADD2-NEXT:   __hadd2(h1 /*__half2*/, h2 /*__half2*/);
// HADD2-NEXT:   __hadd2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// HADD2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HADD2-NEXT:   sycl::ext::intel::math::hadd2(h1, h2);
// HADD2-NEXT:   b1 + b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hadd2_sat | FileCheck %s -check-prefix=HADD2_SAT
// HADD2_SAT: CUDA API:
// HADD2_SAT-NEXT:   __hadd2_sat(h1 /*__half2*/, h2 /*__half2*/);
// HADD2_SAT-NEXT:   __hadd2_sat(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// HADD2_SAT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HADD2_SAT-NEXT:   sycl::ext::intel::math::hadd2_sat(h1, h2);
// HADD2_SAT-NEXT:   dpct::clamp(b1 + b2, {0.f, 0.f}, {1.f, 1.f});

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hfma2 | FileCheck %s -check-prefix=HFMA2
// HFMA2: CUDA API:
// HFMA2-NEXT:   __hfma2(h1 /*__half2*/, h2 /*__half2*/, h3 /*__half2*/);
// HFMA2-NEXT:   __hfma2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/, b3 /*__nv_bfloat162*/);
// HFMA2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HFMA2-NEXT:   sycl::ext::intel::math::hfma2(h1, h2, h3);
// HFMA2-NEXT:   sycl::ext::oneapi::experimental::fma(b1, b2, b3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hfma2_relu | FileCheck %s -check-prefix=HFMA2_RELU
// HFMA2_RELU: CUDA API:
// HFMA2_RELU-NEXT:   __hfma2_relu(h1 /*__half2*/, h2 /*__half2*/, h3 /*__half2*/);
// HFMA2_RELU-NEXT:   __hfma2_relu(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/,
// HFMA2_RELU-NEXT:                b3 /*__nv_bfloat162*/);
// HFMA2_RELU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HFMA2_RELU-NEXT:   sycl::ext::intel::math::hfma2_relu(h1, h2, h3);
// HFMA2_RELU-NEXT:   dpct::relu(sycl::ext::oneapi::experimental::fma(b1, b2, b3));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hfma2_sat | FileCheck %s -check-prefix=HFMA2_SAT
// HFMA2_SAT: CUDA API:
// HFMA2_SAT-NEXT:   __hfma2_sat(h1 /*__half2*/, h2 /*__half2*/, h3 /*__half2*/);
// HFMA2_SAT-NEXT:   __hfma2_sat(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/,
// HFMA2_SAT-NEXT:               b3 /*__nv_bfloat162*/);
// HFMA2_SAT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HFMA2_SAT-NEXT:   sycl::ext::intel::math::hfma2_sat(h1, h2, h3);
// HFMA2_SAT-NEXT:   dpct::clamp(sycl::ext::oneapi::experimental::fma(b1, b2, b3), {0.f, 0.f}, {1.f, 1.f});

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmul2 | FileCheck %s -check-prefix=HMUL2
// HMUL2: CUDA API:
// HMUL2-NEXT:   __hmul2(h1 /*__half2*/, h2 /*__half2*/);
// HMUL2-NEXT:   __hmul2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// HMUL2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HMUL2-NEXT:   sycl::ext::intel::math::hmul2(h1, h2);
// HMUL2-NEXT:   b1 * b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmul2_sat | FileCheck %s -check-prefix=HMUL2_SAT
// HMUL2_SAT: CUDA API:
// HMUL2_SAT-NEXT:   __hmul2_sat(h1 /*__half2*/, h2 /*__half2*/);
// HMUL2_SAT-NEXT:   __hmul2_sat(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// HMUL2_SAT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HMUL2_SAT-NEXT:   sycl::ext::intel::math::hmul2_sat(h1, h2);
// HMUL2_SAT-NEXT:   dpct::clamp(b1 * b2, {0.f, 0.f}, {1.f, 1.f});

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hneg2 | FileCheck %s -check-prefix=HNEG2
// HNEG2: CUDA API:
// HNEG2-NEXT:   __hneg2(h /*__half2*/);
// HNEG2-NEXT:   __hneg2(b /*__nv_bfloat162*/);
// HNEG2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HNEG2-NEXT:   sycl::ext::intel::math::hneg2(h);
// HNEG2-NEXT:   -b;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hsub2 | FileCheck %s -check-prefix=HSUB2
// HSUB2: CUDA API:
// HSUB2-NEXT:   __hsub2(h1 /*__half2*/, h2 /*__half2*/);
// HSUB2-NEXT:   __hsub2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// HSUB2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HSUB2-NEXT:   sycl::ext::intel::math::hsub2(h1, h2);
// HSUB2-NEXT:   b1 - b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hsub2_sat | FileCheck %s -check-prefix=HSUB2_SAT
// HSUB2_SAT: CUDA API:
// HSUB2_SAT-NEXT:   __hsub2_sat(h1 /*__half2*/, h2 /*__half2*/);
// HSUB2_SAT-NEXT:   __hsub2_sat(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// HSUB2_SAT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HSUB2_SAT-NEXT:   sycl::ext::intel::math::hsub2_sat(h1, h2);
// HSUB2_SAT-NEXT:   dpct::clamp(b1 - b2, {0.f, 0.f}, {1.f, 1.f});

/// Half Precision Conversion And Data Movement

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2half | FileCheck %s -check-prefix=DOUBLE2HALF
// DOUBLE2HALF: CUDA API:
// DOUBLE2HALF-NEXT:   __double2half(h /*__half*/);
// DOUBLE2HALF-NEXT: Is migrated to:
// DOUBLE2HALF-NEXT:   sycl::half(h);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ldca | FileCheck %s -check-prefix=LDCA
// LDCA: CUDA API:
// LDCA-NEXT:   /* 1 */ __ldca(h /*__half **/);
// LDCA-NEXT:   /* 2 */ __ldca(h2 /*__half2 **/);
// LDCA-NEXT:   /* 3 */ __ldca(b /*__nv_bfloat16 **/);
// LDCA-NEXT:   /* 4 */ __ldca(b2 /*__nv_bfloat162 **/);
// LDCA-NEXT: Is migrated to:
// LDCA-NEXT:   /* 1 */ *h;
// LDCA-NEXT:   /* 2 */ *h2;
// LDCA-NEXT:   /* 3 */ *b;
// LDCA-NEXT:   /* 4 */ *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ldcg | FileCheck %s -check-prefix=LDCG
// LDCG: CUDA API:
// LDCG-NEXT:   /* 1 */ __ldcg(h /*__half **/);
// LDCG-NEXT:   /* 2 */ __ldcg(h2 /*__half2 **/);
// LDCG-NEXT:   /* 3 */ __ldcg(b /*__nv_bfloat16 **/);
// LDCG-NEXT:   /* 4 */ __ldcg(b2 /*__nv_bfloat162 **/);
// LDCG-NEXT: Is migrated to:
// LDCG-NEXT:   /* 1 */ *h;
// LDCG-NEXT:   /* 2 */ *h2;
// LDCG-NEXT:   /* 3 */ *b;
// LDCG-NEXT:   /* 4 */ *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ldcs | FileCheck %s -check-prefix=LDCS
// LDCS: CUDA API:
// LDCS-NEXT:   /* 1 */ __ldcs(h /*__half **/);
// LDCS-NEXT:   /* 2 */ __ldcs(h2 /*__half2 **/);
// LDCS-NEXT:   /* 3 */ __ldcs(b /*__nv_bfloat16 **/);
// LDCS-NEXT:   /* 4 */ __ldcs(b2 /*__nv_bfloat162 **/);
// LDCS-NEXT: Is migrated to:
// LDCS-NEXT:   /* 1 */ *h;
// LDCS-NEXT:   /* 2 */ *h2;
// LDCS-NEXT:   /* 3 */ *b;
// LDCS-NEXT:   /* 4 */ *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ldcv | FileCheck %s -check-prefix=LDCV
// LDCV: CUDA API:
// LDCV-NEXT:   /* 1 */ __ldcv(h /*__half **/);
// LDCV-NEXT:   /* 2 */ __ldcv(h2 /*__half2 **/);
// LDCV-NEXT:   /* 3 */ __ldcv(b /*__nv_bfloat16 **/);
// LDCV-NEXT:   /* 4 */ __ldcv(b2 /*__nv_bfloat162 **/);
// LDCV-NEXT: Is migrated to:
// LDCV-NEXT:   /* 1 */ *h;
// LDCV-NEXT:   /* 2 */ *h2;
// LDCV-NEXT:   /* 3 */ *b;
// LDCV-NEXT:   /* 4 */ *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ldg | FileCheck %s -check-prefix=LDG
// LDG: CUDA API:
// LDG-NEXT:   /* 1 */ __ldg(h /*__half **/);
// LDG-NEXT:   /* 2 */ __ldg(h2 /*__half2 **/);
// LDG-NEXT:   /* 3 */ __ldg(b /*__nv_bfloat16 **/);
// LDG-NEXT:   /* 4 */ __ldg(b2 /*__nv_bfloat162 **/);
// LDG-NEXT: Is migrated to:
// LDG-NEXT:   /* 1 */ *h;
// LDG-NEXT:   /* 2 */ *h2;
// LDG-NEXT:   /* 3 */ *b;
// LDG-NEXT:   /* 4 */ *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ldlu | FileCheck %s -check-prefix=LDLU
// LDLU: CUDA API:
// LDLU-NEXT:   /* 1 */ __ldlu(h /*__half **/);
// LDLU-NEXT:   /* 2 */ __ldlu(h2 /*__half2 **/);
// LDLU-NEXT:   /* 3 */ __ldlu(b /*__nv_bfloat16 **/);
// LDLU-NEXT:   /* 4 */ __ldlu(b2 /*__nv_bfloat162 **/);
// LDLU-NEXT: Is migrated to:
// LDLU-NEXT:   /* 1 */ *h;
// LDLU-NEXT:   /* 2 */ *h2;
// LDLU-NEXT:   /* 3 */ *b;
// LDLU-NEXT:   /* 4 */ *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__stcg | FileCheck %s -check-prefix=STCG
// STCG: CUDA API:
// STCG-NEXT:   /* 1 */ __stcg(ph /*__half **/, h /*__half*/);
// STCG-NEXT:   /* 2 */ __stcg(ph2 /*__half2 **/, h2 /*__half2*/);
// STCG-NEXT:   /* 3 */ __stcg(pb /*__nv_bfloat16 **/, b /*__nv_bfloat16*/);
// STCG-NEXT:   /* 4 */ __stcg(pb2 /*__nv_bfloat162 **/, b2 /*__nv_bfloat162*/);
// STCG-NEXT: Is migrated to:
// STCG-NEXT:   /* 1 */ *ph = h;
// STCG-NEXT:   /* 2 */ *ph2 = h2;
// STCG-NEXT:   /* 3 */ *pb = b;
// STCG-NEXT:   /* 4 */ *pb2 = b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__stcs | FileCheck %s -check-prefix=STCS
// STCS: CUDA API:
// STCS-NEXT:   /* 1 */ __stcs(ph /*__half **/, h /*__half*/);
// STCS-NEXT:   /* 2 */ __stcs(ph2 /*__half2 **/, h2 /*__half2*/);
// STCS-NEXT:   /* 3 */ __stcs(pb /*__nv_bfloat16 **/, b /*__nv_bfloat16*/);
// STCS-NEXT:   /* 4 */ __stcs(pb2 /*__nv_bfloat162 **/, b2 /*__nv_bfloat162*/);
// STCS-NEXT: Is migrated to:
// STCS-NEXT:   /* 1 */ *ph = h;
// STCS-NEXT:   /* 2 */ *ph2 = h2;
// STCS-NEXT:   /* 3 */ *pb = b;
// STCS-NEXT:   /* 4 */ *pb2 = b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__stwb | FileCheck %s -check-prefix=STWB
// STWB: CUDA API:
// STWB-NEXT:   /* 1 */ __stwb(ph /*__half **/, h /*__half*/);
// STWB-NEXT:   /* 2 */ __stwb(ph2 /*__half2 **/, h2 /*__half2*/);
// STWB-NEXT:   /* 3 */ __stwb(pb /*__nv_bfloat16 **/, b /*__nv_bfloat16*/);
// STWB-NEXT:   /* 4 */ __stwb(pb2 /*__nv_bfloat162 **/, b2 /*__nv_bfloat162*/);
// STWB-NEXT: Is migrated to:
// STWB-NEXT:   /* 1 */ *ph = h;
// STWB-NEXT:   /* 2 */ *ph2 = h2;
// STWB-NEXT:   /* 3 */ *pb = b;
// STWB-NEXT:   /* 4 */ *pb2 = b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__stwt | FileCheck %s -check-prefix=STWT
// STWT: CUDA API:
// STWT-NEXT:   /* 1 */ __stwt(ph /*__half **/, h /*__half*/);
// STWT-NEXT:   /* 2 */ __stwt(ph2 /*__half2 **/, h2 /*__half2*/);
// STWT-NEXT:   /* 3 */ __stwt(pb /*__nv_bfloat16 **/, b /*__nv_bfloat16*/);
// STWT-NEXT:   /* 4 */ __stwt(pb2 /*__nv_bfloat162 **/, b2 /*__nv_bfloat162*/);
// STWT-NEXT: Is migrated to:
// STWT-NEXT:   /* 1 */ *ph = h;
// STWT-NEXT:   /* 2 */ *ph2 = h2;
// STWT-NEXT:   /* 3 */ *pb = b;
// STWT-NEXT:   /* 4 */ *pb2 = b2;

/// Half Math Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hceil | FileCheck %s -check-prefix=HCEIL
// HCEIL: CUDA API:
// HCEIL-NEXT:   hceil(h /*__half*/);
// HCEIL-NEXT:   hceil(b /*__nv_bfloat16*/);
// HCEIL-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HCEIL-NEXT:   sycl::ext::intel::math::ceil(h);
// HCEIL-NEXT:   sycl::ext::oneapi::experimental::ceil(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hcos | FileCheck %s -check-prefix=HCOS
// HCOS: CUDA API:
// HCOS-NEXT:   hcos(h /*__half*/);
// HCOS-NEXT:   hcos(b /*__nv_bfloat16*/);
// HCOS-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HCOS-NEXT:   sycl::ext::intel::math::cos(h);
// HCOS-NEXT:   sycl::ext::oneapi::experimental::cos(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hexp | FileCheck %s -check-prefix=HEXP
// HEXP: CUDA API:
// HEXP-NEXT:   hexp(h /*__half*/);
// HEXP-NEXT:   hexp(b /*__nv_bfloat16*/);
// HEXP-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HEXP-NEXT:   sycl::ext::intel::math::exp(h);
// HEXP-NEXT:   sycl::ext::oneapi::experimental::exp(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hexp10 | FileCheck %s -check-prefix=HEXP10
// HEXP10: CUDA API:
// HEXP10-NEXT:   hexp10(h /*__half*/);
// HEXP10-NEXT:   hexp10(b /*__nv_bfloat16*/);
// HEXP10-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HEXP10-NEXT:   sycl::ext::intel::math::exp10(h);
// HEXP10-NEXT:   sycl::ext::oneapi::experimental::exp10(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hexp2 | FileCheck %s -check-prefix=HEXP2
// HEXP2: CUDA API:
// HEXP2-NEXT:   hexp2(h /*__half*/);
// HEXP2-NEXT:   hexp2(b /*__nv_bfloat16*/);
// HEXP2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HEXP2-NEXT:   sycl::ext::intel::math::exp2(h);
// HEXP2-NEXT:   sycl::ext::oneapi::experimental::exp2(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hfloor | FileCheck %s -check-prefix=HFLOOR
// HFLOOR: CUDA API:
// HFLOOR-NEXT:   hfloor(h /*__half*/);
// HFLOOR-NEXT:   hfloor(b /*__nv_bfloat16*/);
// HFLOOR-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HFLOOR-NEXT:   sycl::ext::intel::math::floor(h);
// HFLOOR-NEXT:   sycl::ext::oneapi::experimental::floor(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hlog | FileCheck %s -check-prefix=HLOG
// HLOG: CUDA API:
// HLOG-NEXT:   hlog(h /*__half*/);
// HLOG-NEXT:   hlog(b /*__nv_bfloat16*/);
// HLOG-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HLOG-NEXT:   sycl::ext::intel::math::log(h);
// HLOG-NEXT:   sycl::ext::oneapi::experimental::log(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hlog10 | FileCheck %s -check-prefix=HLOG10
// HLOG10: CUDA API:
// HLOG10-NEXT:   hlog10(h /*__half*/);
// HLOG10-NEXT:   hlog10(b /*__nv_bfloat16*/);
// HLOG10-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HLOG10-NEXT:   sycl::ext::intel::math::log10(h);
// HLOG10-NEXT:   sycl::ext::oneapi::experimental::log10(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hlog2 | FileCheck %s -check-prefix=HLOG2
// HLOG2: CUDA API:
// HLOG2-NEXT:   hlog2(h /*__half*/);
// HLOG2-NEXT:   hlog2(b /*__nv_bfloat16*/);
// HLOG2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HLOG2-NEXT:   sycl::ext::intel::math::log2(h);
// HLOG2-NEXT:   sycl::ext::oneapi::experimental::log2(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hrcp | FileCheck %s -check-prefix=HRCP
// HRCP: CUDA API:
// HRCP-NEXT:   hrcp(h /*__half*/);
// HRCP-NEXT:   hrcp(b /*__nv_bfloat16*/);
// HRCP-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HRCP-NEXT:   sycl::ext::intel::math::inv(h);
// HRCP-NEXT:   sycl::half_precision::recip(float(b));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hrint | FileCheck %s -check-prefix=HRINT
// HRINT: CUDA API:
// HRINT-NEXT:   hrint(h /*__half*/);
// HRINT-NEXT:   hrint(b /*__nv_bfloat16*/);
// HRINT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HRINT-NEXT:   sycl::ext::intel::math::rint(h);
// HRINT-NEXT:   sycl::ext::oneapi::experimental::rint(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hrsqrt | FileCheck %s -check-prefix=HRSQRT
// HRSQRT: CUDA API:
// HRSQRT-NEXT:   hrsqrt(h /*__half*/);
// HRSQRT-NEXT:   hrsqrt(b /*__nv_bfloat16*/);
// HRSQRT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HRSQRT-NEXT:   sycl::ext::intel::math::rsqrt(h);
// HRSQRT-NEXT:   sycl::ext::oneapi::experimental::rsqrt(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hsin | FileCheck %s -check-prefix=HSIN
// HSIN: CUDA API:
// HSIN-NEXT:   hsin(h /*__half*/);
// HSIN-NEXT:   hsin(b /*__nv_bfloat16*/);
// HSIN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HSIN-NEXT:   sycl::ext::intel::math::sin(h);
// HSIN-NEXT:   sycl::ext::oneapi::experimental::sin(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hsqrt | FileCheck %s -check-prefix=HSQRT
// HSQRT: CUDA API:
// HSQRT-NEXT:   hsqrt(h /*__half*/);
// HSQRT-NEXT:   hsqrt(b /*__nv_bfloat16*/);
// HSQRT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HSQRT-NEXT:   sycl::ext::intel::math::sqrt(h);
// HSQRT-NEXT:   sycl::ext::oneapi::experimental::sqrt(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=htrunc | FileCheck %s -check-prefix=HTRUNC
// HTRUNC: CUDA API:
// HTRUNC-NEXT:   htrunc(h /*__half*/);
// HTRUNC-NEXT:   htrunc(b /*__nv_bfloat16*/);
// HTRUNC-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// HTRUNC-NEXT:   sycl::ext::intel::math::trunc(h);
// HTRUNC-NEXT:   sycl::ext::oneapi::experimental::trunc(b);

/// Half2 Math Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2ceil | FileCheck %s -check-prefix=H2CEIL
// H2CEIL: CUDA API:
// H2CEIL-NEXT:   h2ceil(h /*__half2*/);
// H2CEIL-NEXT:   h2ceil(b /*__nv_bfloat162*/);
// H2CEIL-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2CEIL-NEXT:   sycl::ext::intel::math::ceil(h);
// H2CEIL-NEXT:   sycl::ext::oneapi::experimental::ceil(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2cos | FileCheck %s -check-prefix=H2COS
// H2COS: CUDA API:
// H2COS-NEXT:   h2cos(h /*__half2*/);
// H2COS-NEXT:   h2cos(b /*__nv_bfloat162*/);
// H2COS-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2COS-NEXT:   sycl::ext::intel::math::cos(h);
// H2COS-NEXT:   sycl::ext::oneapi::experimental::cos(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2exp | FileCheck %s -check-prefix=H2EXP
// H2EXP: CUDA API:
// H2EXP-NEXT:   h2exp(h /*__half2*/);
// H2EXP-NEXT:   h2exp(b /*__nv_bfloat162*/);
// H2EXP-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2EXP-NEXT:   sycl::ext::intel::math::exp(h);
// H2EXP-NEXT:   sycl::ext::oneapi::experimental::exp(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2exp10 | FileCheck %s -check-prefix=H2EXP10
// H2EXP10: CUDA API:
// H2EXP10-NEXT:   h2exp10(h /*__half2*/);
// H2EXP10-NEXT:   h2exp10(b /*__nv_bfloat162*/);
// H2EXP10-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2EXP10-NEXT:   sycl::ext::intel::math::exp10(h);
// H2EXP10-NEXT:   sycl::ext::oneapi::experimental::exp10(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2exp2 | FileCheck %s -check-prefix=H2EXP2
// H2EXP2: CUDA API:
// H2EXP2-NEXT:   h2exp2(h /*__half2*/);
// H2EXP2-NEXT:   h2exp2(b /*__nv_bfloat162*/);
// H2EXP2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2EXP2-NEXT:   sycl::ext::intel::math::exp2(h);
// H2EXP2-NEXT:   sycl::ext::oneapi::experimental::exp2(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2floor | FileCheck %s -check-prefix=H2FLOOR
// H2FLOOR: CUDA API:
// H2FLOOR-NEXT:   h2floor(h /*__half2*/);
// H2FLOOR-NEXT:   h2floor(b /*__nv_bfloat162*/);
// H2FLOOR-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2FLOOR-NEXT:   sycl::ext::intel::math::floor(h);
// H2FLOOR-NEXT:   sycl::ext::oneapi::experimental::floor(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2log | FileCheck %s -check-prefix=H2LOG
// H2LOG: CUDA API:
// H2LOG-NEXT:   h2log(h /*__half2*/);
// H2LOG-NEXT:   h2log(b /*__nv_bfloat162*/);
// H2LOG-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2LOG-NEXT:   sycl::ext::intel::math::log(h);
// H2LOG-NEXT:   sycl::ext::oneapi::experimental::log(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2log10 | FileCheck %s -check-prefix=H2LOG10
// H2LOG10: CUDA API:
// H2LOG10-NEXT:   h2log10(h /*__half2*/);
// H2LOG10-NEXT:   h2log10(b /*__nv_bfloat162*/);
// H2LOG10-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2LOG10-NEXT:   sycl::ext::intel::math::log10(h);
// H2LOG10-NEXT:   sycl::ext::oneapi::experimental::log10(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2log2 | FileCheck %s -check-prefix=H2LOG2
// H2LOG2: CUDA API:
// H2LOG2-NEXT:   h2log2(h /*__half2*/);
// H2LOG2-NEXT:   h2log2(b /*__nv_bfloat162*/);
// H2LOG2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2LOG2-NEXT:   sycl::ext::intel::math::log2(h);
// H2LOG2-NEXT:   sycl::ext::oneapi::experimental::log2(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2rcp | FileCheck %s -check-prefix=H2RCP
// H2RCP: CUDA API:
// H2RCP-NEXT:   h2rcp(h /*__half2*/);
// H2RCP-NEXT:   h2rcp(b /*__nv_bfloat162*/);
// H2RCP-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// H2RCP-NEXT:   sycl::ext::intel::math::inv(h);
// H2RCP-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2>(sycl::half_precision::recip(float(b[0])), sycl::half_precision::recip(float(b[1])));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2rint | FileCheck %s -check-prefix=H2RINT
// H2RINT: CUDA API:
// H2RINT-NEXT:   h2rint(h /*__half2*/);
// H2RINT-NEXT:   h2rint(b /*__nv_bfloat162*/);
// H2RINT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2RINT-NEXT:   sycl::ext::intel::math::rint(h);
// H2RINT-NEXT:   sycl::ext::oneapi::experimental::rint(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2rsqrt | FileCheck %s -check-prefix=H2RSQRT
// H2RSQRT: CUDA API:
// H2RSQRT-NEXT:   h2rsqrt(h /*__half2*/);
// H2RSQRT-NEXT:   h2rsqrt(b /*__nv_bfloat162*/);
// H2RSQRT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2RSQRT-NEXT:   sycl::ext::intel::math::rsqrt(h);
// H2RSQRT-NEXT:   sycl::ext::oneapi::experimental::rsqrt(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2sin | FileCheck %s -check-prefix=H2SIN
// H2SIN: CUDA API:
// H2SIN-NEXT:   h2sin(h /*__half2*/);
// H2SIN-NEXT:   h2sin(b /*__nv_bfloat162*/);
// H2SIN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2SIN-NEXT:   sycl::ext::intel::math::sin(h);
// H2SIN-NEXT:   sycl::ext::oneapi::experimental::sin(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2sqrt | FileCheck %s -check-prefix=H2SQRT
// H2SQRT: CUDA API:
// H2SQRT-NEXT:   h2sqrt(h /*__half2*/);
// H2SQRT-NEXT:   h2sqrt(b /*__nv_bfloat162*/);
// H2SQRT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2SQRT-NEXT:   sycl::ext::intel::math::sqrt(h);
// H2SQRT-NEXT:   sycl::ext::oneapi::experimental::sqrt(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=h2trunc | FileCheck %s -check-prefix=H2TRUNC
// H2TRUNC: CUDA API:
// H2TRUNC-NEXT:   h2trunc(h /*__half2*/);
// H2TRUNC-NEXT:   h2trunc(b /*__nv_bfloat162*/);
// H2TRUNC-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// H2TRUNC-NEXT:   sycl::ext::intel::math::trunc(h);
// H2TRUNC-NEXT:   sycl::ext::oneapi::experimental::trunc(b);
