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
// LDCA-NEXT:   __ldca(h /*__half **/);
// LDCA-NEXT:   __ldca(h2 /*__half2 **/);
// LDCA-NEXT:   __ldca(b /*__nv_bfloat16 **/);
// LDCA-NEXT:   __ldca(b2 /*__nv_bfloat162 **/);
// LDCA-NEXT: Is migrated to:
// LDCA-NEXT:   *h;
// LDCA-NEXT:   *h2;
// LDCA-NEXT:   *b;
// LDCA-NEXT:   *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ldcg | FileCheck %s -check-prefix=LDCG
// LDCG: CUDA API:
// LDCG-NEXT:   __ldcg(h /*__half **/);
// LDCG-NEXT:   __ldcg(h2 /*__half2 **/);
// LDCG-NEXT:   __ldcg(b /*__nv_bfloat16 **/);
// LDCG-NEXT:   __ldcg(b2 /*__nv_bfloat162 **/);
// LDCG-NEXT: Is migrated to:
// LDCG-NEXT:   *h;
// LDCG-NEXT:   *h2;
// LDCG-NEXT:   *b;
// LDCG-NEXT:   *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ldcs | FileCheck %s -check-prefix=LDCS
// LDCS: CUDA API:
// LDCS-NEXT:   __ldcs(h /*__half **/);
// LDCS-NEXT:   __ldcs(h2 /*__half2 **/);
// LDCS-NEXT:   __ldcs(b /*__nv_bfloat16 **/);
// LDCS-NEXT:   __ldcs(b2 /*__nv_bfloat162 **/);
// LDCS-NEXT: Is migrated to:
// LDCS-NEXT:   *h;
// LDCS-NEXT:   *h2;
// LDCS-NEXT:   *b;
// LDCS-NEXT:   *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ldcv | FileCheck %s -check-prefix=LDCV
// LDCV: CUDA API:
// LDCV-NEXT:   __ldcv(h /*__half **/);
// LDCV-NEXT:   __ldcv(h2 /*__half2 **/);
// LDCV-NEXT:   __ldcv(b /*__nv_bfloat16 **/);
// LDCV-NEXT:   __ldcv(b2 /*__nv_bfloat162 **/);
// LDCV-NEXT: Is migrated to:
// LDCV-NEXT:   *h;
// LDCV-NEXT:   *h2;
// LDCV-NEXT:   *b;
// LDCV-NEXT:   *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ldg | FileCheck %s -check-prefix=LDG
// LDG: CUDA API:
// LDG-NEXT:   __ldg(h /*__half **/);
// LDG-NEXT:   __ldg(h2 /*__half2 **/);
// LDG-NEXT:   __ldg(b /*__nv_bfloat16 **/);
// LDG-NEXT:   __ldg(b2 /*__nv_bfloat162 **/);
// LDG-NEXT: Is migrated to:
// LDG-NEXT:   *h;
// LDG-NEXT:   *h2;
// LDG-NEXT:   *b;
// LDG-NEXT:   *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ldlu | FileCheck %s -check-prefix=LDLU
// LDLU: CUDA API:
// LDLU-NEXT:   __ldlu(h /*__half **/);
// LDLU-NEXT:   __ldlu(h2 /*__half2 **/);
// LDLU-NEXT:   __ldlu(b /*__nv_bfloat16 **/);
// LDLU-NEXT:   __ldlu(b2 /*__nv_bfloat162 **/);
// LDLU-NEXT: Is migrated to:
// LDLU-NEXT:   *h;
// LDLU-NEXT:   *h2;
// LDLU-NEXT:   *b;
// LDLU-NEXT:   *b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__stcg | FileCheck %s -check-prefix=STCG
// STCG: CUDA API:
// STCG-NEXT:   __stcg(ph /*__half **/, h /*__half*/);
// STCG-NEXT:   __stcg(ph2 /*__half2 **/, h2 /*__half2*/);
// STCG-NEXT:   __stcg(pb /*__nv_bfloat16 **/, b /*__nv_bfloat16*/);
// STCG-NEXT:   __stcg(pb2 /*__nv_bfloat162 **/, b2 /*__nv_bfloat162*/);
// STCG-NEXT: Is migrated to:
// STCG-NEXT:   *ph = h;
// STCG-NEXT:   *ph2 = h2;
// STCG-NEXT:   *pb = b;
// STCG-NEXT:   *pb2 = b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__stcs | FileCheck %s -check-prefix=STCS
// STCS: CUDA API:
// STCS-NEXT:   __stcs(ph /*__half **/, h /*__half*/);
// STCS-NEXT:   __stcs(ph2 /*__half2 **/, h2 /*__half2*/);
// STCS-NEXT:   __stcs(pb /*__nv_bfloat16 **/, b /*__nv_bfloat16*/);
// STCS-NEXT:   __stcs(pb2 /*__nv_bfloat162 **/, b2 /*__nv_bfloat162*/);
// STCS-NEXT: Is migrated to:
// STCS-NEXT:   *ph = h;
// STCS-NEXT:   *ph2 = h2;
// STCS-NEXT:   *pb = b;
// STCS-NEXT:   *pb2 = b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__stwb | FileCheck %s -check-prefix=STWB
// STWB: CUDA API:
// STWB-NEXT:   __stwb(ph /*__half **/, h /*__half*/);
// STWB-NEXT:   __stwb(ph2 /*__half2 **/, h2 /*__half2*/);
// STWB-NEXT:   __stwb(pb /*__nv_bfloat16 **/, b /*__nv_bfloat16*/);
// STWB-NEXT:   __stwb(pb2 /*__nv_bfloat162 **/, b2 /*__nv_bfloat162*/);
// STWB-NEXT: Is migrated to:
// STWB-NEXT:   *ph = h;
// STWB-NEXT:   *ph2 = h2;
// STWB-NEXT:   *pb = b;
// STWB-NEXT:   *pb2 = b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__stwt | FileCheck %s -check-prefix=STWT
// STWT: CUDA API:
// STWT-NEXT:   __stwt(ph /*__half **/, h /*__half*/);
// STWT-NEXT:   __stwt(ph2 /*__half2 **/, h2 /*__half2*/);
// STWT-NEXT:   __stwt(pb /*__nv_bfloat16 **/, b /*__nv_bfloat16*/);
// STWT-NEXT:   __stwt(pb2 /*__nv_bfloat162 **/, b2 /*__nv_bfloat162*/);
// STWT-NEXT: Is migrated to:
// STWT-NEXT:   *ph = h;
// STWT-NEXT:   *ph2 = h2;
// STWT-NEXT:   *pb = b;
// STWT-NEXT:   *pb2 = b2;

/// Bfloat16 Precision Conversion And Data Movement

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat1622float2 | FileCheck %s -check-prefix=BFLOAT1622FLOAT2
// BFLOAT1622FLOAT2: CUDA API:
// BFLOAT1622FLOAT2-NEXT:   __bfloat1622float2(b /*__nv_bfloat162*/);
// BFLOAT1622FLOAT2-NEXT: Is migrated to:
// BFLOAT1622FLOAT2-NEXT:   sycl::float2(b[0], b[1]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__bfloat162float | FileCheck %s -check-prefix=BFLOAT162FLOAT
// BFLOAT162FLOAT: CUDA API:
// BFLOAT162FLOAT-NEXT:   __bfloat162float(b /*__nv_bfloat16*/);
// BFLOAT162FLOAT-NEXT: Is migrated to:
// BFLOAT162FLOAT-NEXT:   static_cast<float>(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float22bfloat162_rn | FileCheck %s -check-prefix=FLOAT22BFLOAT162_RN
// FLOAT22BFLOAT162_RN: CUDA API:
// FLOAT22BFLOAT162_RN-NEXT:   __float22bfloat162_rn(f /*float2*/);
// FLOAT22BFLOAT162_RN-NEXT: Is migrated to:
// FLOAT22BFLOAT162_RN-NEXT:   sycl::marray<sycl::ext::oneapi::bfloat16, 2>(f[0], f[1]);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2bfloat16 | FileCheck %s -check-prefix=FLOAT2BFLOAT16
// FLOAT2BFLOAT16: CUDA API:
// FLOAT2BFLOAT16-NEXT:   __float2bfloat16(f /*float*/);
// FLOAT2BFLOAT16-NEXT: Is migrated to:
// FLOAT2BFLOAT16-NEXT:   sycl::ext::oneapi::bfloat16(f);
