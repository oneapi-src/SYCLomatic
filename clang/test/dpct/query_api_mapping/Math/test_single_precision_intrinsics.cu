// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

/// Single Precision Intrinsics

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__cosf | FileCheck %s -check-prefix=__COSF
// __COSF: CUDA API:
// __COSF-NEXT:   __cosf(f /*float*/);
// __COSF-NEXT: Is migrated to:
// __COSF-NEXT:   sycl::cos(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__exp10f | FileCheck %s -check-prefix=__EXP10F
// __EXP10F: CUDA API:
// __EXP10F-NEXT:   __exp10f(f /*float*/);
// __EXP10F-NEXT: Is migrated to:
// __EXP10F-NEXT:   sycl::exp10(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__expf | FileCheck %s -check-prefix=_EXPF
// _EXPF: CUDA API:
// _EXPF-NEXT:   __expf(f /*float*/);
// _EXPF-NEXT: Is migrated to:
// _EXPF-NEXT:   sycl::native::exp(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fadd_rd | FileCheck %s -check-prefix=__FADD_RD
// __FADD_RD: CUDA API:
// __FADD_RD-NEXT:   __fadd_rd(f1 /*float*/, f2 /*float*/);
// __FADD_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FADD_RD-NEXT:   sycl::ext::intel::math::fadd_rd(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fadd_rn | FileCheck %s -check-prefix=__FADD_RN
// __FADD_RN: CUDA API:
// __FADD_RN-NEXT:   __fadd_rn(f1 /*float*/, f2 /*float*/);
// __FADD_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FADD_RN-NEXT:   sycl::ext::intel::math::fadd_rn(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fadd_ru | FileCheck %s -check-prefix=__FADD_RU
// __FADD_RU: CUDA API:
// __FADD_RU-NEXT:   __fadd_ru(f1 /*float*/, f2 /*float*/);
// __FADD_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FADD_RU-NEXT:   sycl::ext::intel::math::fadd_ru(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fadd_rz | FileCheck %s -check-prefix=__FADD_RZ
// __FADD_RZ: CUDA API:
// __FADD_RZ-NEXT:   __fadd_rz(f1 /*float*/, f2 /*float*/);
// __FADD_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FADD_RZ-NEXT:   sycl::ext::intel::math::fadd_rz(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fdiv_rd | FileCheck %s -check-prefix=__FDIV_RD
// __FDIV_RD: CUDA API:
// __FDIV_RD-NEXT:   __fdiv_rd(f1 /*float*/, f2 /*float*/);
// __FDIV_RD-NEXT: Is migrated to:
// __FDIV_RD-NEXT:   f1 / f2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fdiv_rn | FileCheck %s -check-prefix=__FDIV_RN
// __FDIV_RN: CUDA API:
// __FDIV_RN-NEXT:   __fdiv_rn(f1 /*float*/, f2 /*float*/);
// __FDIV_RN-NEXT: Is migrated to:
// __FDIV_RN-NEXT:   f1 / f2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fdiv_ru | FileCheck %s -check-prefix=__FDIV_RU
// __FDIV_RU: CUDA API:
// __FDIV_RU-NEXT:   __fdiv_ru(f1 /*float*/, f2 /*float*/);
// __FDIV_RU-NEXT: Is migrated to:
// __FDIV_RU-NEXT:   f1 / f2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fdiv_rz | FileCheck %s -check-prefix=__FDIV_RZ
// __FDIV_RZ: CUDA API:
// __FDIV_RZ-NEXT:   __fdiv_rz(f1 /*float*/, f2 /*float*/);
// __FDIV_RZ-NEXT: Is migrated to:
// __FDIV_RZ-NEXT:   f1 / f2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fdividef | FileCheck %s -check-prefix=__FDIVIDEF
// __FDIVIDEF: CUDA API:
// __FDIVIDEF-NEXT:   __fdividef(f1 /*float*/, f2 /*float*/);
// __FDIVIDEF-NEXT: Is migrated to:
// __FDIVIDEF-NEXT:   f1 / f2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fmaf_rd | FileCheck %s -check-prefix=__FMAF_RD
// __FMAF_RD: CUDA API:
// __FMAF_RD-NEXT:   __fmaf_rd(f1 /*float*/, f2 /*float*/, f3 /*float*/);
// __FMAF_RD-NEXT: Is migrated to:
// __FMAF_RD-NEXT:   sycl::fma(f1, f2, f3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fmaf_rn | FileCheck %s -check-prefix=__FMAF_RN
// __FMAF_RN: CUDA API:
// __FMAF_RN-NEXT:   __fmaf_rn(f1 /*float*/, f2 /*float*/, f3 /*float*/);
// __FMAF_RN-NEXT: Is migrated to:
// __FMAF_RN-NEXT:   sycl::fma(f1, f2, f3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fmaf_ru | FileCheck %s -check-prefix=__FMAF_RU
// __FMAF_RU: CUDA API:
// __FMAF_RU-NEXT:   __fmaf_ru(f1 /*float*/, f2 /*float*/, f3 /*float*/);
// __FMAF_RU-NEXT: Is migrated to:
// __FMAF_RU-NEXT:   sycl::fma(f1, f2, f3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fmaf_rz | FileCheck %s -check-prefix=__FMAF_RZ
// __FMAF_RZ: CUDA API:
// __FMAF_RZ-NEXT:   __fmaf_rz(f1 /*float*/, f2 /*float*/, f3 /*float*/);
// __FMAF_RZ-NEXT: Is migrated to:
// __FMAF_RZ-NEXT:   sycl::fma(f1, f2, f3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fmul_rd | FileCheck %s -check-prefix=__FMUL_RD
// __FMUL_RD: CUDA API:
// __FMUL_RD-NEXT:   __fmul_rd(f1 /*float*/, f2 /*float*/);
// __FMUL_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FMUL_RD-NEXT:   sycl::ext::intel::math::fmul_rd(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fmul_rn | FileCheck %s -check-prefix=__FMUL_RN
// __FMUL_RN: CUDA API:
// __FMUL_RN-NEXT:   __fmul_rn(f1 /*float*/, f2 /*float*/);
// __FMUL_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FMUL_RN-NEXT:   sycl::ext::intel::math::fmul_rn(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fmul_ru | FileCheck %s -check-prefix=__FMUL_RU
// __FMUL_RU: CUDA API:
// __FMUL_RU-NEXT:   __fmul_ru(f1 /*float*/, f2 /*float*/);
// __FMUL_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FMUL_RU-NEXT:   sycl::ext::intel::math::fmul_ru(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fmul_rz | FileCheck %s -check-prefix=__FMUL_RZ
// __FMUL_RZ: CUDA API:
// __FMUL_RZ-NEXT:   __fmul_rz(f1 /*float*/, f2 /*float*/);
// __FMUL_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FMUL_RZ-NEXT:   sycl::ext::intel::math::fmul_rz(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__frcp_rd | FileCheck %s -check-prefix=__FRCP_RD
// __FRCP_RD: CUDA API:
// __FRCP_RD-NEXT:   __frcp_rd(f /*float*/);
// __FRCP_RD-NEXT: Is migrated to:
// __FRCP_RD-NEXT:   sycl::native::recip(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__frcp_rn | FileCheck %s -check-prefix=__FRCP_RN
// __FRCP_RN: CUDA API:
// __FRCP_RN-NEXT:   __frcp_rn(f /*float*/);
// __FRCP_RN-NEXT: Is migrated to:
// __FRCP_RN-NEXT:   sycl::native::recip(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__frcp_ru | FileCheck %s -check-prefix=__FRCP_RU
// __FRCP_RU: CUDA API:
// __FRCP_RU-NEXT:   __frcp_ru(f /*float*/);
// __FRCP_RU-NEXT: Is migrated to:
// __FRCP_RU-NEXT:   sycl::native::recip(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__frcp_rz | FileCheck %s -check-prefix=__FRCP_RZ
// __FRCP_RZ: CUDA API:
// __FRCP_RZ-NEXT:   __frcp_rz(f /*float*/);
// __FRCP_RZ-NEXT: Is migrated to:
// __FRCP_RZ-NEXT:   sycl::native::recip(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fsqrt_rd | FileCheck %s -check-prefix=__FSQRT_RD
// __FSQRT_RD: CUDA API:
// __FSQRT_RD-NEXT:   __fsqrt_rd(f /*float*/);
// __FSQRT_RD-NEXT: Is migrated to:
// __FSQRT_RD-NEXT:   sycl::sqrt(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__frsqrt_rn | FileCheck %s -check-prefix=__FRSQRT_RN
// __FRSQRT_RN: CUDA API:
// __FRSQRT_RN-NEXT:   __frsqrt_rn(f /*float*/);
// __FRSQRT_RN-NEXT: Is migrated to:
// __FRSQRT_RN-NEXT:   sycl::rsqrt(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fsqrt_rn | FileCheck %s -check-prefix=__FSQRT_RN
// __FSQRT_RN: CUDA API:
// __FSQRT_RN-NEXT:   __fsqrt_rn(f /*float*/);
// __FSQRT_RN-NEXT: Is migrated to:
// __FSQRT_RN-NEXT:   sycl::sqrt(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fsqrt_ru | FileCheck %s -check-prefix=__FSQRT_RU
// __FSQRT_RU: CUDA API:
// __FSQRT_RU-NEXT:   __fsqrt_ru(f /*float*/);
// __FSQRT_RU-NEXT: Is migrated to:
// __FSQRT_RU-NEXT:   sycl::sqrt(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fsqrt_rz | FileCheck %s -check-prefix=__FSQRT_RZ
// __FSQRT_RZ: CUDA API:
// __FSQRT_RZ-NEXT:   __fsqrt_rz(f /*float*/);
// __FSQRT_RZ-NEXT: Is migrated to:
// __FSQRT_RZ-NEXT:   sycl::sqrt(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fsub_rd | FileCheck %s -check-prefix=__FSUB_RD
// __FSUB_RD: CUDA API:
// __FSUB_RD-NEXT:   __fsub_rd(f1 /*float*/, f2 /*float*/);
// __FSUB_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FSUB_RD-NEXT:   sycl::ext::intel::math::fsub_rd(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fsub_rn | FileCheck %s -check-prefix=__FSUB_RN
// __FSUB_RN: CUDA API:
// __FSUB_RN-NEXT:   __fsub_rn(f1 /*float*/, f2 /*float*/);
// __FSUB_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FSUB_RN-NEXT:   sycl::ext::intel::math::fsub_rn(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fsub_ru | FileCheck %s -check-prefix=__FSUB_RU
// __FSUB_RU: CUDA API:
// __FSUB_RU-NEXT:   __fsub_ru(f1 /*float*/, f2 /*float*/);
// __FSUB_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FSUB_RU-NEXT:   sycl::ext::intel::math::fsub_ru(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fsub_rz | FileCheck %s -check-prefix=__FSUB_RZ
// __FSUB_RZ: CUDA API:
// __FSUB_RZ-NEXT:   __fsub_rz(f1 /*float*/, f2 /*float*/);
// __FSUB_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __FSUB_RZ-NEXT:   sycl::ext::intel::math::fsub_rz(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__log10f | FileCheck %s -check-prefix=__LOG10F
// __LOG10F: CUDA API:
// __LOG10F-NEXT:   __log10f(f /*float*/);
// __LOG10F-NEXT: Is migrated to:
// __LOG10F-NEXT:   sycl::log10(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__log2f | FileCheck %s -check-prefix=__LOG2F
// __LOG2F: CUDA API:
// __LOG2F-NEXT:   __log2f(f /*float*/);
// __LOG2F-NEXT: Is migrated to:
// __LOG2F-NEXT:   sycl::log2(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__logf | FileCheck %s -check-prefix=__LOGF
// __LOGF: CUDA API:
// __LOGF-NEXT:   __logf(f /*float*/);
// __LOGF-NEXT: Is migrated to:
// __LOGF-NEXT:   sycl::log(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__powf | FileCheck %s -check-prefix=__POWF
// __POWF: CUDA API:
// __POWF-NEXT:   __powf(f1 /*float*/, f2 /*float*/);
// __POWF-NEXT: Is migrated to:
// __POWF-NEXT:   dpct::pow(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__saturatef | FileCheck %s -check-prefix=__SATURATEF
// __SATURATEF: CUDA API:
// __SATURATEF-NEXT:   __saturatef(f /*float*/);
// __SATURATEF-NEXT: Is migrated to:
// __SATURATEF-NEXT:   sycl::clamp<float>(f, 0.0f, 1.0f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__sincosf | FileCheck %s -check-prefix=__SINCOSF
// __SINCOSF: CUDA API:
// __SINCOSF-NEXT:   __sincosf(f /*float*/, pf1 /*float **/, pf2 /*float **/);
// __SINCOSF-NEXT: Is migrated to:
// __SINCOSF-NEXT:   *pf1 = sycl::sincos(f, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes, float>(pf2));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__sinf | FileCheck %s -check-prefix=__SINF
// __SINF: CUDA API:
// __SINF-NEXT:   __sinf(f /*float*/);
// __SINF-NEXT: Is migrated to:
// __SINF-NEXT:   sycl::sin(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__tanf | FileCheck %s -check-prefix=__TANF
// __TANF: CUDA API:
// __TANF-NEXT:   __tanf(f /*float*/);
// __TANF-NEXT: Is migrated to:
// __TANF-NEXT:   sycl::tan(f);
