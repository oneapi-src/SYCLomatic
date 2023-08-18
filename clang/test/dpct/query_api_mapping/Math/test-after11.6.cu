// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5

/// Half Arithmetic Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hadd_rn | FileCheck %s -check-prefix=HADD_RN
// HADD_RN: CUDA API:
// HADD_RN-NEXT:   __hadd_rn(h1 /*__half*/, h2 /*__half*/);
// HADD_RN-NEXT:   __hadd_rn(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// HADD_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HADD_RN-NEXT:   sycl::ext::intel::math::hadd(h1, h2);
// HADD_RN-NEXT:   b1 + b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmul_rn | FileCheck %s -check-prefix=HMUL_RN
// HMUL_RN: CUDA API:
// HMUL_RN-NEXT:   __hmul_rn(h1 /*__half*/, h2 /*__half*/);
// HMUL_RN-NEXT:   __hmul_rn(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// HMUL_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HMUL_RN-NEXT:   sycl::ext::intel::math::hmul(h1, h2);
// HMUL_RN-NEXT:   b1 * b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hsub_rn | FileCheck %s -check-prefix=HSUB_RN
// HSUB_RN: CUDA API:
// HSUB_RN-NEXT:   __hsub_rn(h1 /*__half*/, h2 /*__half*/);
// HSUB_RN-NEXT:   __hsub_rn(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// HSUB_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HSUB_RN-NEXT:   sycl::ext::intel::math::hsub(h1, h2);
// HSUB_RN-NEXT:   b1 - b2;

/// Half2 Arithmetic Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hadd2_rn | FileCheck %s -check-prefix=HADD2_RN
// HADD2_RN: CUDA API:
// HADD2_RN-NEXT:   __hadd2_rn(h1 /*__half2*/, h2 /*__half2*/);
// HADD2_RN-NEXT:   __hadd2_rn(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// HADD2_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HADD2_RN-NEXT:   sycl::ext::intel::math::hadd2(h1, h2);
// HADD2_RN-NEXT:   b1 + b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmul2_rn | FileCheck %s -check-prefix=HMUL2_RN
// HMUL2_RN: CUDA API:
// HMUL2_RN-NEXT:   __hmul2_rn(h1 /*__half2*/, h2 /*__half2*/);
// HMUL2_RN-NEXT:   __hmul2_rn(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// HMUL2_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HMUL2_RN-NEXT:   sycl::ext::intel::math::hmul2(h1, h2);
// HMUL2_RN-NEXT:   b1 * b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hsub2_rn | FileCheck %s -check-prefix=HSUB2_RN
// HSUB2_RN: CUDA API:
// HSUB2_RN-NEXT:   __hsub2_rn(h1 /*__half2*/, h2 /*__half2*/);
// HSUB2_RN-NEXT:   __hsub2_rn(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// HSUB2_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HSUB2_RN-NEXT:   sycl::ext::intel::math::hsub2(h1, h2);
// HSUB2_RN-NEXT:   b1 - b2;
