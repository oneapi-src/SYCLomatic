// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

/// Half2 Comparison Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hbeq2 | FileCheck %s -check-prefix=__HBEQ2
// __HBEQ2: CUDA API:
// __HBEQ2-NEXT:   __hbeq2(h1 /*__half2*/, h2 /*__half2*/);
// __HBEQ2-NEXT:   __hbeq2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBEQ2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBEQ2-NEXT:   sycl::ext::intel::math::hbeq2(h1, h2);
// __HBEQ2-NEXT:   dpct::compare_both(b1, b2, std::equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hbequ2 | FileCheck %s -check-prefix=__HBEQU2
// __HBEQU2: CUDA API:
// __HBEQU2-NEXT:   __hbequ2(h1 /*__half2*/, h2 /*__half2*/);
// __HBEQU2-NEXT:   __hbequ2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBEQU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBEQU2-NEXT:   sycl::ext::intel::math::hbequ2(h1, h2);
// __HBEQU2-NEXT:   dpct::unordered_compare_both(b1, b2, std::equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hbge2 | FileCheck %s -check-prefix=__HBGE2
// __HBGE2: CUDA API:
// __HBGE2-NEXT:   __hbge2(h1 /*__half2*/, h2 /*__half2*/);
// __HBGE2-NEXT:   __hbge2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBGE2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBGE2-NEXT:   sycl::ext::intel::math::hbge2(h1, h2);
// __HBGE2-NEXT:   dpct::compare_both(b1, b2, std::greater_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hbgeu2 | FileCheck %s -check-prefix=__HBGEU2
// __HBGEU2: CUDA API:
// __HBGEU2-NEXT:   __hbgeu2(h1 /*__half2*/, h2 /*__half2*/);
// __HBGEU2-NEXT:   __hbgeu2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBGEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBGEU2-NEXT:   sycl::ext::intel::math::hbgeu2(h1, h2);
// __HBGEU2-NEXT:   dpct::unordered_compare_both(b1, b2, std::greater_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hbgt2 | FileCheck %s -check-prefix=__HBGT2
// __HBGT2: CUDA API:
// __HBGT2-NEXT:   __hbgt2(h1 /*__half2*/, h2 /*__half2*/);
// __HBGT2-NEXT:   __hbgt2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBGT2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBGT2-NEXT:   sycl::ext::intel::math::hbgt2(h1, h2);
// __HBGT2-NEXT:   dpct::compare_both(b1, b2, std::greater<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hbgtu2 | FileCheck %s -check-prefix=__HBGTU2
// __HBGTU2: CUDA API:
// __HBGTU2-NEXT:   __hbgtu2(h1 /*__half2*/, h2 /*__half2*/);
// __HBGTU2-NEXT:   __hbgtu2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBGTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBGTU2-NEXT:   sycl::ext::intel::math::hbgtu2(h1, h2);
// __HBGTU2-NEXT:   dpct::unordered_compare_both(b1, b2, std::greater<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hble2 | FileCheck %s -check-prefix=__HBLE2
// __HBLE2: CUDA API:
// __HBLE2-NEXT:   __hble2(h1 /*__half2*/, h2 /*__half2*/);
// __HBLE2-NEXT:   __hble2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBLE2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBLE2-NEXT:   sycl::ext::intel::math::hble2(h1, h2);
// __HBLE2-NEXT:   dpct::compare_both(b1, b2, std::less_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hbleu2 | FileCheck %s -check-prefix=__HBLEU2
// __HBLEU2: CUDA API:
// __HBLEU2-NEXT:   __hbleu2(h1 /*__half2*/, h2 /*__half2*/);
// __HBLEU2-NEXT:   __hbleu2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBLEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBLEU2-NEXT:   sycl::ext::intel::math::hbleu2(h1, h2);
// __HBLEU2-NEXT:   dpct::unordered_compare_both(b1, b2, std::less_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hblt2 | FileCheck %s -check-prefix=__HBLT2
// __HBLT2: CUDA API:
// __HBLT2-NEXT:   __hblt2(h1 /*__half2*/, h2 /*__half2*/);
// __HBLT2-NEXT:   __hblt2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBLT2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBLT2-NEXT:   sycl::ext::intel::math::hblt2(h1, h2);
// __HBLT2-NEXT:   dpct::compare_both(b1, b2, std::less<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hbltu2 | FileCheck %s -check-prefix=__HBLTU2
// __HBLTU2: CUDA API:
// __HBLTU2-NEXT:   __hbltu2(h1 /*__half2*/, h2 /*__half2*/);
// __HBLTU2-NEXT:   __hbltu2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBLTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBLTU2-NEXT:   sycl::ext::intel::math::hbltu2(h1, h2);
// __HBLTU2-NEXT:   dpct::unordered_compare_both(b1, b2, std::less<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hbne2 | FileCheck %s -check-prefix=__HBNE2
// __HBNE2: CUDA API:
// __HBNE2-NEXT:   __hbne2(h1 /*__half2*/, h2 /*__half2*/);
// __HBNE2-NEXT:   __hbne2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBNE2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBNE2-NEXT:   sycl::ext::intel::math::hbne2(h1, h2);
// __HBNE2-NEXT:   dpct::compare_both(b1, b2, std::not_equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hbneu2 | FileCheck %s -check-prefix=__HBNEU2
// __HBNEU2: CUDA API:
// __HBNEU2-NEXT:   __hbneu2(h1 /*__half2*/, h2 /*__half2*/);
// __HBNEU2-NEXT:   __hbneu2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HBNEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HBNEU2-NEXT:   sycl::ext::intel::math::hbneu2(h1, h2);
// __HBNEU2-NEXT:   dpct::unordered_compare_both(b1, b2, std::not_equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__heq2 | FileCheck %s -check-prefix=__HEQ2
// __HEQ2: CUDA API:
// __HEQ2-NEXT:   __heq2(h1 /*__half2*/, h2 /*__half2*/);
// __HEQ2-NEXT:   __heq2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HEQ2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HEQ2-NEXT:   sycl::ext::intel::math::heq2(h1, h2);
// __HEQ2-NEXT:   dpct::compare(b1, b2, std::equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hequ2 | FileCheck %s -check-prefix=__HEQU2
// __HEQU2: CUDA API:
// __HEQU2-NEXT:   __hequ2(h1 /*__half2*/, h2 /*__half2*/);
// __HEQU2-NEXT:   __hequ2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HEQU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HEQU2-NEXT:   sycl::ext::intel::math::hequ2(h1, h2);
// __HEQU2-NEXT:   dpct::unordered_compare(b1, b2, std::equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hge2 | FileCheck %s -check-prefix=__HGE2
// __HGE2: CUDA API:
// __HGE2-NEXT:   __hge2(h1 /*__half2*/, h2 /*__half2*/);
// __HGE2-NEXT:   __hge2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HGE2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HGE2-NEXT:   sycl::ext::intel::math::hge2(h1, h2);
// __HGE2-NEXT:   dpct::compare(b1, b2, std::greater_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hgeu2 | FileCheck %s -check-prefix=__HGEU2
// __HGEU2: CUDA API:
// __HGEU2-NEXT:   __hgeu2(h1 /*__half2*/, h2 /*__half2*/);
// __HGEU2-NEXT:   __hgeu2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HGEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HGEU2-NEXT:   sycl::ext::intel::math::hgeu2(h1, h2);
// __HGEU2-NEXT:   dpct::unordered_compare(b1, b2, std::greater_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hgt2 | FileCheck %s -check-prefix=__HGT2
// __HGT2: CUDA API:
// __HGT2-NEXT:   __hgt2(h1 /*__half2*/, h2 /*__half2*/);
// __HGT2-NEXT:   __hgt2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HGT2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HGT2-NEXT:   sycl::ext::intel::math::hgt2(h1, h2);
// __HGT2-NEXT:   dpct::compare(b1, b2, std::greater<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hgtu2 | FileCheck %s -check-prefix=__HGTU2
// __HGTU2: CUDA API:
// __HGTU2-NEXT:   __hgtu2(h1 /*__half2*/, h2 /*__half2*/);
// __HGTU2-NEXT:   __hgtu2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HGTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HGTU2-NEXT:   sycl::ext::intel::math::hgtu2(h1, h2);
// __HGTU2-NEXT:   dpct::unordered_compare(b1, b2, std::greater<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hisnan2 | FileCheck %s -check-prefix=__HISNAN2
// __HISNAN2: CUDA API:
// __HISNAN2-NEXT:   __hisnan2(h /*__half2*/);
// __HISNAN2-NEXT:   __hisnan2(b /*__nv_bfloat162*/);
// __HISNAN2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// __HISNAN2-NEXT:   sycl::ext::intel::math::hisnan2(h);
// __HISNAN2-NEXT:   sycl::ext::oneapi::experimental::isnan(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hle2 | FileCheck %s -check-prefix=__HLE2
// __HLE2: CUDA API:
// __HLE2-NEXT:   __hle2(h1 /*__half2*/, h2 /*__half2*/);
// __HLE2-NEXT:   __hle2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HLE2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HLE2-NEXT:   sycl::ext::intel::math::hle2(h1, h2);
// __HLE2-NEXT:   dpct::compare(b1, b2, std::less_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hleu2 | FileCheck %s -check-prefix=__HLEU2
// __HLEU2: CUDA API:
// __HLEU2-NEXT:   __hleu2(h1 /*__half2*/, h2 /*__half2*/);
// __HLEU2-NEXT:   __hleu2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HLEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HLEU2-NEXT:   sycl::ext::intel::math::hleu2(h1, h2);
// __HLEU2-NEXT:   dpct::unordered_compare(b1, b2, std::less_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hlt2 | FileCheck %s -check-prefix=__HLT2
// __HLT2: CUDA API:
// __HLT2-NEXT:   __hlt2(h1 /*__half2*/, h2 /*__half2*/);
// __HLT2-NEXT:   __hlt2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HLT2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HLT2-NEXT:   sycl::ext::intel::math::hlt2(h1, h2);
// __HLT2-NEXT:   dpct::compare(b1, b2, std::less<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hltu2 | FileCheck %s -check-prefix=__HLTU2
// __HLTU2: CUDA API:
// __HLTU2-NEXT:   __hltu2(h1 /*__half2*/, h2 /*__half2*/);
// __HLTU2-NEXT:   __hltu2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HLTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HLTU2-NEXT:   sycl::ext::intel::math::hltu2(h1, h2);
// __HLTU2-NEXT:   dpct::unordered_compare(b1, b2, std::less<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmax2 | FileCheck %s -check-prefix=__HMAX2
// __HMAX2: CUDA API:
// __HMAX2-NEXT:   __hmax2(h1 /*__half2*/, h2 /*__half2*/);
// __HMAX2-NEXT:   __hmax2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HMAX2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// __HMAX2-NEXT:   sycl::ext::intel::math::hmax2(h1, h2);
// __HMAX2-NEXT:   sycl::ext::oneapi::experimental::fmax(b1, b2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmax2_nan | FileCheck %s -check-prefix=__HMAX2_NAN
// __HMAX2_NAN: CUDA API:
// __HMAX2_NAN-NEXT:   __hmax2_nan(h1 /*__half2*/, h2 /*__half2*/);
// __HMAX2_NAN-NEXT:   __hmax2_nan(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HMAX2_NAN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HMAX2_NAN-NEXT:   sycl::ext::intel::math::hmax2_nan(h1, h2);
// __HMAX2_NAN-NEXT:   dpct::fmax_nan(b1, b2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmin2 | FileCheck %s -check-prefix=__HMIN2
// __HMIN2: CUDA API:
// __HMIN2-NEXT:   __hmin2(h1 /*__half2*/, h2 /*__half2*/);
// __HMIN2-NEXT:   __hmin2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HMIN2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// __HMIN2-NEXT:   sycl::ext::intel::math::hmin2(h1, h2);
// __HMIN2-NEXT:   sycl::ext::oneapi::experimental::fmin(b1, b2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmin2_nan | FileCheck %s -check-prefix=__HMIN2_NAN
// __HMIN2_NAN: CUDA API:
// __HMIN2_NAN-NEXT:   __hmin2_nan(h1 /*__half2*/, h2 /*__half2*/);
// __HMIN2_NAN-NEXT:   __hmin2_nan(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HMIN2_NAN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HMIN2_NAN-NEXT:   sycl::ext::intel::math::hmin2_nan(h1, h2);
// __HMIN2_NAN-NEXT:   dpct::fmin_nan(b1, b2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hne2 | FileCheck %s -check-prefix=__HNE2
// __HNE2: CUDA API:
// __HNE2-NEXT:   __hne2(h1 /*__half2*/, h2 /*__half2*/);
// __HNE2-NEXT:   __hne2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HNE2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HNE2-NEXT:   sycl::ext::intel::math::hne2(h1, h2);
// __HNE2-NEXT:   dpct::compare(b1, b2, std::not_equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hneu2 | FileCheck %s -check-prefix=__HNEU2
// __HNEU2: CUDA API:
// __HNEU2-NEXT:   __hneu2(h1 /*__half2*/, h2 /*__half2*/);
// __HNEU2-NEXT:   __hneu2(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HNEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HNEU2-NEXT:   sycl::ext::intel::math::hneu2(h1, h2);
// __HNEU2-NEXT:   dpct::unordered_compare(b1, b2, std::not_equal_to<>());
