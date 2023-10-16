// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

/// Half Comparison Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__heq | FileCheck %s -check-prefix=__HEQ
// __HEQ: CUDA API:
// __HEQ-NEXT:   __heq(h1 /*__half*/, h2 /*__half*/);
// __HEQ-NEXT:   __heq(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HEQ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HEQ-NEXT:   sycl::ext::intel::math::heq(h1, h2);
// __HEQ-NEXT:   b1 == b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hequ | FileCheck %s -check-prefix=__HEQU
// __HEQU: CUDA API:
// __HEQU-NEXT:   __hequ(h1 /*__half*/, h2 /*__half*/);
// __HEQU-NEXT:   __hequ(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HEQU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HEQU-NEXT:   sycl::ext::intel::math::hequ(h1, h2);
// __HEQU-NEXT:   dpct::unordered_compare(b1, b2, std::equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hge | FileCheck %s -check-prefix=__HGE
// __HGE: CUDA API:
// __HGE-NEXT:   __hge(h1 /*__half*/, h2 /*__half*/);
// __HGE-NEXT:   __hge(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HGE-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HGE-NEXT:   sycl::ext::intel::math::hge(h1, h2);
// __HGE-NEXT:   b1 >= b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hgeu | FileCheck %s -check-prefix=__HGEU
// __HGEU: CUDA API:
// __HGEU-NEXT:   __hgeu(h1 /*__half*/, h2 /*__half*/);
// __HGEU-NEXT:   __hgeu(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HGEU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HGEU-NEXT:   sycl::ext::intel::math::hgeu(h1, h2);
// __HGEU-NEXT:   dpct::unordered_compare(b1, b2, std::greater_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hgt | FileCheck %s -check-prefix=__HGT
// __HGT: CUDA API:
// __HGT-NEXT:   __hgt(h1 /*__half*/, h2 /*__half*/);
// __HGT-NEXT:   __hgt(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HGT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HGT-NEXT:   sycl::ext::intel::math::hgt(h1, h2);
// __HGT-NEXT:   b1 > b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hgtu | FileCheck %s -check-prefix=__HGTU
// __HGTU: CUDA API:
// __HGTU-NEXT:   __hgtu(h1 /*__half*/, h2 /*__half*/);
// __HGTU-NEXT:   __hgtu(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HGTU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HGTU-NEXT:   sycl::ext::intel::math::hgtu(h1, h2);
// __HGTU-NEXT:   dpct::unordered_compare(b1, b2, std::greater<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hisinf | FileCheck %s -check-prefix=__HISINF
// __HISINF: CUDA API:
// __HISINF-NEXT:   __hisinf(h /*__half*/);
// __HISINF-NEXT:   __hisinf(b /*__nv_bfloat16*/);
// __HISINF-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HISINF-NEXT:   sycl::ext::intel::math::hisinf(h);
// __HISINF-NEXT:   sycl::isinf(float(b));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hisnan | FileCheck %s -check-prefix=__HISNAN
// __HISNAN: CUDA API:
// __HISNAN-NEXT:   __hisnan(h /*__half*/);
// __HISNAN-NEXT:   __hisnan(b /*__nv_bfloat16*/);
// __HISNAN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// __HISNAN-NEXT:   sycl::ext::intel::math::hisnan(h);
// __HISNAN-NEXT:   sycl::ext::oneapi::experimental::isnan(b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hle | FileCheck %s -check-prefix=__HLE
// __HLE: CUDA API:
// __HLE-NEXT:   __hle(h1 /*__half*/, h2 /*__half*/);
// __HLE-NEXT:   __hle(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HLE-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HLE-NEXT:   sycl::ext::intel::math::hle(h1, h2);
// __HLE-NEXT:   b1 <= b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hleu | FileCheck %s -check-prefix=__HLEU
// __HLEU: CUDA API:
// __HLEU-NEXT:   __hleu(h1 /*__half*/, h2 /*__half*/);
// __HLEU-NEXT:   __hleu(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HLEU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HLEU-NEXT:   sycl::ext::intel::math::hleu(h1, h2);
// __HLEU-NEXT:   dpct::unordered_compare(b1, b2, std::less_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hlt | FileCheck %s -check-prefix=__HLT
// __HLT: CUDA API:
// __HLT-NEXT:   __hlt(h1 /*__half*/, h2 /*__half*/);
// __HLT-NEXT:   __hlt(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HLT-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HLT-NEXT:   sycl::ext::intel::math::hlt(h1, h2);
// __HLT-NEXT:   b1 < b2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hltu | FileCheck %s -check-prefix=__HLTU
// __HLTU: CUDA API:
// __HLTU-NEXT:   __hltu(h1 /*__half*/, h2 /*__half*/);
// __HLTU-NEXT:   __hltu(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HLTU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HLTU-NEXT:   sycl::ext::intel::math::hltu(h1, h2);
// __HLTU-NEXT:   dpct::unordered_compare(b1, b2, std::less<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmax | FileCheck %s -check-prefix=__HMAX
// __HMAX: CUDA API:
// __HMAX-NEXT:   __hmax(h1 /*__half*/, h2 /*__half*/);
// __HMAX-NEXT:   __hmax(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HMAX-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// __HMAX-NEXT:   sycl::ext::intel::math::hmax(h1, h2);
// __HMAX-NEXT:   sycl::ext::oneapi::experimental::fmax(b1, b2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmax_nan | FileCheck %s -check-prefix=__HMAX_NAN
// __HMAX_NAN: CUDA API:
// __HMAX_NAN-NEXT:   __hmax_nan(h1 /*__half*/, h2 /*__half*/);
// __HMAX_NAN-NEXT:   __hmax_nan(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HMAX_NAN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HMAX_NAN-NEXT:   sycl::ext::intel::math::hmax_nan(h1, h2);
// __HMAX_NAN-NEXT:   dpct::fmax_nan(b1, b2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmin | FileCheck %s -check-prefix=__HMIN
// __HMIN: CUDA API:
// __HMIN-NEXT:   __hmin(h1 /*__half*/, h2 /*__half*/);
// __HMIN-NEXT:   __hmin(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HMIN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math --use-experimental-features=bfloat16_math_functions):
// __HMIN-NEXT:   sycl::ext::intel::math::hmin(h1, h2);
// __HMIN-NEXT:   sycl::ext::oneapi::experimental::fmin(b1, b2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hmin_nan | FileCheck %s -check-prefix=__HMIN_NAN
// __HMIN_NAN: CUDA API:
// __HMIN_NAN-NEXT:   __hmin_nan(h1 /*__half*/, h2 /*__half*/);
// __HMIN_NAN-NEXT:   __hmin_nan(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HMIN_NAN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HMIN_NAN-NEXT:   sycl::ext::intel::math::hmin_nan(h1, h2);
// __HMIN_NAN-NEXT:   dpct::fmin_nan(b1, b2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hne | FileCheck %s -check-prefix=__HNE
// __HNE: CUDA API:
// __HNE-NEXT:   __hne(h1 /*__half*/, h2 /*__half*/);
// __HNE-NEXT:   __hne(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HNE-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HNE-NEXT:   sycl::ext::intel::math::hne(h1, h2);
// __HNE-NEXT:   dpct::compare(b1, b2, std::not_equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hneu | FileCheck %s -check-prefix=__HNEU
// __HNEU: CUDA API:
// __HNEU-NEXT:   __hneu(h1 /*__half*/, h2 /*__half*/);
// __HNEU-NEXT:   __hneu(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/);
// __HNEU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __HNEU-NEXT:   sycl::ext::intel::math::hneu(h1, h2);
// __HNEU-NEXT:   dpct::unordered_compare(b1, b2, std::not_equal_to<>());
