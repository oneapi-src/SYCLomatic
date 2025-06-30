// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

/// Half2 Comparison Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__heq2_mask | FileCheck %s -check-prefix=__HEQ2_MASK
// __HEQ2_MASK: CUDA API:
// __HEQ2_MASK-NEXT:   __heq2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HEQ2_MASK-NEXT:   __heq2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HEQ2_MASK-NEXT: Is migrated to:
// __HEQ2_MASK-NEXT:   dpct::compare_mask(h1, h2, std::equal_to<>());
// __HEQ2_MASK-NEXT:   dpct::compare_mask(b1, b2, std::equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hequ2_mask | FileCheck %s -check-prefix=__HEQU2_MASK
// __HEQU2_MASK: CUDA API:
// __HEQU2_MASK-NEXT:   __hequ2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HEQU2_MASK-NEXT:   __hequ2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HEQU2_MASK-NEXT: Is migrated to:
// __HEQU2_MASK-NEXT:   dpct::unordered_compare_mask(h1, h2, std::equal_to<>());
// __HEQU2_MASK-NEXT:   dpct::unordered_compare_mask(b1, b2, std::equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hge2_mask | FileCheck %s -check-prefix=__HGE2_MASK
// __HGE2_MASK: CUDA API:
// __HGE2_MASK-NEXT:   __hge2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HGE2_MASK-NEXT:   __hge2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HGE2_MASK-NEXT: Is migrated to:
// __HGE2_MASK-NEXT:   dpct::compare_mask(h1, h2, std::greater_equal<>());
// __HGE2_MASK-NEXT:   dpct::compare_mask(b1, b2, std::greater_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hgeu2_mask | FileCheck %s -check-prefix=__HGEU2_MASK
// __HGEU2_MASK: CUDA API:
// __HGEU2_MASK-NEXT:   __hgeu2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HGEU2_MASK-NEXT:   __hgeu2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HGEU2_MASK-NEXT: Is migrated to:
// __HGEU2_MASK-NEXT:   dpct::unordered_compare_mask(h1, h2, std::greater_equal<>());
// __HGEU2_MASK-NEXT:   dpct::unordered_compare_mask(b1, b2, std::greater_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hgt2_mask | FileCheck %s -check-prefix=__HGT2_MASK
// __HGT2_MASK: CUDA API:
// __HGT2_MASK-NEXT:   __hgt2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HGT2_MASK-NEXT:   __hgt2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HGT2_MASK-NEXT: Is migrated to:
// __HGT2_MASK-NEXT:   dpct::compare_mask(h1, h2, std::greater<>());
// __HGT2_MASK-NEXT:   dpct::compare_mask(b1, b2, std::greater<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hgtu2_mask | FileCheck %s -check-prefix=__HGTU2_MASK
// __HGTU2_MASK: CUDA API:
// __HGTU2_MASK-NEXT:   __hgtu2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HGTU2_MASK-NEXT:   __hgtu2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HGTU2_MASK-NEXT: Is migrated to:
// __HGTU2_MASK-NEXT:   dpct::unordered_compare_mask(h1, h2, std::greater<>());
// __HGTU2_MASK-NEXT:   dpct::unordered_compare_mask(b1, b2, std::greater<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hle2_mask | FileCheck %s -check-prefix=__HLE2_MASK
// __HLE2_MASK: CUDA API:
// __HLE2_MASK-NEXT:   __hle2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HLE2_MASK-NEXT:   __hle2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HLE2_MASK-NEXT: Is migrated to:
// __HLE2_MASK-NEXT:   dpct::compare_mask(h1, h2, std::less_equal<>());
// __HLE2_MASK-NEXT:   dpct::compare_mask(b1, b2, std::less_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hleu2_mask | FileCheck %s -check-prefix=__HLEU2_MASK
// __HLEU2_MASK: CUDA API:
// __HLEU2_MASK-NEXT:   __hleu2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HLEU2_MASK-NEXT:   __hleu2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HLEU2_MASK-NEXT: Is migrated to:
// __HLEU2_MASK-NEXT:   dpct::unordered_compare_mask(h1, h2, std::less_equal<>());
// __HLEU2_MASK-NEXT:   dpct::unordered_compare_mask(b1, b2, std::less_equal<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hlt2_mask | FileCheck %s -check-prefix=__HLT2_MASK
// __HLT2_MASK: CUDA API:
// __HLT2_MASK-NEXT:   __hlt2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HLT2_MASK-NEXT:   __hlt2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HLT2_MASK-NEXT: Is migrated to:
// __HLT2_MASK-NEXT:   dpct::compare_mask(h1, h2, std::less<>());
// __HLT2_MASK-NEXT:   dpct::compare_mask(b1, b2, std::less<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hltu2_mask | FileCheck %s -check-prefix=__HLTU2_MASK
// __HLTU2_MASK: CUDA API:
// __HLTU2_MASK-NEXT:   __hltu2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HLTU2_MASK-NEXT:   __hltu2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HLTU2_MASK-NEXT: Is migrated to:
// __HLTU2_MASK-NEXT:   dpct::unordered_compare_mask(h1, h2, std::less<>());
// __HLTU2_MASK-NEXT:   dpct::unordered_compare_mask(b1, b2, std::less<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hne2_mask | FileCheck %s -check-prefix=__HNE2_MASK
// __HNE2_MASK: CUDA API:
// __HNE2_MASK-NEXT:   __hne2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HNE2_MASK-NEXT:   __hne2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HNE2_MASK-NEXT: Is migrated to:
// __HNE2_MASK-NEXT:   dpct::compare_mask(h1, h2, std::not_equal_to<>());
// __HNE2_MASK-NEXT:   dpct::compare_mask(b1, b2, std::not_equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hneu2_mask | FileCheck %s -check-prefix=__HNEU2_MASK
// __HNEU2_MASK: CUDA API:
// __HNEU2_MASK-NEXT:   __hneu2_mask(h1 /*__half2*/, h2 /*__half2*/);
// __HNEU2_MASK-NEXT:   __hneu2_mask(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/);
// __HNEU2_MASK-NEXT: Is migrated to:
// __HNEU2_MASK-NEXT:   dpct::unordered_compare_mask(h1, h2, std::not_equal_to<>());
// __HNEU2_MASK-NEXT:   dpct::unordered_compare_mask(b1, b2, std::not_equal_to<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmax_s16x2 | FileCheck %s -check-prefix=__viaddmax_s16x2
// __viaddmax_s16x2: CUDA API:
// __viaddmax_s16x2-NEXT:   __viaddmax_s16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __viaddmax_s16x2-NEXT:                    c /*const unsigned int*/);
// __viaddmax_s16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmax_s16x2-NEXT:   sycl::ext::intel::math::viaddmax_s16x2<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmax_s16x2_relu | FileCheck %s -check-prefix=__viaddmax_s16x2_relu
// __viaddmax_s16x2_relu: CUDA API:
// __viaddmax_s16x2_relu-NEXT:   __viaddmax_s16x2_relu(a /*const unsigned int*/, b /*const unsigned int*/,
// __viaddmax_s16x2_relu-NEXT:                         c /*const unsigned int*/);
// __viaddmax_s16x2_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmax_s16x2_relu-NEXT:   sycl::ext::intel::math::viaddmax_s16x2_relu<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmax_s32 | FileCheck %s -check-prefix=__viaddmax_s32
// __viaddmax_s32: CUDA API:
// __viaddmax_s32-NEXT:   __viaddmax_s32(a /*const int*/, b /*const int*/, c /*const int*/);
// __viaddmax_s32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmax_s32-NEXT:   sycl::ext::intel::math::viaddmax_s32<int>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmax_s32_relu | FileCheck %s -check-prefix=__viaddmax_s32_relu
// __viaddmax_s32_relu: CUDA API:
// __viaddmax_s32_relu-NEXT:   __viaddmax_s32_relu(a /*const int*/, b /*const int*/, c /*const int*/);
// __viaddmax_s32_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmax_s32_relu-NEXT:   sycl::ext::intel::math::viaddmax_s32_relu<int>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmax_u16x2 | FileCheck %s -check-prefix=__viaddmax_u16x2
// __viaddmax_u16x2: CUDA API:
// __viaddmax_u16x2-NEXT:   __viaddmax_u16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __viaddmax_u16x2-NEXT:                    c /*const unsigned int*/);
// __viaddmax_u16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmax_u16x2-NEXT:   sycl::ext::intel::math::viaddmax_u16x2<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmax_u32 | FileCheck %s -check-prefix=__viaddmax_u32
// __viaddmax_u32: CUDA API:
// __viaddmax_u32-NEXT:   __viaddmax_u32(a /*const unsigned int*/, b /*const unsigned int*/,
// __viaddmax_u32-NEXT:                  c /*const unsigned int*/);
// __viaddmax_u32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmax_u32-NEXT:   sycl::ext::intel::math::viaddmax_u32<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmin_s16x2 | FileCheck %s -check-prefix=__viaddmin_s16x2
// __viaddmin_s16x2: CUDA API:
// __viaddmin_s16x2-NEXT:   __viaddmin_s16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __viaddmin_s16x2-NEXT:                    c /*const unsigned int*/);
// __viaddmin_s16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmin_s16x2-NEXT:   sycl::ext::intel::math::viaddmin_s16x2<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmin_s16x2_relu | FileCheck %s -check-prefix=__viaddmin_s16x2_relu
// __viaddmin_s16x2_relu: CUDA API:
// __viaddmin_s16x2_relu-NEXT:   __viaddmin_s16x2_relu(a /*const unsigned int*/, b /*const unsigned int*/,
// __viaddmin_s16x2_relu-NEXT:                         c /*const unsigned int*/);
// __viaddmin_s16x2_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmin_s16x2_relu-NEXT:   sycl::ext::intel::math::viaddmin_s16x2_relu<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmin_s32 | FileCheck %s -check-prefix=__viaddmin_s32
// __viaddmin_s32: CUDA API:
// __viaddmin_s32-NEXT:   __viaddmin_s32(a /*const int*/, b /*const int*/, c /*const int*/);
// __viaddmin_s32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmin_s32-NEXT:   sycl::ext::intel::math::viaddmin_s32<int>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmin_s32_relu | FileCheck %s -check-prefix=__viaddmin_s32_relu
// __viaddmin_s32_relu: CUDA API:
// __viaddmin_s32_relu-NEXT:   __viaddmin_s32_relu(a /*const int*/, b /*const int*/, c /*const int*/);
// __viaddmin_s32_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmin_s32_relu-NEXT:   sycl::ext::intel::math::viaddmin_s32_relu<int>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmin_u16x2 | FileCheck %s -check-prefix=__viaddmin_u16x2
// __viaddmin_u16x2: CUDA API:
// __viaddmin_u16x2-NEXT:   __viaddmin_u16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __viaddmin_u16x2-NEXT:                    c /*const unsigned int*/);
// __viaddmin_u16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmin_u16x2-NEXT:   sycl::ext::intel::math::viaddmin_u16x2<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__viaddmin_u32 | FileCheck %s -check-prefix=__viaddmin_u32
// __viaddmin_u32: CUDA API:
// __viaddmin_u32-NEXT:   __viaddmin_u32(a /*const unsigned int*/, b /*const unsigned int*/,
// __viaddmin_u32-NEXT:                  c /*const unsigned int*/);
// __viaddmin_u32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __viaddmin_u32-NEXT:   sycl::ext::intel::math::viaddmin_u32<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vibmax_s16x2 | FileCheck %s -check-prefix=__vibmax_s16x2
// __vibmax_s16x2: CUDA API:
// __vibmax_s16x2-NEXT:   __vibmax_s16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __vibmax_s16x2-NEXT:                  pred_hi /*bool *const*/, pred_lo /*bool *const*/);
// __vibmax_s16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vibmax_s16x2-NEXT:   sycl::ext::intel::math::vibmax_s16x2<unsigned>(a, b, pred_hi, pred_lo);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vibmax_s32 | FileCheck %s -check-prefix=__vibmax_s32
// __vibmax_s32: CUDA API:
// __vibmax_s32-NEXT:   __vibmax_s32(a /*const int*/, b /*const int*/, pred /*bool *const*/);
// __vibmax_s32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vibmax_s32-NEXT:   sycl::ext::intel::math::vibmax_s32<int>(a, b, pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vibmax_u16x2 | FileCheck %s -check-prefix=__vibmax_u16x2
// __vibmax_u16x2: CUDA API:
// __vibmax_u16x2-NEXT:   __vibmax_u16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __vibmax_u16x2-NEXT:                  pred_hi /*bool *const*/, pred_lo /*bool *const*/);
// __vibmax_u16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vibmax_u16x2-NEXT:   sycl::ext::intel::math::vibmax_u16x2<unsigned>(a, b, pred_hi, pred_lo);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vibmax_u32 | FileCheck %s -check-prefix=__vibmax_u32
// __vibmax_u32: CUDA API:
// __vibmax_u32-NEXT:   __vibmax_u32(a /*const unsigned int*/, b /*const unsigned int*/,
// __vibmax_u32-NEXT:                pred /*bool *const*/);
// __vibmax_u32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vibmax_u32-NEXT:   sycl::ext::intel::math::vibmax_u32<unsigned>(a, b, pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vibmin_s16x2 | FileCheck %s -check-prefix=__vibmin_s16x2
// __vibmin_s16x2: CUDA API:
// __vibmin_s16x2-NEXT:   __vibmin_s16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __vibmin_s16x2-NEXT:                  pred_hi /*bool *const*/, pred_lo /*bool *const*/);
// __vibmin_s16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vibmin_s16x2-NEXT:   sycl::ext::intel::math::vibmin_s16x2<unsigned>(a, b, pred_hi, pred_lo);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vibmin_s32 | FileCheck %s -check-prefix=__vibmin_s32
// __vibmin_s32: CUDA API:
// __vibmin_s32-NEXT:   __vibmin_s32(a /*const int*/, b /*const int*/, pred /*bool *const*/);
// __vibmin_s32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vibmin_s32-NEXT:   sycl::ext::intel::math::vibmin_s32<int>(a, b, pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vibmin_u16x2 | FileCheck %s -check-prefix=__vibmin_u16x2
// __vibmin_u16x2: CUDA API:
// __vibmin_u16x2-NEXT:   __vibmin_u16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __vibmin_u16x2-NEXT:                  pred_hi /*bool *const*/, pred_lo /*bool *const*/);
// __vibmin_u16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vibmin_u16x2-NEXT:   sycl::ext::intel::math::vibmin_u16x2<unsigned>(a, b, pred_hi, pred_lo);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vibmin_u32 | FileCheck %s -check-prefix=__vibmin_u32
// __vibmin_u32: CUDA API:
// __vibmin_u32-NEXT:   __vibmin_u32(a /*const unsigned int*/, b /*const unsigned int*/,
// __vibmin_u32-NEXT:                pred /*bool *const*/);
// __vibmin_u32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vibmin_u32-NEXT:   sycl::ext::intel::math::vibmin_u32<unsigned>(a, b, pred);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimax3_s16x2 | FileCheck %s -check-prefix=__vimax3_s16x2
// __vimax3_s16x2: CUDA API:
// __vimax3_s16x2-NEXT:   __vimax3_s16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __vimax3_s16x2-NEXT:                  c /*const unsigned int*/);
// __vimax3_s16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimax3_s16x2-NEXT:   sycl::ext::intel::math::vimax3_s16x2<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimax3_s16x2_relu | FileCheck %s -check-prefix=__vimax3_s16x2_relu
// __vimax3_s16x2_relu: CUDA API:
// __vimax3_s16x2_relu-NEXT:   __vimax3_s16x2_relu(a /*const unsigned int*/, b /*const unsigned int*/,
// __vimax3_s16x2_relu-NEXT:                       c /*const unsigned int*/);
// __vimax3_s16x2_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimax3_s16x2_relu-NEXT:   sycl::ext::intel::math::vimax3_s16x2_relu<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimax3_s32 | FileCheck %s -check-prefix=__vimax3_s32
// __vimax3_s32: CUDA API:
// __vimax3_s32-NEXT:   __vimax3_s32(a /*const int*/, b /*const int*/, c /*const int*/);
// __vimax3_s32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimax3_s32-NEXT:   sycl::ext::intel::math::vimax3_s32<int>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimax3_s32_relu | FileCheck %s -check-prefix=__vimax3_s32_relu
// __vimax3_s32_relu: CUDA API:
// __vimax3_s32_relu-NEXT:   __vimax3_s32_relu(a /*const int*/, b /*const int*/, c /*const int*/);
// __vimax3_s32_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimax3_s32_relu-NEXT:   sycl::ext::intel::math::vimax3_s32_relu<int>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimax3_u16x2 | FileCheck %s -check-prefix=__vimax3_u16x2
// __vimax3_u16x2: CUDA API:
// __vimax3_u16x2-NEXT:   __vimax3_u16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __vimax3_u16x2-NEXT:                  c /*const unsigned int*/);
// __vimax3_u16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimax3_u16x2-NEXT:   sycl::ext::intel::math::vimax3_u16x2<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimax3_u32 | FileCheck %s -check-prefix=__vimax3_u32
// __vimax3_u32: CUDA API:
// __vimax3_u32-NEXT:   __vimax3_u32(a /*const unsigned int*/, b /*const unsigned int*/,
// __vimax3_u32-NEXT:                c /*const unsigned int*/);
// __vimax3_u32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimax3_u32-NEXT:   sycl::ext::intel::math::vimax3_u32<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimax_s16x2_relu | FileCheck %s -check-prefix=__vimax_s16x2_relu
// __vimax_s16x2_relu: CUDA API:
// __vimax_s16x2_relu-NEXT:   __vimax_s16x2_relu(a /*const unsigned int*/, b /*const unsigned int*/);
// __vimax_s16x2_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimax_s16x2_relu-NEXT:   sycl::ext::intel::math::vimax_s16x2_relu<unsigned>(a, b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimax_s32_relu | FileCheck %s -check-prefix=__vimax_s32_relu
// __vimax_s32_relu: CUDA API:
// __vimax_s32_relu-NEXT:   __vimax_s32_relu(a /*const int*/, b /*const int*/);
// __vimax_s32_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimax_s32_relu-NEXT:   sycl::ext::intel::math::vimax_s32_relu<int>(a, b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimin3_s16x2 | FileCheck %s -check-prefix=__vimin3_s16x2
// __vimin3_s16x2: CUDA API:
// __vimin3_s16x2-NEXT:   __vimin3_s16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __vimin3_s16x2-NEXT:                  c /*const unsigned int*/);
// __vimin3_s16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimin3_s16x2-NEXT:   sycl::ext::intel::math::vimin3_s16x2<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimin3_s16x2_relu | FileCheck %s -check-prefix=__vimin3_s16x2_relu
// __vimin3_s16x2_relu: CUDA API:
// __vimin3_s16x2_relu-NEXT:   __vimin3_s16x2_relu(a /*const unsigned int*/, b /*const unsigned int*/,
// __vimin3_s16x2_relu-NEXT:                       c /*const unsigned int*/);
// __vimin3_s16x2_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimin3_s16x2_relu-NEXT:   sycl::ext::intel::math::vimin3_s16x2_relu<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimin3_s32 | FileCheck %s -check-prefix=__vimin3_s32
// __vimin3_s32: CUDA API:
// __vimin3_s32-NEXT:   __vimin3_s32(a /*const int*/, b /*const int*/, c /*const int*/);
// __vimin3_s32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimin3_s32-NEXT:   sycl::ext::intel::math::vimin3_s32<int>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimin3_s32_relu | FileCheck %s -check-prefix=__vimin3_s32_relu
// __vimin3_s32_relu: CUDA API:
// __vimin3_s32_relu-NEXT:   __vimin3_s32_relu(a /*const int*/, b /*const int*/, c /*const int*/);
// __vimin3_s32_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimin3_s32_relu-NEXT:   sycl::ext::intel::math::vimin3_s32_relu<int>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimin3_u16x2 | FileCheck %s -check-prefix=__vimin3_u16x2
// __vimin3_u16x2: CUDA API:
// __vimin3_u16x2-NEXT:   __vimin3_u16x2(a /*const unsigned int*/, b /*const unsigned int*/,
// __vimin3_u16x2-NEXT:                  c /*const unsigned int*/);
// __vimin3_u16x2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimin3_u16x2-NEXT:   sycl::ext::intel::math::vimin3_u16x2<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimin3_u32 | FileCheck %s -check-prefix=__vimin3_u32
// __vimin3_u32: CUDA API:
// __vimin3_u32-NEXT:   __vimin3_u32(a /*const unsigned int*/, b /*const unsigned int*/,
// __vimin3_u32-NEXT:                c /*const unsigned int*/);
// __vimin3_u32-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimin3_u32-NEXT:   sycl::ext::intel::math::vimin3_u32<unsigned>(a, b, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimin_s16x2_relu | FileCheck %s -check-prefix=__vimin_s16x2_relu
// __vimin_s16x2_relu: CUDA API:
// __vimin_s16x2_relu-NEXT:   __vimin_s16x2_relu(a /*const unsigned int*/, b /*const unsigned int*/);
// __vimin_s16x2_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimin_s16x2_relu-NEXT:   sycl::ext::intel::math::vimin_s16x2_relu<unsigned>(a, b);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vimin_s32_relu | FileCheck %s -check-prefix=__vimin_s32_relu
// __vimin_s32_relu: CUDA API:
// __vimin_s32_relu-NEXT:   __vimin_s32_relu(a /*const int*/, b /*const int*/);
// __vimin_s32_relu-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __vimin_s32_relu-NEXT:   sycl::ext::intel::math::vimin_s32_relu<int>(a, b);
