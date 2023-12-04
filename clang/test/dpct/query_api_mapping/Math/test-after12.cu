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
