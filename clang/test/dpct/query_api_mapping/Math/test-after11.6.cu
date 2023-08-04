// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hfma | FileCheck %s -check-prefix=HFMA
// HFMA: CUDA API:
// HFMA-NEXT:   __hfma(h1 /*__half*/, h2 /*__half*/, h3 /*__half*/);
// HFMA-NEXT:   __hfma(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/, b3 /*__nv_bfloat16*/);
// HFMA-NEXT: Is migrated to (with some neccessary option):
// HFMA-NEXT:   sycl::ext::intel::math::hfma(h1, h2);
// HFMA-NEXT:   b1 * b2 + b3;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hfma_sat | FileCheck %s -check-prefix=HFMA_SAT
// HFMA_SAT: CUDA API:
// HFMA_SAT-NEXT:   __hfma_sat(h1 /*__half*/, h2 /*__half*/, h3 /*__half*/);
// HFMA_SAT-NEXT:   __hfma_sat(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/, b3 /*__nv_bfloat16*/);
// HFMA_SAT-NEXT: Is migrated to (with some neccessary option):
// HFMA_SAT-NEXT:   sycl::ext::intel::math::hfma_sat(h1, h2, h3);
// HFMA_SAT-NEXT:   dpct::clamp(b1 * b2 + b3, 0.f, 1.0f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=aaa | FileCheck %s -check-prefix=AAA
// AAA: The API Mapping is not available
