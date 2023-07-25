// RUN: dpct --in-root=%S/api_mapping_cases --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamGetFlags | FileCheck %s -check-prefix=CUDASTREAMGETFLAGS
// CUDASTREAMGETFLAGS: CUDA API: cudaStreamGetFlags(s, f);
// CUDASTREAMGETFLAGS-NEXT: Is migrated to: *(f) = 0;

// RUN: dpct --in-root=%S/api_mapping_cases --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaEventDestroy | FileCheck %s -check-prefix=CUDAEVENTDESTROY
// CUDAEVENTDESTROY: CUDA API: cudaEventDestroy(e);
// CUDAEVENTDESTROY-NEXT: Is migrated to: dpct::destroy_event(e);

// RUN: dpct --in-root=%S/api_mapping_cases --cuda-include-path="%cuda-path/include" --query-api-mapping=__hfma | FileCheck %s -check-prefix=HFMA
// HFMA: CUDA API: __hfma(h1, h2, h3);
// HFMA-NEXT: Is migrated to: sycl::fma(h1, h2, h3);

// RUN: dpct --in-root=%S/api_mapping_cases --cuda-include-path="%cuda-path/include" --query-api-mapping=__hfma_sat | FileCheck %s -check-prefix=HFMA_SAT
// HFMA_SAT: CUDA API: __hfma_sat(h1, h2, h3);
// HFMA_SAT-NEXT: Is migrated to: sycl::clamp<sycl::half>(sycl::fma(h1, h2, h3), 0.f, 1.0f);

// RUN: dpct --in-root=%S/api_mapping_cases --cuda-include-path="%cuda-path/include" --query-api-mapping=aaa | FileCheck %s -check-prefix=AAA
// AAA: The API Mapping is not available
