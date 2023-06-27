// RUN: dpct --query-api-mapping=cudaStreamGetFlags | FileCheck %s -check-prefix=CUDASTREAMGETFLAGS
// CUDASTREAMGETFLAGS: CUDA API: cudaStreamGetFlags
// CUDASTREAMGETFLAGS-NEXT: Is migrated to: an expression statement which set the output parameter 'flags' to 0

// RUN: dpct --query-api-mapping=cudaEventDestroy | FileCheck %s -check-prefix=CUDAEVENTDESTROY
// CUDAEVENTDESTROY: CUDA API: cudaEventDestroy
// CUDAEVENTDESTROY-NEXT: Is migrated to: dpct::destroy_event(event_ptr event)

// RUN: dpct --query-api-mapping=__hfma | FileCheck %s -check-prefix=HFMA
// HFMA: CUDA API: __hfma
// HFMA-NEXT: Is migrated to: sycl::fma(genfloat a, genfloat b, genfloat c)

// RUN: dpct --query-api-mapping=__hfma_sat | FileCheck %s -check-prefix=HFMA_SAT
// HFMA_SAT: CUDA API: __hfma_sat
// HFMA_SAT-NEXT: Is migrated to: sycl::ext::intel::math::hfma_sat(sycl::half x, sycl::half y, sycl::half z)
// HFMA_SAT-NEXT: There are multi migration solutions for this API with different migration options,
// HFMA_SAT-NEXT: suggest to use the tool to migrate a complete test case to see more detail of the migration.
