// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2

/// Execution Control

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaFuncSetAttribute | FileCheck %s -check-prefix=CUDAFUNCSETATTRIBUTE
// CUDAFUNCSETATTRIBUTE: CUDA API:
// CUDAFUNCSETATTRIBUTE-NEXT:   cudaFuncSetAttribute(f /*const void **/, attr /*cudaFuncAttribute*/,
// CUDAFUNCSETATTRIBUTE-NEXT:                        i /*int*/);
// CUDAFUNCSETATTRIBUTE-NEXT: The API is Removed.
// CUDAFUNCSETATTRIBUTE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaLaunchCooperativeKernel | FileCheck %s -check-prefix=CUDALAUNCHCOOPERATIVEKERNEL
// CUDALAUNCHCOOPERATIVEKERNEL: CUDA API:
// CUDALAUNCHCOOPERATIVEKERNEL-NEXT:   cudaLaunchCooperativeKernel(f /*cudaError_t*/, gridDim /*dim3*/,
// CUDALAUNCHCOOPERATIVEKERNEL-NEXT:                               blockDim /*dim3*/, args /*void ***/,
// CUDALAUNCHCOOPERATIVEKERNEL-NEXT:                               sharedMem /*size_t*/, s /*cudaStream_t*/);
// CUDALAUNCHCOOPERATIVEKERNEL-NEXT: Is migrated to:
// CUDALAUNCHCOOPERATIVEKERNEL-NEXT:   s->parallel_for(
// CUDALAUNCHCOOPERATIVEKERNEL-NEXT:     sycl::nd_range<3>(gridDim * blockDim, blockDim),
// CUDALAUNCHCOOPERATIVEKERNEL-NEXT:     [=](sycl::nd_item<3> item_ct1) {
// CUDALAUNCHCOOPERATIVEKERNEL-NEXT:       f();
// CUDALAUNCHCOOPERATIVEKERNEL-NEXT:     });
