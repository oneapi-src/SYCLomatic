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
// CUDALAUNCHCOOPERATIVEKERNEL-NEXT:   dpct::kernel_launcher::launch(f, gridDim, blockDim, args, sharedMem, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__barrier_sync | FileCheck %s -check-prefix=__barrier_sync
// __barrier_sync: CUDA API:
// __barrier_sync-NEXT:   __barrier_sync(id /*unsigned*/);
// __barrier_sync-NEXT: Is migrated to:
// __barrier_sync-NEXT:   sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier();
