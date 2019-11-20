// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --usm-level=restricted -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++11
// RUN: FileCheck --match-full-lines --input-file %T/memory_management_restricted.dp.cpp %s

#include <cuda_runtime.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {}

#define DATAMACRO 32*32

int main(){
    float **data = NULL;
    float *d_A = NULL;
    int* a;
    cudaStream_t stream;
    int deviceID = 0;
    cudaError_t err;

    //CHECK:  /*
    //CHECK-NEXT:  DPCT1003:0: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT:  */
    //CHECK-NEXT:  checkCudaErrors((*(data) = (float *)cl::sycl::malloc_device(DATAMACRO, dpct::get_current_device(), dpct::get_default_context()), 0));
    checkCudaErrors(cudaMalloc((void **)data, DATAMACRO));

    size_t size2;
    // CHECK: size2 = d_A.get_size();
    cudaGetSymbolSize(&size2, d_A);

    // CHECK: /*
    // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:  */
    // CHECK-NEXT:  err = (size2 = d_A.get_size(), 0);
    err = cudaGetSymbolSize(&size2, d_A);

    // CHECK: /*
    // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:*/
    // CHECK-NEXT:checkCudaErrors((size2 = d_A.get_size(), 0));
    checkCudaErrors(cudaGetSymbolSize(&size2, d_A));

    // CHECK: stream->prefetch(a,100);
    cudaMemPrefetchAsync (a, 100, deviceID, stream);

    // CHECK: (*&stream)->prefetch(a,100);
    cudaMemPrefetchAsync (a, 100, deviceID, *&stream);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: err = (dpct::get_device_manager().get_device(deviceID).default_queue().prefetch(a,100), 0);
    err = cudaMemPrefetchAsync(a, 100, deviceID);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((dpct::get_device_manager().get_device(deviceID).default_queue().prefetch(a,100), 0));
    checkCudaErrors(cudaMemPrefetchAsync(a, 100, deviceID, NULL));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((dpct::get_device_manager().get_device(deviceID).default_queue().prefetch(a,100), 0));
    checkCudaErrors(cudaMemPrefetchAsync(a, 100, deviceID, 0));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((dpct::get_device_manager().get_device(deviceID).default_queue().prefetch(a,100), 0));
    checkCudaErrors(cudaMemPrefetchAsync(a, 100, deviceID, nullptr));
}