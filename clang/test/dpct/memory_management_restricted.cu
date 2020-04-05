// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none --usm-level=restricted -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++11
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
    //CHECK-NEXT:  DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT:  */
    //CHECK-NEXT:  checkCudaErrors((*data = (float *)sycl::malloc_device(DATAMACRO, dpct::get_current_device(), dpct::get_default_context()), 0));
    checkCudaErrors(cudaMalloc((void **)data, DATAMACRO));

    size_t size2;
    // CHECK: size2 = d_A.get_size();
    cudaGetSymbolSize(&size2, d_A);

    // CHECK: /*
    // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:  */
    // CHECK-NEXT:  err = (size2 = d_A.get_size(), 0);
    err = cudaGetSymbolSize(&size2, d_A);

    // CHECK: /*
    // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:*/
    // CHECK-NEXT:checkCudaErrors((size2 = d_A.get_size(), 0));
    checkCudaErrors(cudaGetSymbolSize(&size2, d_A));

    // CHECK: stream->prefetch(a,100);
    cudaMemPrefetchAsync (a, 100, deviceID, stream);

    // CHECK: (*&stream)->prefetch(a,100);
    cudaMemPrefetchAsync (a, 100, deviceID, *&stream);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: err = (dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0);
    err = cudaMemPrefetchAsync(a, 100, deviceID);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    checkCudaErrors(cudaMemPrefetchAsync(a, 100, deviceID, NULL));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    checkCudaErrors(cudaMemPrefetchAsync(a, 100, deviceID, 0));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    checkCudaErrors(cudaMemPrefetchAsync(a, 100, deviceID, nullptr));
}


template <typename T>
int foo() {
    T* a;
    cudaStream_t stream;
    int deviceID = 0;
    cudaError_t err;
    // CHECK: stream->prefetch(a,100);
    cudaMemPrefetchAsync (a, 100, deviceID, stream);

    // CHECK: (*&stream)->prefetch(a,100);
    cudaMemPrefetchAsync (a, 100, deviceID, *&stream);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: err = (dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0);
    err = cudaMemPrefetchAsync(a, 100, deviceID);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    checkCudaErrors(cudaMemPrefetchAsync(a, 100, deviceID, NULL));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    checkCudaErrors(cudaMemPrefetchAsync(a, 100, deviceID, 0));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    checkCudaErrors(cudaMemPrefetchAsync(a, 100, deviceID, nullptr));
    return 0;
}

template int foo<float>();
template int foo<int>();

void checkError(cudaError_t err) {
}

void foobar() {
  int errorCode;

  cudaChannelFormatDesc desc;
  cudaExtent extent;
  unsigned int flags;
  cudaArray_t array;

  // CHECK: array->get_info(desc, extent, flags);
  cudaArrayGetInfo(&desc, &extent, &flags, array);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError((array->get_info(desc, extent, flags), 0));
  checkError(cudaArrayGetInfo(&desc, &extent, &flags, array));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: errorCode = (array->get_info(desc, extent, flags), 0);
  errorCode = cudaArrayGetInfo(&desc, &extent, &flags, array);

  int host;
  // CHECK: flags = 0;
  cudaHostGetFlags(&flags, &host);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError((flags = 0, 0));
  checkError(cudaHostGetFlags(&flags, &host));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: errorCode = (flags = 0, 0);
  errorCode = cudaHostGetFlags(&flags, &host);

  int *devPtr;
  size_t count;
  // CHECK: pi_mem_advice advice;
  cudaMemoryAdvise advice;
  int device;
  // CHECK: dpct::get_device(device).default_queue().mem_advise(devPtr, count, pi_mem_advice(advice - 1));
  cudaMemAdvise(devPtr, count, advice, device);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError((dpct::get_device(device).default_queue().mem_advise(devPtr, count, pi_mem_advice(advice - 1)), 0));
  checkError(cudaMemAdvise(devPtr, count, advice, device));
  // CHECK: checkError((dpct::get_device(device).default_queue().mem_advise(devPtr, count, PI_MEM_ADVICE_SET_READ_MOSTLY), 0));
  checkError(cudaMemAdvise(devPtr, count, cudaMemoryAdvise(1), device));
  // CHECK: checkError((dpct::get_device(device).default_queue().mem_advise(devPtr, count, PI_MEM_ADVICE_SET_READ_MOSTLY), 0));
  checkError(cudaMemAdvise(devPtr, count, (cudaMemoryAdvise)1, device));
  // CHECK: checkError((dpct::get_device(device).default_queue().mem_advise(devPtr, count, PI_MEM_ADVICE_SET_READ_MOSTLY), 0));
  checkError(cudaMemAdvise(devPtr, count, static_cast<cudaMemoryAdvise>(1), device));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: errorCode = (dpct::get_device(device).default_queue().mem_advise(devPtr, count, pi_mem_advice(advice - 1)), 0);
  errorCode = cudaMemAdvise(devPtr, count, advice, device);

  // CHECK: dpct::get_device(device).default_queue().mem_advise(devPtr, count, PI_MEM_ADVICE_SET_READ_MOSTLY);
  cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, device);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError((dpct::get_device(device).default_queue().mem_advise(devPtr, count, PI_MEM_ADVICE_SET_READ_MOSTLY), 0));
  checkError(cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, device));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: errorCode = (dpct::get_device(device).default_queue().mem_advise(devPtr, count, PI_MEM_ADVICE_SET_READ_MOSTLY), 0);
  errorCode = cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, device);
}
