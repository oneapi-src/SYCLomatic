// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none --usm-level=restricted -out-root %T/memory_management_restricted %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++11
// RUN: FileCheck --match-full-lines --input-file %T/memory_management_restricted/memory_management_restricted.dp.cpp %s

#include <cuda_runtime.h>
#include <cuda.h>
#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)
template <typename T>
void my_error_checker(T ReturnValue, char const *const FuncName) {}

#define DATAMACRO 32*32

//CHECK: template<typename T>
//CHECK-NEXT: void test(){
//CHECK-NEXT:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK-NEXT:   sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK-NEXT:   int i = 0;
//CHECK-NEXT:   T** ptr;
//CHECK-NEXT:   T* array[10];
//CHECK-NEXT:   ptr[i] = (T *)sycl::malloc_device(10 * sizeof(T), q_ct1);
//CHECK-NEXT:   ptr[i] = (T *)sycl::malloc_device(10 * sizeof(T), q_ct1);
//CHECK-NEXT:   array[i] = (T *)sycl::malloc_device(10 * sizeof(T), q_ct1);
//CHECK-NEXT: }
template<typename T>
void test(){
  int i = 0;
  T** ptr;
  T* array[10];
  cudaMalloc(&ptr[i], 10 * sizeof(T));
  cudaMalloc(&(ptr[i]), 10 * sizeof(T));
  cudaMalloc(&array[i], 10 * sizeof(T));
}

__device__ float d_A_static[10];

int main(){
    //CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
    //CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();

    float **data = NULL;
    float *d_A = NULL;
    float *h_A = NULL;
    int* a;
    cudaStream_t stream;
    cudaStream_t stream_array[10];
    int deviceID = 0;
    cudaError_t err;

    //CHECK:  /*
    //CHECK-NEXT:  DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT:  */
    //CHECK-NEXT:  MY_ERROR_CHECKER((*data = (float *)sycl::malloc_device(DATAMACRO, q_ct1), 0));
    MY_ERROR_CHECKER(cudaMalloc((void **)data, DATAMACRO));

    //Currently, migration of using template version API only covers the simple case: the argument specifiy the size is sizeof(T)*Expr, Expr*sizeof(T) and sizeof(T)
    //CHECK:  *data = sycl::malloc_device<float>(10*10, q_ct1);
    cudaMalloc(data, 10*10*sizeof(float));

    //CHECK:  *data = (float *)sycl::malloc_device(10*sizeof(float)*10, q_ct1);
    cudaMalloc(data, 10*sizeof(float)*10);

    //CHECK:  *data = (float *)sycl::malloc_device(sizeof(float)*10*10, q_ct1);
    cudaMalloc(data, sizeof(float)*10*10);

    size_t size2;
    // CHECK: size2 = d_A_static.get_size();
    cudaGetSymbolSize(&size2, d_A_static);

    // CHECK: /*
    // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:  */
    // CHECK-NEXT:  err = (size2 = d_A_static.get_size(), 0);
    err = cudaGetSymbolSize(&size2, d_A_static);

    // CHECK: /*
    // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:*/
    // CHECK-NEXT:MY_ERROR_CHECKER((size2 = d_A_static.get_size(), 0));
    MY_ERROR_CHECKER(cudaGetSymbolSize(&size2, d_A_static));

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
    // CHECK-NEXT: MY_ERROR_CHECKER((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, deviceID, NULL));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: MY_ERROR_CHECKER((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, deviceID, 0));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: MY_ERROR_CHECKER((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, deviceID, nullptr));

    //CHECK: stream_array[0]->memcpy(h_A, d_A, size2);
    cudaMemcpyAsync(h_A, d_A, size2, cudaMemcpyDeviceToHost, stream_array[0]);

    // CHECK: q_ct1.memcpy(h_A, d_A, size2);
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: MY_ERROR_CHECKER((q_ct1.memcpy(h_A, d_A, size2), 0));
    // CHECK-NEXT: q_ct1.memcpy(h_A, d_A, size2);
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: MY_ERROR_CHECKER((q_ct1.memcpy(h_A, d_A, size2), 0));
    // CHECK-NEXT: q_ct1.memcpy(h_A, d_A, size2);
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: MY_ERROR_CHECKER((q_ct1.memcpy(h_A, d_A, size2), 0));
    cudaMemcpyAsync(h_A, d_A, size2, cudaMemcpyDeviceToHost, cudaStreamDefault);
    MY_ERROR_CHECKER(cudaMemcpyAsync(h_A, d_A, size2, cudaMemcpyDeviceToHost, cudaStreamDefault));
    cudaMemcpyAsync(h_A, d_A, size2, cudaMemcpyDeviceToHost, cudaStreamPerThread);
    MY_ERROR_CHECKER(cudaMemcpyAsync(h_A, d_A, size2, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    cudaMemcpyAsync(h_A, d_A, size2, cudaMemcpyDeviceToHost, cudaStreamLegacy);
    MY_ERROR_CHECKER(cudaMemcpyAsync(h_A, d_A, size2, cudaMemcpyDeviceToHost, cudaStreamLegacy));
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
    // CHECK-NEXT: MY_ERROR_CHECKER((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, deviceID, NULL));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: MY_ERROR_CHECKER((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, deviceID, 0));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: MY_ERROR_CHECKER((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100), 0));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, deviceID, nullptr));
    return 0;
}

template int foo<float>();
template int foo<int>();

void checkError(cudaError_t err) {
}
void cuCheckError(CUresult err) {
}
void foobar() {
  int errorCode;

  cudaChannelFormatDesc desc;
  cudaExtent extent = make_cudaExtent(1, 1, 1);
  unsigned int flags;
  cudaArray_t array;

  // CHECK: desc = array->get_channel();
  // CHECK: extent = array->get_range();
  // CHECK: flags = 0;
  cudaArrayGetInfo(&desc, &extent, &flags, array);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError(([&](){
  // CHECK-NEXT:   desc = array->get_channel();
  // CHECK-NEXT:   extent = array->get_range();
  // CHECK-NEXT:   flags = 0;
  // CHECK-NEXT:   }(), 0));
  checkError(cudaArrayGetInfo(&desc, &extent, &flags, array));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: errorCode = ([&](){
  // CHECK-NEXT:   desc = array->get_channel();
  // CHECK-NEXT:   extent = array->get_range();
  // CHECK-NEXT:   flags = 0;
  // CHECK-NEXT:   }(), 0);
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

#define QRNG_DIMENSIONS 3
#define AAA(x)   (x + 2)
#define BBB(x)   x * 2
//CHECK: #define SIZEOF_FLOAT sizeof(float)
//CHECK-NEXT: #define SIZEOF(x) sizeof(x)
#define SIZEOF_FLOAT sizeof(float)
#define SIZEOF(x) sizeof(x)


  const int N = 1048576;
  float *d_Output;


  // a * sizeof
  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(AAA(N) * N, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, AAA(N) * N * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(N * AAA(N), q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, N * AAA(N) * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(N * QRNG_DIMENSIONS, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, N * QRNG_DIMENSIONS * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS * N, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER((d_Output = (float *)sycl::malloc_device(QRNG_DIMENSIONS * N * SIZEOF_FLOAT, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * SIZEOF_FLOAT));

  //CHECK: d_Output = (float *)sycl::malloc_device(QRNG_DIMENSIONS * N * SIZEOF_FLOAT, q_ct1);
  cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * SIZEOF_FLOAT);

  //CHECK: MY_ERROR_CHECKER((d_Output = (float *)sycl::malloc_device(QRNG_DIMENSIONS * N * SIZEOF(float), q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * SIZEOF(float)));

  //CHECK: d_Output = (float *)sycl::malloc_device(QRNG_DIMENSIONS * N * SIZEOF(float), q_ct1);
  cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * SIZEOF(float));

  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(N * N, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, N * N * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS * QRNG_DIMENSIONS, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * QRNG_DIMENSIONS * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(AAA(N) * AAA(N), q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, AAA(N) * AAA(N) * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(AAA(N) * QRNG_DIMENSIONS, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, AAA(N) * QRNG_DIMENSIONS * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS * AAA(N), q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * AAA(N) * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER((d_Output = (float *)sycl::malloc_device(N * N * SIZEOF_FLOAT, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, N * N * SIZEOF_FLOAT));

  //CHECK: MY_ERROR_CHECKER((d_Output = (float *)sycl::malloc_device(N * N * SIZEOF(float), q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, N * N * SIZEOF(float)));



  //CHECK: d_Output = sycl::malloc_device<float>(AAA(N) * N, q_ct1);
  cudaMalloc((void **)&d_Output, AAA(N) * N * sizeof(float));

  //CHECK: d_Output = sycl::malloc_device<float>(N * AAA(N), q_ct1);
  cudaMalloc((void **)&d_Output, N * AAA(N) * sizeof(float));

  //CHECK: d_Output = sycl::malloc_device<float>(N * QRNG_DIMENSIONS, q_ct1);
  cudaMalloc((void **)&d_Output, N * QRNG_DIMENSIONS * sizeof(float));

  //CHECK: d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS * N, q_ct1);
  cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * sizeof(float));

  //CHECK: d_Output = sycl::malloc_device<float>(N * N, q_ct1);
  cudaMalloc((void **)&d_Output, N * N * sizeof(float));

  //CHECK: d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS * QRNG_DIMENSIONS, q_ct1);
  cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * QRNG_DIMENSIONS * sizeof(float));

  //CHECK: d_Output = sycl::malloc_device<float>(AAA(N) * AAA(N), q_ct1);
  cudaMalloc((void **)&d_Output, AAA(N) * AAA(N) * sizeof(float));

  //CHECK: d_Output = sycl::malloc_device<float>(AAA(N) * QRNG_DIMENSIONS, q_ct1);
  cudaMalloc((void **)&d_Output, AAA(N) * QRNG_DIMENSIONS * sizeof(float));

  //CHECK: d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS * AAA(N), q_ct1);
  cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * AAA(N) * sizeof(float));

  //CHECK: d_Output = (float *)sycl::malloc_device(N * N * SIZEOF_FLOAT, q_ct1);
  cudaMalloc((void **)&d_Output, N * N * SIZEOF_FLOAT);

  //CHECK: d_Output = (float *)sycl::malloc_device(N * N * SIZEOF(float), q_ct1);
  cudaMalloc((void **)&d_Output, N * N * SIZEOF(float));



  // sizeof * a
  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, sizeof(float) * QRNG_DIMENSIONS));

  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(N, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, sizeof(float) * N));

  //CHECK: MY_ERROR_CHECKER((d_Output = (float *)sycl::malloc_device(sizeof(float) * BBB(N), q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, sizeof(float) * BBB(N)));

  //CHECK: d_Output = (float *)sycl::malloc_device(sizeof(float) * BBB(N), q_ct1);
  cudaMalloc((void **)&d_Output, sizeof(float) * BBB(N));

  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(AAA(N), q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, sizeof(float) * AAA(N)));

  //CHECK: d_Output = sycl::malloc_device<float>(AAA(N), q_ct1);
  cudaMalloc((void **)&d_Output, sizeof(float) * AAA(N));

  //CHECK: d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS, q_ct1);
  cudaMalloc((void **)&d_Output, sizeof(float) * QRNG_DIMENSIONS);

  //CHECK: d_Output = sycl::malloc_device<float>(N, q_ct1);
  cudaMalloc((void **)&d_Output, sizeof(float) * N);

  //CHECK: d_Output = (float *)sycl::malloc_device(SIZEOF(float) * N, q_ct1);
  cudaMalloc((void **)&d_Output, SIZEOF(float) * N);

  //CHECK: d_Output = (float *)sycl::malloc_device(SIZEOF_FLOAT * N, q_ct1);
  cudaMalloc((void **)&d_Output, SIZEOF_FLOAT * N);



  // sizeof
  //CHECK: d_Output = sycl::malloc_device<float>(1, q_ct1);
  cudaMalloc((void **)&d_Output, sizeof(float));

  //CHECK: d_Output = (float *)sycl::malloc_device(SIZEOF_FLOAT, q_ct1);
  cudaMalloc((void **)&d_Output, SIZEOF_FLOAT);

  //CHECK: d_Output = (float *)sycl::malloc_device(SIZEOF(float), q_ct1);
  cudaMalloc((void **)&d_Output, SIZEOF(float));

  //CHECK: MY_ERROR_CHECKER((d_Output = (float *)sycl::malloc_device(SIZEOF_FLOAT, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, SIZEOF_FLOAT));

  //CHECK: MY_ERROR_CHECKER((d_Output = (float *)sycl::malloc_device(SIZEOF(float), q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, SIZEOF(float)));

  //CHECK: MY_ERROR_CHECKER((d_Output = sycl::malloc_device<float>(1, q_ct1), 0));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, sizeof(float)));

  size_t free_mem, total_mem;

  //CHECK: dpct::get_current_device().get_memory_info(free_mem, total_mem);
  cudaMemGetInfo(&free_mem, &total_mem);
  //CHECK: MY_ERROR_CHECKER((dpct::get_current_device().get_memory_info(free_mem, total_mem), 0));
  MY_ERROR_CHECKER(cudaMemGetInfo(&free_mem, &total_mem));
  //CHECK: dpct::get_current_device().get_memory_info(free_mem, total_mem);
  cuMemGetInfo(&free_mem, &total_mem);
  //CHECK: MY_ERROR_CHECKER((dpct::get_current_device().get_memory_info(free_mem, total_mem), 0));
  MY_ERROR_CHECKER(cuMemGetInfo(&free_mem, &total_mem));
}

template <typename T>
struct Data {
  T *Pos1[2];
  T **Pos2;
};


template <typename T>
void foo(int a, int b) {
  auto c = new Data<T>[a];
  // CHECK: c[0].Pos1[0] = (typename std::remove_reference<decltype(c[0].Pos1[0])>::type)sycl::malloc_device(b, q_ct1);
  // CHECK-NEXT: *(c[0].Pos2) = (typename std::remove_pointer<decltype(c[0].Pos2)>::type)sycl::malloc_device(b, q_ct1);
  cudaMalloc((void **)&c[0].Pos1[0], b);
  cudaMalloc((void **)c[0].Pos2, b);
}

__managed__ char *a1;
__managed__ char *a2;
void foo2() {
  // CHECK: *(a1.get_ptr()) = (char *)sycl::malloc_shared(4, q_ct1);
  // CHECK-NEXT: *(a2.get_ptr()) = sycl::malloc_shared<char>(1, q_ct1);
  // CHECK-NEXT: sycl::free(*(a1.get_ptr()), q_ct1);
  // CHECK-NEXT: sycl::free(*(a2.get_ptr()), q_ct1);
  cudaMallocManaged((void **)&a1, 4);
  cudaMallocManaged((void **)&a2, sizeof(char));
  cudaFree(a1);
  cudaFree(a2);
}
