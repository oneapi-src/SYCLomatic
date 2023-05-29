// FIXME
// UNSUPPORTED: system-windows
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

    //CHECK:  MY_ERROR_CHECKER(DPCT_CHECK_ERROR(*data = (float *)sycl::malloc_device(DATAMACRO, q_ct1)));
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

    // CHECK:  err = DPCT_CHECK_ERROR(size2 = d_A_static.get_size());
    err = cudaGetSymbolSize(&size2, d_A_static);

    // CHECK:MY_ERROR_CHECKER(DPCT_CHECK_ERROR(size2 = d_A_static.get_size()));
    MY_ERROR_CHECKER(cudaGetSymbolSize(&size2, d_A_static));

    // CHECK: stream->prefetch(a,100);
    cudaMemPrefetchAsync (a, 100, deviceID, stream);

    // CHECK: (*&stream)->prefetch(a,100);
    cudaMemPrefetchAsync (a, 100, deviceID, *&stream);

    // CHECK: err = DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100));
    err = cudaMemPrefetchAsync(a, 100, deviceID);

    // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100)));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, deviceID, NULL));

    // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100)));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, deviceID, 0));

    // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100)));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, deviceID, nullptr));

    // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dpct::cpu_device().default_queue().prefetch(a,100)));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, cudaCpuDeviceId, nullptr));

    //CHECK: stream_array[0]->memcpy(h_A, d_A, size2);
    cudaMemcpyAsync(h_A, d_A, size2, cudaMemcpyDeviceToHost, stream_array[0]);

    // CHECK: q_ct1.memcpy(h_A, d_A, size2);
    // CHECK-NEXT: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(q_ct1.memcpy(h_A, d_A, size2)));
    // CHECK-NEXT: q_ct1.memcpy(h_A, d_A, size2);
    // CHECK-NEXT: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(q_ct1.memcpy(h_A, d_A, size2)));
    // CHECK-NEXT: q_ct1.memcpy(h_A, d_A, size2);
    // CHECK-NEXT: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(q_ct1.memcpy(h_A, d_A, size2)));
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

    // CHECK: err = DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100));
    err = cudaMemPrefetchAsync(a, 100, deviceID);

    // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100)));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, deviceID, NULL));

    // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100)));
    MY_ERROR_CHECKER(cudaMemPrefetchAsync(a, 100, deviceID, 0));

    // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(a,100)));
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

  //CHECK: checkError(DPCT_CHECK_ERROR([&](){
  //CHECK-NEXT:   desc = array->get_channel();
  //CHECK-NEXT:   extent = array->get_range();
  //CHECK-NEXT:   flags = 0;
  //CHECK-NEXT:   }()));
  checkError(cudaArrayGetInfo(&desc, &extent, &flags, array));

  //CHECK: errorCode = DPCT_CHECK_ERROR([&](){
  //CHECK-NEXT:   desc = array->get_channel();
  //CHECK-NEXT:   extent = array->get_range();
  //CHECK-NEXT:   flags = 0;
  //CHECK-NEXT:   }());
  errorCode = cudaArrayGetInfo(&desc, &extent, &flags, array);

  int host;
  // CHECK: flags = 0;
  cudaHostGetFlags(&flags, &host);

  // CHECK: checkError(DPCT_CHECK_ERROR(flags = 0));
  checkError(cudaHostGetFlags(&flags, &host));

  // CHECK: errorCode = DPCT_CHECK_ERROR(flags = 0);
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
  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(AAA(N) * N, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, AAA(N) * N * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(N * AAA(N), q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, N * AAA(N) * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(N * QRNG_DIMENSIONS, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, N * QRNG_DIMENSIONS * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS * N, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = (float *)sycl::malloc_device(QRNG_DIMENSIONS * N * SIZEOF_FLOAT, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * SIZEOF_FLOAT));

  //CHECK: d_Output = (float *)sycl::malloc_device(QRNG_DIMENSIONS * N * SIZEOF_FLOAT, q_ct1);
  cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * SIZEOF_FLOAT);

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = (float *)sycl::malloc_device(QRNG_DIMENSIONS * N * SIZEOF(float), q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * SIZEOF(float)));

  //CHECK: d_Output = (float *)sycl::malloc_device(QRNG_DIMENSIONS * N * SIZEOF(float), q_ct1);
  cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * SIZEOF(float));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(N * N, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, N * N * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS * QRNG_DIMENSIONS, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * QRNG_DIMENSIONS * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(AAA(N) * AAA(N), q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, AAA(N) * AAA(N) * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(AAA(N) * QRNG_DIMENSIONS, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, AAA(N) * QRNG_DIMENSIONS * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS * AAA(N), q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * AAA(N) * sizeof(float)));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = (float *)sycl::malloc_device(N * N * SIZEOF_FLOAT, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, N * N * SIZEOF_FLOAT));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = (float *)sycl::malloc_device(N * N * SIZEOF(float), q_ct1)));
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
  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(QRNG_DIMENSIONS, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, sizeof(float) * QRNG_DIMENSIONS));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(N, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, sizeof(float) * N));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = (float *)sycl::malloc_device(sizeof(float) * BBB(N), q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, sizeof(float) * BBB(N)));

  //CHECK: d_Output = (float *)sycl::malloc_device(sizeof(float) * BBB(N), q_ct1);
  cudaMalloc((void **)&d_Output, sizeof(float) * BBB(N));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(AAA(N), q_ct1)));
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

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = (float *)sycl::malloc_device(SIZEOF_FLOAT, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, SIZEOF_FLOAT));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = (float *)sycl::malloc_device(SIZEOF(float), q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, SIZEOF(float)));

  //CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(d_Output = sycl::malloc_device<float>(1, q_ct1)));
  MY_ERROR_CHECKER(cudaMalloc((void **)&d_Output, sizeof(float)));

  size_t free_mem, total_mem;

  // CHECK:  /*
  // CHECK-NEXT:DPCT1106:{{[0-9]+}}: 'cudaMemGetInfo' was migrated with the Intel extensions for device information which may not be supported by all compilers or runtimes. You may need to adjust the code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:dpct::get_current_device().get_memory_info(free_mem, total_mem);
  cudaMemGetInfo(&free_mem, &total_mem);


  // CHECK: /*
  // CHECK-NEXT:DPCT1106:{{[0-9]+}}: 'cudaMemGetInfo' was migrated with the Intel extensions for device information which may not be supported by all compilers or runtimes. You may need to adjust the code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dpct::get_current_device().get_memory_info(free_mem, total_mem)));
  MY_ERROR_CHECKER(cudaMemGetInfo(&free_mem, &total_mem));

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1106:{{[0-9]+}}: 'cuMemGetInfo' was migrated with the Intel extensions for device information which may not be supported by all compilers or runtimes. You may need to adjust the code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  dpct::get_current_device().get_memory_info(free_mem, total_mem);
  cuMemGetInfo(&free_mem, &total_mem);

  // CHECK: /*
  // CHECK-NEXT:DPCT1106:{{[0-9]+}}: 'cuMemGetInfo' was migrated with the Intel extensions for device information which may not be supported by all compilers or runtimes. You may need to adjust the code.
  // CHECK-NEXT:*/
  // CHECK-NEXT: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dpct::get_current_device().get_memory_info(free_mem, total_mem)));
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

void foo3() {
  float *a, *b;
  // CHECK: auto flag = 0;
  // CHECK-NEXT: a = (float *)sycl::malloc_shared(100, q_ct1);
  // CHECK-NEXT: b = (float *)sycl::malloc_shared(100, q_ct1);
  auto flag = cudaMemAttachGlobal;
  cudaMallocManaged(&a, 100, flag);
  cudaMallocManaged(&b, 100, cudaMemAttachGlobal);
}

__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth){}

// CHECK:void foo_8() {
// CHECK-NEXT:  int width = 64, height = 64, depth = 64;
// CHECK-NEXT:  sycl::range<3> extent = sycl::range<3>(width * sizeof(float), height, depth);
// CHECK-NEXT:  dpct::pitched_data devPitchedPtr;
// CHECK-NEXT:  devPitchedPtr = dpct::dpct_malloc(extent);
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
// CHECK-NEXT:  */
// CHECK-NEXT:  dpct::get_default_queue().parallel_for(
// CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 100) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)),
// CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:      MyKernel(devPitchedPtr, width, height, depth);
// CHECK-NEXT:    });
// CHECK-NEXT:}
void foo_8() {
  int width = 64, height = 64, depth = 64;
  cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
  cudaPitchedPtr devPitchedPtr;
  cudaMalloc3D(&devPitchedPtr, extent);
  MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);
}
