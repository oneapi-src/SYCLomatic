// RUN: dpct --format-range=none -usm-level=none -out-root %T/time-measure-usm-none %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/time-measure-usm-none/time-measure-usm-none.dp.cpp --match-full-lines %s
#include <stdio.h>

#define N 1000

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}


__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = 2*a[i];
    }
}

int main() {
    // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
    // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
    cudaStream_t stream;

    int ha[N], hb[N];
    // CHECK: std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    // CHECK: std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    cudaEvent_t start, stop;
    cudaError_t cudaStatus;

    int *da, *db;
    float elapsedTime;

    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));

    for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }


    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CHECK: start_ct1 = std::chrono::steady_clock::now();
    cudaEventRecord(start, 0);

    // CHECK: dpct::async_dpct_memcpy(da, ha, N*sizeof(int), dpct::host_to_device);
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);
    // CHECK: dpct::async_dpct_memcpy(da, ha, N*sizeof(int), dpct::host_to_device);
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice, 0);
    // CHECK: dpct::async_dpct_memcpy(da, ha, N*sizeof(int), dpct::host_to_device, *stream);
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice, stream);

    // CHECK: stream->wait();
    // CHECK: q_ct1.wait();
    // CHECK: stop_ct1 = std::chrono::steady_clock::now();
    // CHECK: elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    add<<<N, 1>>>(da, db);

    // CHECK: dpct::async_dpct_memcpy(hb, db, N*sizeof(int), dpct::device_to_host);
    cudaMemcpyAsync(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();


    for (int i = 0; i<N; ++i) {
        printf("%d\n", hb[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(da);
    cudaFree(db);

    return 0;
}


__global__ void kernel_foo(){}

void foo_test_1() {

    cudaEvent_t     start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

// CHECK:    start_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:        for (int i=0; i<4; i++) {
// CHECK-NEXT:            dpct::get_default_queue().submit(
// CHECK-NEXT:              [&](sycl::handler &cgh) {
// CHECK-NEXT:                cgh.parallel_for<dpct_kernel_name<class kernel_foo_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                  sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:                  [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                    kernel_foo();
// CHECK-NEXT:                  });
// CHECK-NEXT:              });
// CHECK-NEXT:        }
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    cudaEventRecord( start, 0 );
        for (int i=0; i<4; i++) {
            kernel_foo<<<1, 1>>>();
        }
    cudaThreadSynchronize();

    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;
    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
}

__global__ void kernel(float *g_data, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + value;
}

void foo_test_2() {
    int num = 128;
    int nbytes = num * sizeof(int);
    float value = 10.0f;

    float *h_a = 0;
    float *d_a = 0;
    dim3 block = dim3(128);
    dim3 grid  = dim3((num + block.x - 1) / block.x);

    // create cuda event handles
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&stop));

    // asynchronously issue work to the GPU (all to stream 0)
    CHECK(cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    kernel<<<grid, block>>>(d_a, value);
    CHECK(cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

    // CHECK:    q_ct1.wait();
    // CHECK-NEXT:    stop_ct1 = std::chrono::steady_clock::now();
    // CHECK-NEXT:    CHECK(0);
    CHECK(cudaEventRecord(stop));

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }
}

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
    }                                                                          \
  }

#define NSTREAM 4
#define BDIM 128

__global__ void sumArrays(float *A, float *B, float *C, const int NN) {}

void foo_test_3() {
  int nElem = 1 << 18;
  size_t nBytes = nElem * sizeof(float);

  // malloc pinned host memory for async memcpy
  float *h_A, *h_B, *hostRef, *gpuRef;

  // malloc device global memory
  float *d_A, *d_B, *d_C;

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  // invoke kernel at host side
  dim3 block(BDIM);
  dim3 grid;

  // grid parallel operation
  int iElem = nElem / NSTREAM;
  size_t iBytes = iElem * sizeof(float);

  cudaStream_t stream[NSTREAM];

  for (int i = 0; i < NSTREAM; ++i) {
    // CHECK:    CHECK((stream[i] = dpct::get_current_device().create_queue(), 0));
    CHECK(cudaStreamCreate(&stream[i]));
  }

  CHECK(cudaEventRecord(start, 0));

  // initiate all work on the device asynchronously in depth-first order
  for (int i = 0; i < NSTREAM; ++i) {
    int ioffset = i * iElem;
    CHECK(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes,
                          cudaMemcpyHostToDevice, stream[i]));
    CHECK(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes,
                          cudaMemcpyHostToDevice, stream[i]));
    sumArrays<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], &d_B[ioffset],
                                             &d_C[ioffset], iElem);
    // CHECK:    CHECK((dpct::async_dpct_memcpy(&gpuRef[ioffset], &d_C[ioffset], iBytes,
    // CHECK-NEXT:                          dpct::device_to_host, *(stream[i])), 0));
    CHECK(cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes,
                          cudaMemcpyDeviceToHost, stream[i]));
  }

  // CHECK: dpct::dev_mgr::instance().current_device().queues_wait_and_throw();
  // CHECK-NEXT: stop_ct1 = std::chrono::steady_clock::now();
  // CHECK-NEXT: CHECK(0);
  // CHECK-NEXT: CHECK(0);
  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));
  float execution_time;
  CHECK(cudaEventElapsedTime(&execution_time, start, stop));
}

#define SAFE_CALL(call)                                                   \
  do {                                                                         \
    int err = call;                                                            \
  } while (0)

void foo_usm() {
  cudaStream_t s1, s2;
  int *gpu_t, *host_t, n = 10;
  cudaEvent_t start, stop;
  SAFE_CALL(cudaEventRecord(start, 0));

  // CHECK:  DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  SAFE_CALL((dpct::async_dpct_memcpy(gpu_t, host_t, n * sizeof(int), dpct::host_to_device, *s1), 0));
  SAFE_CALL(cudaMemcpyAsync(gpu_t, host_t, n * sizeof(int), cudaMemcpyHostToDevice, s1));

  // CHECK:  s1->wait();
  // CHECK-NEXT:  stop_ct1 = std::chrono::steady_clock::now();
  // CHECK-NEXT:  SAFE_CALL(0);
  // CHECK-NEXT:  SAFE_CALL(0);
  // CHECK-NEXT:  float Time = 0.0f;
  // CHECK-NEXT:  Time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  SAFE_CALL(cudaEventRecord(stop, 0));
  SAFE_CALL(cudaEventSynchronize(stop));
  float Time = 0.0f;
  cudaEventElapsedTime(&Time, start, stop);
}
