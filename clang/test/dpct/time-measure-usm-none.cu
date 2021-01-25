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

__global__ void readTexels(int n, float *d_out, int width){}
__global__ void readTexelsFoo1(int n, float *d_out){}
__global__ void readTexelsFoo2(int n, float *d_out, int width, int height){}
texture<float4, 2, cudaReadModeElementType> texA;

void foo()
{
    const unsigned int passes = 100;
    const unsigned int nsizes = 5;
    const unsigned int sizes[] = { 16, 64, 256, 1024, 4096 };
    const unsigned int kernelRepFoo[] = { 1024, 1024, 1024, 1024, 256 };
    const unsigned int iterations = 10;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int j = 0; j < nsizes; j++)
    {
        const unsigned int size      = 1024 * sizes[j];
        const unsigned int numFloat  = size / sizeof(float);
        const unsigned int numFloat4 = size / sizeof(float4);
        size_t width, height;
        const unsigned int kernelRepFactor = kernelRepFoo[j];

        // Image memory sizes should be power of 2.
        size_t sizeLog = lround(log2(double(numFloat4)));
        height = 1 << (sizeLog >> 1);  // height is the smaller size
        width = numFloat4 / height;

        const dim3 blockSize(16, 8);
        const dim3 gridSize(width/blockSize.x, height/blockSize.y);

        float *h_in = new float[numFloat];
        float *h_out = new float[numFloat4];
        float *d_out;
        cudaMalloc((void**) &d_out, numFloat4 * sizeof(float));

        // Allocate a cuda array
        cudaArray* cuArray;
        cudaMallocArray(&cuArray, &texA.channelDesc, width, height);

        // Copy in source data
        cudaMemcpyToArray(cuArray, 0, 0, h_in, size, cudaMemcpyHostToDevice);

        // Bind texture to the array
        cudaBindTextureToArray(texA, cuArray);

        for (int p = 0; p < passes; p++)
        {
            // Test 1: Repeated Linear Access
            float t = 0.0f;

            cudaEventRecord(start, 0);
            // read texels from texture
            for (int iter = 0; iter < iterations; iter++)
            {
// CHECK:                DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:                */
// CHECK-NEXT:                  dpct::buffer_t d_out_buf_ct1 = dpct::get_buffer(d_out);
// CHECK-NEXT:                  q_ct1.submit(
// CHECK-NEXT:                    [&](sycl::handler &cgh) {
// CHECK-NEXT:                      auto d_out_acc_ct1 = d_out_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
// CHECK-EMPTY:
// CHECK-NEXT:                      cgh.parallel_for<dpct_kernel_name<class readTexels_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                        sycl::nd_range<3>(gridSize * blockSize, blockSize), 
// CHECK-NEXT:                        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                          readTexels(kernelRepFactor, (float *)(&d_out_acc_ct1[0]), width);
// CHECK-NEXT:                        });
// CHECK-NEXT:                    });
                readTexels<<<gridSize, blockSize>>>(kernelRepFactor, d_out,
                                                    width);
            }

// CHECK:            DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:            */
// CHECK-NEXT:            dpct::dev_mgr::instance().current_device().queues_wait_and_throw();
// CHECK-NEXT:            stop_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:            t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);
            t /= 1.e3;

            // Verify results
            cudaMemcpy(h_out, d_out, numFloat4*sizeof(float),
                    cudaMemcpyDeviceToHost);

            // Test 2 Repeated Cache Access
            cudaEventRecord(start, 0);
            for (int iter = 0; iter < iterations; iter++)
            {
// CHECK:                DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:                */
// CHECK-NEXT:                  dpct::buffer_t d_out_buf_ct1 = dpct::get_buffer(d_out);
// CHECK-NEXT:                  q_ct1.submit(
// CHECK-NEXT:                    [&](sycl::handler &cgh) {
// CHECK-NEXT:                      auto d_out_acc_ct1 = d_out_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
// CHECK-EMPTY:
// CHECK-NEXT:                      cgh.parallel_for<dpct_kernel_name<class readTexelsFoo1_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                        sycl::nd_range<3>(gridSize * blockSize, blockSize), 
// CHECK-NEXT:                        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                          readTexelsFoo1(kernelRepFactor, (float *)(&d_out_acc_ct1[0]));
// CHECK-NEXT:                        });
// CHECK-NEXT:                    });
                readTexelsFoo1<<<gridSize, blockSize>>>
                        (kernelRepFactor, d_out);
            }

// CHECK:            DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:             */
// CHECK-NEXT:             dpct::dev_mgr::instance().current_device().queues_wait_and_throw();
// CHECK-NEXT:             stop_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:             t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);

            // Verify results
            cudaMemcpy(h_out, d_out, numFloat4*sizeof(float),
                    cudaMemcpyDeviceToHost);

            // Test 3 Repeated "Random" Access
            cudaEventRecord(start, 0);

            // read texels from texture
            for (int iter = 0; iter < iterations; iter++)
            {
// CHECK:                DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:                */
// CHECK-NEXT:                  dpct::buffer_t d_out_buf_ct1 = dpct::get_buffer(d_out);
// CHECK-NEXT:                  q_ct1.submit(
// CHECK-NEXT:                    [&](sycl::handler &cgh) {
// CHECK-NEXT:                      auto d_out_acc_ct1 = d_out_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
// CHECK-EMPTY:
// CHECK-NEXT:                      cgh.parallel_for<dpct_kernel_name<class readTexelsFoo2_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                        sycl::nd_range<3>(gridSize * blockSize, blockSize), 
// CHECK-NEXT:                        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                          readTexelsFoo2(kernelRepFactor, (float *)(&d_out_acc_ct1[0]), width, height);
// CHECK-NEXT:                        });
// CHECK-NEXT:                    });
                readTexelsFoo2<<<gridSize, blockSize>>>
                                (kernelRepFactor, d_out, width, height);
            }

// CHECK:             DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:            */
// CHECK-NEXT:            dpct::dev_mgr::instance().current_device().queues_wait_and_throw();
// CHECK-NEXT:            stop_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:            t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);
        }
        delete[] h_in;
        delete[] h_out;
        cudaFree(d_out);
        cudaFreeArray(cuArray);
        cudaUnbindTexture(texA);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



__global__ void foo_kernel_1(){}
__global__ void foo_kernel_2(){}
__global__ void foo_kernel_3(){}
__global__ void foo_kernel_4(){}

int foo_test_2()
{
    int n_streams = NSTREAM;
    int isize = 1;
    int iblock = 1;
    float elapsed_time;

    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams*sizeof(cudaStream_t));

    for (int i = 0 ; i < n_streams ; i++)
    {
        cudaStreamCreate(&(streams[i]));
    }

    dim3 block (iblock);
    dim3 grid  (isize / iblock);

    // creat events
// CHECK:    sycl::event start, stop;
// CHECK-NEXT:    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
// CHECK-NEXT:    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0, because this call is redundant in DPC++.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK(0);
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0, because this call is redundant in DPC++.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK(0);
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    cudaEvent_t *kernelEvent;
    kernelEvent = (cudaEvent_t *) malloc(n_streams * sizeof(cudaEvent_t));

    // record start event
// CHECK:    /*
// CHECK-NEXT:    DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:    */
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1024:{{[0-9]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT:    */
// CHECK-NEXT:    start_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:    CHECK(0);
    CHECK(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
// CHECK:        DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:        */
// CHECK-NEXT:        streams[i]->submit(
// CHECK-NEXT:          [&](sycl::handler &cgh) {
// CHECK-NEXT:            cgh.parallel_for<dpct_kernel_name<class foo_kernel_1_{{[a-z0-9]+}}>>(
// CHECK-NEXT:              sycl::nd_range<3>(grid * block, block), 
// CHECK-NEXT:              [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                foo_kernel_1();
// CHECK-NEXT:              });
// CHECK-NEXT:          });
        foo_kernel_1<<<grid, block, 0, streams[i]>>>();
        foo_kernel_2<<<grid, block, 0, streams[i]>>>();
        foo_kernel_3<<<grid, block, 0, streams[i]>>>();
        foo_kernel_4<<<grid, block, 0, streams[i]>>>();

// CHECK:        kernelEvent_ct1_i = std::chrono::steady_clock::now(); 
// CHECK-NEXT:        CHECK(0);
// CHECK-NEXT:        kernelEvent[i].wait();
        CHECK(cudaEventRecord(kernelEvent[i], streams[i]));
        cudaStreamWaitEvent(streams[n_streams - 1], kernelEvent[i], 0);
    }

// CHECK:    dpct::dev_mgr::instance().current_device().queues_wait_and_throw();
// CHECK-NEXT:    stop_ct1 = std::chrono::steady_clock::now(); 
// CHECK-NEXT:    CHECK(0);
// CHECK-NEXT:    CHECK(0);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    return 0;
}
