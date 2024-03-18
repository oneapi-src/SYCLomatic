// RUN: dpct --enable-profiling  --no-dpcpp-extensions=enqueued_barriers --format-range=none -usm-level=none -out-root %T/tm-nonusm-no-submit-barrier-profiling %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/tm-nonusm-no-submit-barrier-profiling/tm-nonusm-no-submit-barrier-profiling.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST  %T/tm-nonusm-no-submit-barrier-profiling/tm-nonusm-no-submit-barrier-profiling.dp.cpp -o %T/tm-nonusm-no-submit-barrier-profiling/tm-nonusm-no-submit-barrier-profiling.dp.o %}
// RUN: rm -rf %T/tm-nonusm-no-submit-barrier-profiling/

#ifndef BUILD_TEST
// CHECK:#define DPCT_PROFILING_ENABLED
// CHECK-NEXT:#define DPCT_USM_LEVEL_NONE
// CHECK-NEXT:#include <sycl/sycl.hpp>
// CHECK-NEXT:#include <dpct/dpct.hpp>
// CHECK-NEXT:#include <stdio.h>
// CHECK-NEXT:#include <cmath>
#include "cuda.h"
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
    // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
    cudaStream_t stream;

    int ha[N], hb[N];
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

    // CHECK:   dpct::get_current_device().queues_wait_and_throw();
    // CHECK-NEXT:    *start = q_ct1.single_task([=](){});
    // CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    cudaEventRecord(start, 0);

    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice, stream);

    // CHECK:    dpct::get_current_device().queues_wait_and_throw();
    // CHECK-NEXT:    *stop = q_ct1.single_task([=](){});
    // CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    // CHECK-NEXT:     dpct::sycl_event_synchronize(stop);
    // CHECK-NEXT:    elapsedTime = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
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

// CHECK:    dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    *start = q_ct1.single_task([=](){});
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    cudaEventRecord( start, 0 );

        for (int i=0; i<4; i++) {
            kernel_foo<<<1, 1>>>();
        }
    cudaThreadSynchronize();

// CHECK:    dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    *stop = q_ct1.single_task([=](){});
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw() ;
// CHECK-NEXT:    dpct::sycl_event_synchronize( stop ) ;
// CHECK-NEXT:    float   elapsedTime;
// CHECK-NEXT:    elapsedTime = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f ;
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

    // CHECK:    CHECK([](){dpct::get_current_device().queues_wait_and_throw();
    // CHECK-NEXT:    *stop = q_ct1.single_task([=](){});
    // CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    // CHECK-NEXT:    return 0;}());
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

__global__ void sumArrays(float *A, float *B, float *C, const int NN) {
}

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
// CHECK:    CHECK(DPCT_CHECK_ERROR(stream[i] = dev_ct1.create_queue()));
    CHECK(cudaStreamCreate(&stream[i]));
  }

// CHECK:  CHECK([](){dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:  *start = q_ct1.single_task([=](){});
// CHECK-NEXT:  dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:  return 0;}());
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
    //CHECK: CHECK(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(&gpuRef[ioffset], &d_C[ioffset], iBytes,
    //CHECK-NEXT:    dpct::device_to_host, *(stream[i]))));
    CHECK(cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes,
                          cudaMemcpyDeviceToHost, stream[i]));
  }

  // CHECK:  CHECK([](){dpct::get_current_device().queues_wait_and_throw();
  // CHECK-NEXT:  *stop = q_ct1.single_task([=](){});
  // CHECK-NEXT:  dpct::get_current_device().queues_wait_and_throw();
  // CHECK-NEXT:  return 0;}());
  // CHECK-NEXT:  CHECK(DPCT_CHECK_ERROR(dpct::sycl_event_synchronize(stop)));
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

  // CHECK:  dpct::event_ptr start, stop;
  // CHECK-NEXT:  SAFE_CALL([](){dpct::get_current_device().queues_wait_and_throw();
  // CHECK-NEXT:  *start = q_ct1.single_task([=](){});
  // CHECK-NEXT:  dpct::get_current_device().queues_wait_and_throw();
  // CHECK-NEXT:  return 0;}());
  cudaEvent_t start, stop;
  SAFE_CALL(cudaEventRecord(start, 0));

  // CHECK:  SAFE_CALL(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(gpu_t, host_t, n * sizeof(int), dpct::host_to_device, *s1)));
  SAFE_CALL(cudaMemcpyAsync(gpu_t, host_t, n * sizeof(int), cudaMemcpyHostToDevice, s1));

  // CHECK:  SAFE_CALL([](){dpct::get_current_device().queues_wait_and_throw();
  // CHECK-NEXT:  *stop = q_ct1.single_task([=](){});
  // CHECK-NEXT:  dpct::get_current_device().queues_wait_and_throw();
  // CHECK-NEXT:  return 0;}());
  // CHECK-NEXT:  SAFE_CALL(DPCT_CHECK_ERROR(dpct::sycl_event_synchronize(stop)));
  // CHECK-NEXT:  float Time = 0.0f;
  // CHECK-NEXT:  Time = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
  SAFE_CALL(cudaEventRecord(stop, 0));
  SAFE_CALL(cudaEventSynchronize(stop));
  float Time = 0.0f;
  cudaEventElapsedTime(&Time, start, stop);
}

__global__ void readTexels(int n, float *d_out, int width){
}
__global__ void readTexelsFoo1(int n, float *d_out){
}
__global__ void readTexelsFoo2(int n, float *d_out, int width, int height){
}
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

// CHECK:            dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:            *start = q_ct1.single_task([=](){});
// CHECK-NEXT:            dpct::get_current_device().queues_wait_and_throw();
            cudaEventRecord(start, 0);
            // read texels from texture
            for (int iter = 0; iter < iterations; iter++) {
                readTexels<<<gridSize, blockSize>>>(kernelRepFactor, d_out, width);
            }

// CHECK:             dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:            *stop = q_ct1.single_task([=](){});
// CHECK-NEXT:            dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:            dpct::sycl_event_synchronize(stop);
// CHECK-NEXT:            t = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);
            t /= 1.e3;

            // Verify results
            cudaMemcpy(h_out, d_out, numFloat4*sizeof(float),
                    cudaMemcpyDeviceToHost);

            // Test 2 Repeated Cache Access
// CHECK:            dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:            *start = q_ct1.single_task([=](){});
// CHECK-NEXT:            dpct::get_current_device().queues_wait_and_throw();
            cudaEventRecord(start, 0);
            for (int iter = 0; iter < iterations; iter++) {
                readTexelsFoo1<<<gridSize, blockSize>>> (kernelRepFactor, d_out);
            }

// CHECK:            dpct::sycl_event_synchronize(stop);
// CHECK-NEXT:            t = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);

            // Verify results
            cudaMemcpy(h_out, d_out, numFloat4*sizeof(float),
                    cudaMemcpyDeviceToHost);

            // Test 3 Repeated "Random" Access
// CHECK:            dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:            *start = q_ct1.single_task([=](){});
// CHECK-NEXT:            dpct::get_current_device().queues_wait_and_throw();
            cudaEventRecord(start, 0);

            // read texels from texture
            for (int iter = 0; iter < iterations; iter++){
                readTexelsFoo2<<<gridSize, blockSize>>>(kernelRepFactor, d_out, width, height);
            }

// CHECK:            dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:            *stop = q_ct1.single_task([=](){});
// CHECK-NEXT:            dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:            dpct::sycl_event_synchronize(stop);
// CHECK-NEXT:            t = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
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

int foo_test_4()
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
// CHECK:    dpct::event_ptr start, stop;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    cudaEvent_t *kernelEvent;
    kernelEvent = (cudaEvent_t *) malloc(n_streams * sizeof(cudaEvent_t));

    // record start event
// CHECK:    CHECK([](){dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    *start = q_ct1.single_task([=](){});
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    return 0;}());
    CHECK(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
// CHECK:        DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
// CHECK-NEXT:        */
// CHECK-NEXT:        streams[i]->parallel_for<dpct_kernel_name<class foo_kernel_1_{{[a-z0-9]+}}>>(
// CHECK-NEXT:              sycl::nd_range<3>(grid * block, block),
// CHECK-NEXT:              [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                foo_kernel_1();
// CHECK-NEXT:              });
        foo_kernel_1<<<grid, block, 0, streams[i]>>>();
        foo_kernel_2<<<grid, block, 0, streams[i]>>>();
        foo_kernel_3<<<grid, block, 0, streams[i]>>>();
        foo_kernel_4<<<grid, block, 0, streams[i]>>>();

// CHECK:        CHECK([](){dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:        streams[i]->single_task([=](){});
// CHECK-NEXT:        dpct::get_current_device().queues_wait_and_throw()}());
// CHECK-NEXT:        kernelEvent[i]->wait();
        CHECK(cudaEventRecord(kernelEvent[i], streams[i]));
        cudaStreamWaitEvent(streams[n_streams - 1], kernelEvent[i], 0);
    }

// CHECK:    CHECK([](){dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    *stop = q_ct1.single_task([=](){});
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    return 0;}());
// CHECK-NEXT:    CHECK(DPCT_CHECK_ERROR(dpct::sycl_event_synchronize(stop)));
// CHECK-NEXT:    CHECK(DPCT_CHECK_ERROR(elapsed_time = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    return 0;
}

template <class T, int blockSize>
__global__ void
reduce(const T* __restrict__ g_idata, T* __restrict__ g_odata,
       int n) {
}

template <class T, class vecT>
void RunTest()
{
    int probSizes[4] = { 1, 8, 32, 64 };
    int size;
    // Convert to MiB
    size = (size * 1024 * 1024) / sizeof(T);
    // create input data on CPU
    unsigned int bytes = size * sizeof(T);

    // Allocate Host Memory
    T* h_idata;
    T* reference;
    T* h_odata;

    int num_blocks  = 64;
    int num_threads = 256;
    int smem_size = sizeof(T) * num_threads;

    // Allocate device memory
    T* d_idata, *d_odata, *d_block_sums;

    cudaEvent_t start, stop;
    int passes;
    int iters;

    for (int k = 0; k < passes; k++)
    {
        float totalScanTime = 0.0f;
// CHECK:         SAFE_CALL([](){dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:        *start = q_ct1.single_task([=](){});
// CHECK-NEXT:        dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:        return 0;}());
        SAFE_CALL(cudaEventRecord(start, 0));
        for (int j = 0; j < iters; j++)
        {
// CHECK:          q_ct1.submit(
// CHECK-NEXT:            [&](sycl::handler &cgh) {
// CHECK-NEXT:              dpct::access_wrapper<T *> d_idata_acc_ct0(d_idata, cgh);
// CHECK-NEXT:              dpct::access_wrapper<T *> d_block_sums_acc_ct1(d_block_sums, cgh);
// CHECK-EMPTY:
// CHECK-NEXT:              cgh.parallel_for<dpct_kernel_name<class reduce_{{[a-z0-9]+}}, T, dpct_kernel_scalar<256>>>(
// CHECK-NEXT:                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, num_threads), sycl::range<3>(1, 1, num_threads)),
// CHECK-NEXT:                [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                  reduce<T, 256>(d_idata_acc_ct0.get_raw_pointer(), d_block_sums_acc_ct1.get_raw_pointer(), size);
// CHECK-NEXT:                });
// CHECK-NEXT:            });
            reduce<T, 256><<<num_blocks, num_threads, smem_size>>>(d_idata, d_block_sums, size);
        }
        SAFE_CALL(cudaEventRecord(stop, 0));
        SAFE_CALL(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&totalScanTime, start, stop);
    }
    SAFE_CALL(cudaFree(d_idata));
    SAFE_CALL(cudaFree(d_odata));
    SAFE_CALL(cudaFree(d_block_sums));
    SAFE_CALL(cudaFreeHost(h_idata));
    SAFE_CALL(cudaFreeHost(h_odata));
    SAFE_CALL(cudaFreeHost(reference));
    SAFE_CALL(cudaEventDestroy(start));
    SAFE_CALL(cudaEventDestroy(stop));
}

int foo_test_5() {
   RunTest<float, float4>();
}

__global__ void foo_kernel_1(unsigned short* blk_sad, unsigned short* frame,
                            int mb_width, int mb_height,
                            unsigned short* img_ref) {
}
__global__ void foo_kernel_2(unsigned short* blk_sad, int mb_width,
                                  int mb_height) {
}

__global__ void foo_kernel_3(unsigned short* blk_sad, int mb_width,
                                   int mb_height) {
}

void test_1999(void* ref_image, void* cur_image,
                    float* sad_calc_ms, float* sad_calc_8_ms,
                    float* sad_calc_16_ms,
                    unsigned short** h_sads) {
    size_t image_width_macroblocks;
    size_t image_height_macroblocks;
    size_t image_size_macroblocks;
    size_t nsads;
    unsigned short* imgRef = NULL;
    unsigned short* d_cur_image = NULL;
    unsigned short* d_sads = NULL;

// CHECK:     dpct::event_ptr sad_calc_start, sad_calc_stop;
    cudaEvent_t sad_calc_start, sad_calc_stop;
    cudaEventCreate(&sad_calc_start);
    cudaEventCreate(&sad_calc_stop);
// CHECK:    dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    *sad_calc_start = q_ct1.single_task([=](){});
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    cudaEventRecord(sad_calc_start);
    dim3 foo_kernel_1_threads_in_block;
    dim3 foo_kernel_1_blocks_in_grid;

// CHECK:    q_ct1.submit(
// CHECK-NEXT:      [&](sycl::handler &cgh) {
// CHECK-NEXT:        dpct::access_wrapper<unsigned short *> d_sads_acc_ct0(d_sads, cgh);
// CHECK-NEXT:        dpct::access_wrapper<unsigned short *> d_cur_image_acc_ct1(d_cur_image, cgh);
// CHECK-NEXT:        dpct::access_wrapper<unsigned short *> imgRef_acc_ct4(imgRef, cgh);
// CHECK-EMPTY:
// CHECK-NEXT:        cgh.parallel_for<dpct_kernel_name<class foo_kernel_1_{{[a-z0-9]+}}>>(
// CHECK-NEXT:          sycl::nd_range<3>(foo_kernel_1_blocks_in_grid * foo_kernel_1_threads_in_block, foo_kernel_1_threads_in_block),
// CHECK-NEXT:          [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:            foo_kernel_1(d_sads_acc_ct0.get_raw_pointer(), d_cur_image_acc_ct1.get_raw_pointer(), image_width_macroblocks, image_height_macroblocks, imgRef_acc_ct4.get_raw_pointer());
// CHECK-NEXT:          });
// CHECK-NEXT:      });
    foo_kernel_1<<<foo_kernel_1_blocks_in_grid,
                  foo_kernel_1_threads_in_block>>>(d_sads, d_cur_image,
                                                  image_width_macroblocks,
                                                  image_height_macroblocks,
                                                  imgRef);

// CHECK:    dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    *sad_calc_stop = q_ct1.single_task([=](){});
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    cudaEventRecord(sad_calc_stop);

// CHECK:    dpct::event_ptr sad_calc_8_start, sad_calc_8_stop;
    cudaEvent_t sad_calc_8_start, sad_calc_8_stop;

    cudaEventCreate(&sad_calc_8_start);
    cudaEventCreate(&sad_calc_8_stop);
// CHECK:        dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    *sad_calc_8_start = q_ct1.single_task([=](){});
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    cudaEventRecord(sad_calc_8_start);
    dim3 foo_kernel_2_threads_in_block;
    dim3 foo_kernel_2_blocks_in_grid;

// CHECK:    q_ct1.submit(
// CHECK-NEXT:      [&](sycl::handler &cgh) {
// CHECK-NEXT:        dpct::access_wrapper<unsigned short *> d_sads_acc_ct0(d_sads, cgh);
// CHECK-EMPTY:
// CHECK-NEXT:        cgh.parallel_for<dpct_kernel_name<class foo_kernel_2_{{[a-z0-9]+}}>>(
// CHECK-NEXT:          sycl::nd_range<3>(foo_kernel_2_blocks_in_grid * foo_kernel_2_threads_in_block, foo_kernel_2_threads_in_block),
// CHECK-NEXT:          [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:            foo_kernel_2(d_sads_acc_ct0.get_raw_pointer(), image_width_macroblocks, image_height_macroblocks);
// CHECK-NEXT:          });
// CHECK-NEXT:      });
    foo_kernel_2<<<
      foo_kernel_2_blocks_in_grid,
      foo_kernel_2_threads_in_block>>>(d_sads, image_width_macroblocks,
                                            image_height_macroblocks);
// CHECK:    dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    *sad_calc_8_stop = q_ct1.single_task([=](){});
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    cudaEventRecord(sad_calc_8_stop);


// CHECK:    dpct::event_ptr sad_calc_16_start, sad_calc_16_stop;
    cudaEvent_t sad_calc_16_start, sad_calc_16_stop;

    cudaEventCreate(&sad_calc_16_start);
    cudaEventCreate(&sad_calc_16_stop);
// CHECK:        dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    *sad_calc_16_start = q_ct1.single_task([=](){});
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    cudaEventRecord(sad_calc_16_start);
    dim3 foo_kernel_3_threads_in_block;
    dim3 foo_kernel_3_blocks_in_grid;

// CHECK:    q_ct1.submit(
// CHECK-NEXT:      [&](sycl::handler &cgh) {
// CHECK-NEXT:        dpct::access_wrapper<unsigned short *> d_sads_acc_ct0(d_sads, cgh);
// CHECK-EMPTY:
// CHECK-NEXT:        cgh.parallel_for<dpct_kernel_name<class foo_kernel_3_{{[a-z0-9]+}}>>(
// CHECK-NEXT:          sycl::nd_range<3>(foo_kernel_3_blocks_in_grid * foo_kernel_3_threads_in_block, foo_kernel_3_threads_in_block),
// CHECK-NEXT:          [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:            foo_kernel_3(d_sads_acc_ct0.get_raw_pointer(), image_width_macroblocks, image_height_macroblocks);
// CHECK-NEXT:          });
// CHECK-NEXT:      });
    foo_kernel_3<<<
      foo_kernel_3_blocks_in_grid,
      foo_kernel_3_threads_in_block>>>(d_sads, image_width_macroblocks,
                                             image_height_macroblocks);
// CHECK:    dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    *sad_calc_16_stop = q_ct1.single_task([=](){});
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    cudaEventRecord(sad_calc_16_stop);

    cudaMallocHost((void **)h_sads, nsads * sizeof(unsigned short));
    cudaMemcpy(*h_sads, d_sads, nsads * sizeof(*d_sads), cudaMemcpyDeviceToHost);
    cudaFree(d_sads);
    cudaFree(d_cur_image);
    cudaFree(imgRef);

// CHECK:    *(sad_calc_ms) = (sad_calc_stop->get_profiling_info<sycl::info::event_profiling::command_end>() - sad_calc_start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
// CHECK-NEXT:    *(sad_calc_8_ms) = (sad_calc_8_stop->get_profiling_info<sycl::info::event_profiling::command_end>() - sad_calc_8_start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
// CHECK-NEXT:    *(sad_calc_16_ms) = (sad_calc_16_stop->get_profiling_info<sycl::info::event_profiling::command_end>() - sad_calc_16_start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
    cudaEventElapsedTime(sad_calc_ms, sad_calc_start, sad_calc_stop);
    cudaEventElapsedTime(sad_calc_8_ms, sad_calc_8_start, sad_calc_8_stop);
    cudaEventElapsedTime(sad_calc_16_ms, sad_calc_16_start, sad_calc_16_stop);
}
#endif
