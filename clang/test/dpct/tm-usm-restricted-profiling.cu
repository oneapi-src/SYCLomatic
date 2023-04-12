// RUN: dpct --enable-profiling  --format-range=none -out-root %T/tm-usm-restricted-profiling %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/tm-usm-restricted-profiling/tm-usm-restricted-profiling.dp.cpp --match-full-lines %s

// CHECK:#define DPCT_PROFILING_ENABLED
// CHECK-NEXT:#include <sycl/sycl.hpp>
// CHECK-NEXT:#include <dpct/dpct.hpp>
// CHECK-NEXT:#include <stdio.h>
// CHECK-NEXT:#include <cmath>
#include <stdio.h>

#define N 1000

__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = 2*a[i];
    }
}

int main() {
    cudaStream_t stream;

    int ha[N], hb[N];
  // CHECK: dpct::event_ptr start, stop;
    cudaEvent_t start, stop;
    cudaError_t cudaStatus;

    int *da, *db;
    float elapsedTime;

    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));

    for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }


  // CHECK:    start = new sycl::event();
  // CHECK-NEXT:    stop = new sycl::event();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

 // CHECK:    *start = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(start, 0);

  // CHECK: q_ct1.memcpy(da, ha, N*sizeof(int));
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);
  // CHECK: q_ct1.memcpy(da, ha, N*sizeof(int));
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice, 0);
  // CHECK: stream->memcpy(da, ha, N*sizeof(int));
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice, stream);

  // CHECK:    *stop = q_ct1.ext_oneapi_submit_barrier();
  // CHECK-NEXT:    stop->wait_and_throw();
  // CHECK-NEXT:   elapsedTime = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    add<<<N, 1>>>(da, db);

  // CHECK: q_ct1.memcpy(hb, db, N*sizeof(int));
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

#define SAFE_CALL(call)                                                   \
  do {                                                                         \
    cudaError err = call;                                                            \
  } while (0)

void foo_usm() {
  cudaStream_t s1, s2;
  int *gpu_t, *host_t, n = 10;
  cudaEvent_t start, stop;

// CHECK:  SAFE_CALL((*start = q_ct1.ext_oneapi_submit_barrier(), 0));
  SAFE_CALL(cudaEventRecord(start, 0));

// CHECK:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:  */
// CHECK-NEXT:SAFE_CALL((s1->memcpy(gpu_t, host_t, n * sizeof(int)), 0));
  SAFE_CALL(cudaMemcpyAsync(gpu_t, host_t, n * sizeof(int), cudaMemcpyHostToDevice, s1));

// CHECK:  /*
// CHECK-NEXT:  DPCT1024:{{[0-9]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT:  */
// CHECK-NEXT:  SAFE_CALL((*stop = q_ct1.ext_oneapi_submit_barrier(), 0));
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:  */
// CHECK-NEXT:   SAFE_CALL((stop->wait_and_throw(), 0));
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

// CHECK:            *start = q_ct1.ext_oneapi_submit_barrier();
            cudaEventRecord(start, 0);
            // read texels from texture
            for (int iter = 0; iter < iterations; iter++)
            {
// CHECK:                 DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
// CHECK-NEXT:                 */
// CHECK-NEXT:                q_ct1.parallel_for<dpct_kernel_name<class readTexels_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                      sycl::nd_range<3>(gridSize * blockSize, blockSize),
// CHECK-NEXT:                      [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                        readTexels(kernelRepFactor, d_out, width);
// CHECK-NEXT:                      });
                readTexels<<<gridSize, blockSize>>>(kernelRepFactor, d_out,
                                                    width);
            }

// CHECK:             *stop = q_ct1.ext_oneapi_submit_barrier();
// CHECK-NEXT:             stop->wait_and_throw();
// CHECK-NEXT:             t = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);

            // Verify results
            cudaMemcpy(h_out, d_out, numFloat4*sizeof(float),
                    cudaMemcpyDeviceToHost);

            // Test 2 Repeated Cache Access
// CHECK:            *start = q_ct1.ext_oneapi_submit_barrier();
            cudaEventRecord(start, 0);
            for (int iter = 0; iter < iterations; iter++)
            {

// CHECK:                DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
// CHECK-NEXT:                */
// CHECK-NEXT:                q_ct1.parallel_for<dpct_kernel_name<class readTexelsFoo1_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                      sycl::nd_range<3>(gridSize * blockSize, blockSize),
// CHECK-NEXT:                      [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                        readTexelsFoo1(kernelRepFactor, d_out);
// CHECK-NEXT:                      });
                readTexelsFoo1<<<gridSize, blockSize>>>
                        (kernelRepFactor, d_out);
            }

// CHECK:            *stop = q_ct1.ext_oneapi_submit_barrier();
// CHECK-NEXT:            stop->wait_and_throw();
// CHECK-NEXT:            t = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);

            // Verify results
            cudaMemcpy(h_out, d_out, numFloat4*sizeof(float),
                    cudaMemcpyDeviceToHost);

            // Test 3 Repeated "Random" Access
// CHECK:            *start = q_ct1.ext_oneapi_submit_barrier();
            cudaEventRecord(start, 0);

            // read texels from texture
            for (int iter = 0; iter < iterations; iter++)
            {

// CHECK:                DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
// CHECK-NEXT:                */
// CHECK-NEXT:                q_ct1.parallel_for<dpct_kernel_name<class readTexelsFoo2_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                      sycl::nd_range<3>(gridSize * blockSize, blockSize),
// CHECK-NEXT:                      [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                        readTexelsFoo2(kernelRepFactor, d_out, width, height);
// CHECK-NEXT:                      });
                readTexelsFoo2<<<gridSize, blockSize>>>
                                (kernelRepFactor, d_out, width, height);
            }

// CHECK:            *stop = q_ct1.ext_oneapi_submit_barrier();
// CHECK-NEXT:            stop->wait_and_throw();
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


__global__ void kernelFunc(){}

void barr(int maxCalls) {
  cudaEvent_t evtStart[maxCalls];
  cudaEvent_t evtEnd[maxCalls];
  float time[maxCalls];
  for (int i = 0; i < maxCalls; i++) {
    cudaEventCreate( &(evtStart[i]) );
    cudaEventCreate( &(evtEnd[i]) );
    time[i] = 0.0;
  }

// CHECK: *evtStart[0] = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord( evtStart[0], 0 );
  kernelFunc<<<1, 1>>>();
// CHECK:   *evtEnd[0] = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord( evtEnd[0], 0 );

// CHECK: *evtStart[1] = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord( evtStart[1], 0 );

  kernelFunc<<<1, 1>>>();
// CHECK: *evtEnd[1] = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord( evtEnd[1], 0 );

// CHECK: *evtStart[2] = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord( evtStart[2], 0 );
  kernelFunc<<<1, 1>>>();
// CHECK: *evtEnd[2] = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord( evtEnd[2], 0 );

// CHECK: dev_ct1.queues_wait_and_throw();
  cudaDeviceSynchronize();

  float total;
  int i=0;
  cudaEventElapsedTime( &(time[i]), evtStart[i], evtEnd[i]);
  float timesum = 0.0f;
  for (int i = 1; i < maxCalls; i++) {
    cudaEventElapsedTime( &(time[i]), evtStart[i], evtEnd[i]);
    timesum += time[i];
  }
  cudaEventElapsedTime( &total, evtStart[1], evtEnd[maxCalls-1]);
}

template <class T, int blockSize>
__global__ void
reduce(const T* __restrict__ g_idata, T* __restrict__ g_odata,
       int n) {}

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
        SAFE_CALL(cudaEventRecord(start, 0));
        for (int j = 0; j < iters; j++)
        {
// CHECK:            q_ct1.parallel_for<dpct_kernel_name<class reduce_{{[a-z0-9]+}}, T, dpct_kernel_scalar<256>>>(
// CHECK-NEXT:                  sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, num_threads), sycl::range<3>(1, 1, num_threads)),
// CHECK-NEXT:                  [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                    reduce<T, 256>(d_idata, d_block_sums, size);
// CHECK-NEXT:                  });
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
                            unsigned short* img_ref) {}

__global__ void foo_kernel_2(unsigned short* blk_sad, int mb_width,
                                  int mb_height) {}

__global__ void foo_kernel_3(unsigned short* blk_sad, int mb_width,
                                   int mb_height) {}

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
// CHECK:    *sad_calc_start = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(sad_calc_start);
    dim3 foo_kernel_1_threads_in_block;
    dim3 foo_kernel_1_blocks_in_grid;

// CHECK:    q_ct1.parallel_for<dpct_kernel_name<class foo_kernel_1_{{[a-z0-9]+}}>>(
// CHECK-NEXT:          sycl::nd_range<3>(foo_kernel_1_blocks_in_grid * foo_kernel_1_threads_in_block, foo_kernel_1_threads_in_block),
// CHECK-NEXT:          [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:            foo_kernel_1(d_sads, d_cur_image, image_width_macroblocks, image_height_macroblocks, imgRef);
// CHECK-NEXT:          });
    foo_kernel_1<<<foo_kernel_1_blocks_in_grid,
                  foo_kernel_1_threads_in_block>>>(d_sads, d_cur_image,
                                                  image_width_macroblocks,
                                                  image_height_macroblocks,
                                                  imgRef);

// CHECK:    *sad_calc_stop = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(sad_calc_stop);

// CHECK:    dpct::event_ptr sad_calc_8_start, sad_calc_8_stop;
    cudaEvent_t sad_calc_8_start, sad_calc_8_stop;

    cudaEventCreate(&sad_calc_8_start);
    cudaEventCreate(&sad_calc_8_stop);
// CHECK:    *sad_calc_8_start = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(sad_calc_8_start);
    dim3 foo_kernel_2_threads_in_block;
    dim3 foo_kernel_2_blocks_in_grid;

// CHECK:    q_ct1.parallel_for<dpct_kernel_name<class foo_kernel_2_{{[a-z0-9]+}}>>(
// CHECK-NEXT:          sycl::nd_range<3>(foo_kernel_2_blocks_in_grid * foo_kernel_2_threads_in_block, foo_kernel_2_threads_in_block),
// CHECK-NEXT:          [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:            foo_kernel_2(d_sads, image_width_macroblocks, image_height_macroblocks);
// CHECK-NEXT:          });
    foo_kernel_2<<<
      foo_kernel_2_blocks_in_grid,
      foo_kernel_2_threads_in_block>>>(d_sads, image_width_macroblocks,
                                            image_height_macroblocks);
// CHECK:    *sad_calc_8_stop = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(sad_calc_8_stop);


// CHECK:    dpct::event_ptr sad_calc_16_start, sad_calc_16_stop;
    cudaEvent_t sad_calc_16_start, sad_calc_16_stop;

    cudaEventCreate(&sad_calc_16_start);
    cudaEventCreate(&sad_calc_16_stop);
    cudaEventRecord(sad_calc_16_start);
    dim3 foo_kernel_3_threads_in_block;
    dim3 foo_kernel_3_blocks_in_grid;

// CHECK:    q_ct1.parallel_for<dpct_kernel_name<class foo_kernel_3_{{[a-z0-9]+}}>>(
// CHECK-NEXT:          sycl::nd_range<3>(foo_kernel_3_blocks_in_grid * foo_kernel_3_threads_in_block, foo_kernel_3_threads_in_block),
// CHECK-NEXT:          [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:            foo_kernel_3(d_sads, image_width_macroblocks, image_height_macroblocks);
// CHECK-NEXT:          });
    foo_kernel_3<<<
      foo_kernel_3_blocks_in_grid,
      foo_kernel_3_threads_in_block>>>(d_sads, image_width_macroblocks,
                                             image_height_macroblocks);
// CHECK:    *sad_calc_16_stop = q_ct1.ext_oneapi_submit_barrier();
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

__global__ void kernel() {}
void foo_test_1983() {
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  cudaEvent_t event1, event2;
  cudaEventCreate(&event1);
  cudaEventCreate(&event2);
  int repeat = 2;

  for (int i = 0; i < repeat; i++) {
    kernel<<<1, 1, 0, stream1>>>();
// CHECK:    *event1 = stream1->ext_oneapi_submit_barrier();
    cudaEventRecord(event1, stream1);
    kernel<<<1, 1, 0, stream2>>>();

// CHECK:    *event2 = stream2->ext_oneapi_submit_barrier();
// CHECK-NEXT:    event1->wait_and_throw();
// CHECK-NEXT:    event2->wait_and_throw();
    cudaEventRecord(event2, stream2);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
  }
}

template <class T, class vecT> void foo_test_2131();
int foo_test_2131_host() { foo_test_2131<float, float4>(); }

template <class T, class vecT> void foo_test_2131() {
  int size;
  int num_blocks = 64;
  int num_threads = 256;
  int smem_size = sizeof(T) * num_threads;

  // Allocate device memory
  T *d_idata, *d_odata, *d_block_sums;
  cudaEvent_t start, stop;
  int passes;
  int iters;

  for (int k = 0; k < passes; k++) {
    float totalScanTime = 0.0f;
  // CHECK:     SAFE_CALL((*start = q_ct1.ext_oneapi_submit_barrier(), 0));
    SAFE_CALL(cudaEventRecord(start, 0));
    for (int j = 0; j < iters; j++) {
      reduce<T, 256>
          <<<num_blocks, num_threads, smem_size>>>(d_idata, d_block_sums, size);
    }

  // CHECK: SAFE_CALL((*stop = q_ct1.ext_oneapi_submit_barrier(), 0));
  // CHECK: SAFE_CALL((stop->wait_and_throw(), 0));
  // CHECK: totalScanTime = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
    SAFE_CALL(cudaEventRecord(stop, 0));
    SAFE_CALL(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&totalScanTime, start, stop);
  }
}
