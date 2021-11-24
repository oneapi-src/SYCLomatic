// RUN: dpct -out-root %T/time-measure-complex %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/time-measure-complex/time-measure-complex.dp.cpp --match-full-lines %s
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 300
#define NSTREAM 4

__global__ void kernel_1()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_3()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_4()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

int main(int argc, char **argv)
{
    int n_streams = NSTREAM;
    int isize = 1;
    int iblock = 1;
    int bigcase = 0;

    // get argument from command line
    if (argc > 1) n_streams = atoi(argv[1]);

    if (argc > 2) bigcase = atoi(argv[2]);

    float elapsed_time;

    // set up max connectioin
    char * iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv (iname, "32", 1);
    char *ivalue =  getenv (iname);
    printf ("%s = %s\n", iname, ivalue);

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("> Using Device %d: %s with num_streams %d\n", dev, deviceProp.name,
           n_streams);
    cudaSetDevice(dev);

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
    {
        if (deviceProp.concurrentKernels == 0)
        {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                    "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else
        {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // Allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(
                                cudaStream_t));

    for (int i = 0 ; i < n_streams ; i++)
    {
        cudaStreamCreate(&(streams[i]));
    }

    // run kernel with more threads
    if (bigcase == 1)
    {
        iblock = 512;
        isize = 1 << 12;
    }

    // set up execution configuration
    dim3 block (iblock);
    dim3 grid  (isize / iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEvent_t *kernelEvent;
    kernelEvent = (cudaEvent_t *) malloc(n_streams * sizeof(cudaEvent_t));

    for (int i = 0; i < n_streams; i++)
    {
        cudaEventCreateWithFlags(&(kernelEvent[i]),
                    cudaEventDisableTiming);
    }

    // record start event
    // CHECK: start_ct1 = std::chrono::steady_clock::now();
    // CHECK:     start = dpct::get_default_queue().ext_oneapi_submit_barrier();
    cudaEventRecord(start, 0);

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
        // CHECK: streams[i]->parallel_for<dpct_kernel_name<class kernel_1_{{[a-z0-9]+}}>>(
        // CHECK-NEXT:             sycl::nd_range<3>(grid * block, block),
        // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
        // CHECK-NEXT:                 kernel_1();
        // CHECK-NEXT:             });
        kernel_1<<<grid, block, 0, streams[i]>>>();
        // CHECK: streams[i]->parallel_for<dpct_kernel_name<class kernel_2_{{[a-z0-9]+}}>>(
        // CHECK-NEXT:             sycl::nd_range<3>(grid * block, block),
        // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
        // CHECK-NEXT:                 kernel_2();
        // CHECK-NEXT:             });
        kernel_2<<<grid, block, 0, streams[i]>>>();
        // CHECK: streams[i]->parallel_for<dpct_kernel_name<class kernel_3_{{[a-z0-9]+}}>>(
        // CHECK-NEXT:             sycl::nd_range<3>(grid * block, block),
        // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
        // CHECK-NEXT:                 kernel_3();
        // CHECK-NEXT:             });
        kernel_3<<<grid, block, 0, streams[i]>>>();
        // CHECK: streams[i]->parallel_for<dpct_kernel_name<class kernel_4_{{[a-z0-9]+}}>>(
        // CHECK-NEXT:             sycl::nd_range<3>(grid * block, block),
        // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
        // CHECK-NEXT:                 kernel_4();
        // CHECK-NEXT:             });
        kernel_4<<<grid, block, 0, streams[i]>>>();

        // CHECK: kernelEvent_ct1_i = std::chrono::steady_clock::now();
        // CHECK-NEXT:        kernelEvent[i] = streams[i]->ext_oneapi_submit_barrier();
        cudaEventRecord(kernelEvent[i], streams[i]);
        // CHECK: kernelEvent[i] = streams[n_streams - 1]->ext_oneapi_submit_barrier({kernelEvent[i]});
        cudaStreamWaitEvent(streams[n_streams - 1], kernelEvent[i], 0);
    }

    // record stop event
    // CHECK:    dpct::get_current_device().queues_wait_and_throw();
    // CHECK-NEXT:    stop_ct1 = std::chrono::steady_clock::now();
    // CHECK-NEXT:    stop = dpct::get_default_queue().ext_oneapi_submit_barrier();
    // CHECK-NEXT:    elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);


    // release all stream
    for (int i = 0 ; i < n_streams ; i++)
    {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(kernelEvent[i]);
    }

    free(streams);
    free(kernelEvent);

    // reset device
    cudaDeviceReset();

    return 0;
}

#define CHECK_FOO(call)                                                   \
  do {                                                                         \
    int err = call;                                                            \
  } while (0)

template <typename T>
// CHECK: void my_error_checker(T ReturnValue, char const *const FuncName) {
void my_error_checker(T ReturnValue, char const *const FuncName) {
}

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)

#define MY_CHECKER(CALL) do {                           \
  cudaError_t Error = CALL;                             \
  if (Error != cudaSuccess) {                           \
  }                                                     \
} while(0)


void foo_test_1() {
  float et;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  CHECK_FOO(cudaEventRecord(start));
// CHECK:    stop = dpct::get_default_queue().parallel_for<dpct_kernel_name<class kernel_1_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:                [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                    kernel_1();
// CHECK-NEXT:                });
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:  */
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1024:{{[0-9]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT:  */
// CHECK-NEXT:  stop.wait();
// CHECK-NEXT:  stop_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:  CHECK_FOO(0);
// CHECK-NEXT:  MY_ERROR_CHECKER(0);
// CHECK-NEXT:  MY_CHECKER(0);
// CHECK-NEXT:  int ret;
// CHECK-NEXT:  ret = (0);
// CHECK-NEXT:  int a = (0);
// CHECK-NEXT:  et = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  kernel_1<<<1, 1>>>();
  CHECK_FOO(cudaEventRecord(stop));
  cudaEventSynchronize(stop);
  MY_ERROR_CHECKER(cudaEventSynchronize(stop));
  MY_CHECKER(cudaEventSynchronize(stop));
  (cudaEventSynchronize(stop));
  int ret;
  ret = (cudaEventSynchronize(stop));
  int a = (cudaEventSynchronize(stop));
  cudaEventElapsedTime(&et, start, stop);
}

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

// CHECK:    sycl::queue **streams = (sycl::queue **)malloc(n_streams * sizeof(
// CHECK-NEXT:                                                                   sycl::queue *));
    cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(
                                cudaStream_t));

    dim3 block (iblock);
    dim3 grid  (isize / iblock);

    // creat events
// CHECK:    sycl::event start, stop;
// CHECK-NEXT:    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
// CHECK-NEXT:    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0 because this call is redundant in DPC++.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK(0);
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0 because this call is redundant in DPC++.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK(0);
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

// CHECK:    sycl::event *kernelEvent;
// CHECK-NEXT:    std::chrono::time_point<std::chrono::steady_clock> kernelEvent_ct1_i;
// CHECK-NEXT:    kernelEvent = new sycl::event[n_streams];
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
// CHECK-NEXT:    CHECK((start = dpct::get_default_queue().ext_oneapi_submit_barrier(), 0));
    CHECK(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
// CHECK:        /*
// CHECK-NEXT:        DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:        */
// CHECK-NEXT:        streams[i]->parallel_for<dpct_kernel_name<class foo_kernel_1_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                    sycl::nd_range<3>(grid * block, block),
// CHECK-NEXT:                    [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                        foo_kernel_1();
// CHECK-NEXT:                    });
        foo_kernel_1<<<grid, block, 0, streams[i]>>>();

// CHECK:        /*
// CHECK-NEXT:        DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:        */
// CHECK-NEXT:        streams[i]->parallel_for<dpct_kernel_name<class foo_kernel_2_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                    sycl::nd_range<3>(grid * block, block),
// CHECK-NEXT:                    [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                        foo_kernel_2();
// CHECK-NEXT:                    });

        foo_kernel_2<<<grid, block, 0, streams[i]>>>();
// CHECK:        /*
// CHECK-NEXT:        DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:        */
// CHECK-NEXT:        streams[i]->parallel_for<dpct_kernel_name<class foo_kernel_3_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                    sycl::nd_range<3>(grid * block, block),
// CHECK-NEXT:                    [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                        foo_kernel_3();
// CHECK-NEXT:                    });

        foo_kernel_3<<<grid, block, 0, streams[i]>>>();
// CHECK:        /*
// CHECK-NEXT:        DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:        */
// CHECK-NEXT:        streams[i]->parallel_for<dpct_kernel_name<class foo_kernel_4_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                    sycl::nd_range<3>(grid * block, block),
// CHECK-NEXT:                    [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                        foo_kernel_4();
// CHECK-NEXT:                    });

        foo_kernel_4<<<grid, block, 0, streams[i]>>>();

// CHECK:        /*
// CHECK-NEXT:        DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:        */
// CHECK-NEXT:        /*
// CHECK-NEXT:        DPCT1024:{{[0-9]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT:        */
// CHECK-NEXT:        kernelEvent_ct1_i = std::chrono::steady_clock::now();
// CHECK-NEXT:         CHECK((kernelEvent[i] = streams[i]->ext_oneapi_submit_barrier(), 0));
// CHECK-NEXT:         kernelEvent[i] = streams[n_streams - 1]->ext_oneapi_submit_barrier({kernelEvent[i]});
        CHECK(cudaEventRecord(kernelEvent[i], streams[i]));
        cudaStreamWaitEvent(streams[n_streams - 1], kernelEvent[i], 0);
    }

// CHECK:    /*
// CHECK-NEXT:    DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:    */
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1024:{{[0-9]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT:    */
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    stop_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:    CHECK((stop = dpct::get_default_queue().ext_oneapi_submit_barrier(), 0));
// CHECK-NEXT:    CHECK(0);
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK((elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count(), 0));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    return 0;
}

__global__ void kernel(float *g_data, float value){}

void foo_test_3()
{

    int num = 128;
    int nbytes = num * sizeof(int);
    float value = 10.0f;

    float *h_a = 0;
// CHECK:    /*
// CHECK-NEXT:    DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK((h_a = (float *)sycl::malloc_host(nbytes, dpct::get_default_queue()), 0));
    CHECK(cudaMallocHost((void **)&h_a, nbytes));

    float *d_a = 0;
// CHECK:    /*
// CHECK-NEXT:    DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK((d_a = (float *)sycl::malloc_device(nbytes, dpct::get_default_queue()), 0));
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK((dpct::get_default_queue().memset(d_a, 255, nbytes).wait(), 0));
    CHECK(cudaMalloc((void **)&d_a, nbytes));
    CHECK(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    dim3 block;
    dim3 grid;

    // create cuda event handles
// CHECK:    sycl::event stop;
// CHECK-NEXT:    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0 because this call is redundant in DPC++.
// CHECK-NEXT:    */
// CHECK-NEXT:    sycl::event stop_q_ct1_1;
// CHECK-NEXT:    sycl::event stop_q_ct1_2;
// CHECK-NEXT:    CHECK(0);
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&stop));

    // asynchronously issue work to the GPU (all to stream 0)
// CHECK:    /*
// CHECK-NEXT:    DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK((stop_q_ct1_1 = dpct::get_default_queue().memcpy(d_a, h_a, nbytes), 0));
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:    */
// CHECK-NEXT:    stop = dpct::get_default_queue().parallel_for<dpct_kernel_name<class kernel_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                sycl::nd_range<3>(grid * block, block),
// CHECK-NEXT:                [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                    kernel(d_a, value);
// CHECK-NEXT:                });
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK((stop_q_ct1_2 = dpct::get_default_queue().memcpy(h_a, d_a, nbytes), 0));
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:    */
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1024:{{[0-9]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT:    */
// CHECK-NEXT:    stop_q_ct1_2.wait();
// CHECK-NEXT:    stop_q_ct1_1.wait();
// CHECK-NEXT:    stop.wait();
// CHECK-NEXT:    stop_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:    CHECK(0);
    CHECK(cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    kernel<<<grid, block>>>(d_a, value);
    CHECK(cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop));

    unsigned long int counter = 0;
// CHECK:    while ((int)stop.get_info<sycl::info::event::command_execution_status>() != 0) {
// CHECK-NEXT:        counter++;
// CHECK-NEXT:    }
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    // release resources
// CHECK:    /*
// CHECK-NEXT:    DPCT1027:{{[0-9]+}}: The call to cudaEventDestroy was replaced with 0 because this call is redundant in DPC++.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK(0);
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK((sycl::free(h_a, dpct::get_default_queue()), 0));
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK((sycl::free(d_a, dpct::get_default_queue()), 0));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFree(d_a));
}

#define N 64000000
#define NTIMES 5000
const double dbl_eps = 2.2204460492503131e-16;

__global__ void set_array(double *a, double value, size_t len) {}
__global__ void STREAM_Copy(double *a, double *b, size_t len) {}
__global__ void STREAM_Copy_Optimized(double *a, double *b, size_t len) {}

void foo_test_4() {
  double *d_a, *d_b, *d_c;

  int k;
  float times[8][NTIMES];
  double scalar;

  dim3 dimBlock;
  dim3 dimGrid;

// CHECK:  /*
// CHECK-NEXT:  DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:  */
// CHECK-NEXT:    dpct::get_default_queue().parallel_for<dpct_kernel_name<class set_array_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
// CHECK-NEXT:                [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                    set_array(d_a, 2., N);
// CHECK-NEXT:                });
  set_array<<<dimGrid, dimBlock>>>(d_a, 2., N);

// CHECK:  sycl::event start, stop;
// CHECK-NEXT:  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
// CHECK-NEXT:  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1026:{{[0-9]+}}: The call to cudaEventCreate was removed because this call is redundant in DPC++.
// CHECK-NEXT:  */
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1026:{{[0-9]+}}: The call to cudaEventCreate was removed because this call is redundant in DPC++.
// CHECK-NEXT:  */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (k = 0; k < NTIMES; k++) {
// CHECK:    /*
// CHECK-NEXT:    DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:    */
// CHECK-NEXT:    start_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:    start = dpct::get_default_queue().ext_oneapi_submit_barrier();
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:    */
// CHECK-NEXT:        dpct::get_default_queue().parallel_for<dpct_kernel_name<class STREAM_Copy_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
// CHECK-NEXT:                    [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                        STREAM_Copy(d_a, d_c, N);
// CHECK-NEXT:                    });
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:    */
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    stop_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:    stop = dpct::get_default_queue().ext_oneapi_submit_barrier();
// CHECK-NEXT:    times[0][k] = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    cudaEventRecord(start, 0);
    STREAM_Copy<<<dimGrid, dimBlock>>>(d_a, d_c, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[0][k], start, stop);

// CHECK:    /*
// CHECK-NEXT:    DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:    */
// CHECK-NEXT:    start_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:    start = dpct::get_default_queue().ext_oneapi_submit_barrier();
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
// CHECK-NEXT:    */
// CHECK-NEXT:        dpct::get_default_queue().parallel_for<dpct_kernel_name<class STREAM_Copy_Optimized_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
// CHECK-NEXT:                    [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                        STREAM_Copy_Optimized(d_a, d_c, N);
// CHECK-NEXT:                    });
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
// CHECK-NEXT:    */
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:    stop_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:    stop = dpct::get_default_queue().ext_oneapi_submit_barrier();
// CHECK-NEXT:    times[1][k] = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    cudaEventRecord(start, 0);
    STREAM_Copy_Optimized<<<dimGrid, dimBlock>>>(d_a, d_c, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[1][k], start, stop);
  }
}

__global__ void kernel_ctst2184() {}

void foo_ctst2184() {
  int nbytes;
  float value = 10.0f;
  float gpu_time = 0.0f;

  float *h_a = 0;
  float *d_a = 0;

  // CHECK: sycl::event stop, start;
  // CHECK-NEXT:  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  // CHECK-NEXT:  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  // CHECK-NEXT:  /*
  // CHECK-NEXT:    DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0 because this call is redundant in DPC++.
  // CHECK-NEXT:    */
  // CHECK-NEXT:  CHECK(0);
  // CHECK-NEXT:  /*
  // CHECK-NEXT:    DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0 because this call is redundant in DPC++.
  // CHECK-NEXT:    */
  // CHECK-NEXT:  CHECK(0);
  cudaEvent_t stop, start;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  // CHECK:  sycl::event stop_q_ct1_1;
  // CHECK:  sycl::event stop_q_ct1_2;
  // CHECK:  start_ct1 = std::chrono::steady_clock::now();
  // CHECK:  CHECK(0);
  CHECK(cudaEventRecord(start));
  CHECK(cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
  kernel_ctst2184<<<1, 1>>>();
  CHECK(cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

  // CHECK: stop_q_ct1_2.wait();
  // CHECK: stop_q_ct1_1.wait();
  // CHECK: stop.wait();
  // CHECK: stop_ct1 = std::chrono::steady_clock::now();
  // CHECK: CHECK(0);
  CHECK(cudaEventRecord(stop));

  unsigned long int counter = 0;
  while (cudaEventQuery(stop) == cudaErrorNotReady) {
    counter++;
  }
  CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
}
