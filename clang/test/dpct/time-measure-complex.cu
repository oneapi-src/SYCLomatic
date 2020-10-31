// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/time-measure-complex.dp.cpp --match-full-lines %s
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
    cudaEventRecord(start, 0);

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
        // CHECK: kernelEvent[i] = streams[i]->submit(
        // CHECK-NEXT:     [&](sycl::handler &cgh) {
        // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class kernel_1_{{[a-z0-9]+}}>>(
        // CHECK-NEXT:             sycl::nd_range<3>(grid * block, block),
        // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
        // CHECK-NEXT:                 kernel_1();
        // CHECK-NEXT:             });
        // CHECK-NEXT:     });
        // CHECK-NEXT: kernelEvent[i].wait();
        kernel_1<<<grid, block, 0, streams[i]>>>();
        // CHECK: kernelEvent[i] = streams[i]->submit(
        // CHECK-NEXT:     [&](sycl::handler &cgh) {
        // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class kernel_2_{{[a-z0-9]+}}>>(
        // CHECK-NEXT:             sycl::nd_range<3>(grid * block, block),
        // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
        // CHECK-NEXT:                 kernel_2();
        // CHECK-NEXT:             });
        // CHECK-NEXT:     });
        // CHECK-NEXT: kernelEvent[i].wait();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        // CHECK: kernelEvent[i] = streams[i]->submit(
        // CHECK-NEXT:     [&](sycl::handler &cgh) {
        // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class kernel_3_{{[a-z0-9]+}}>>(
        // CHECK-NEXT:             sycl::nd_range<3>(grid * block, block),
        // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
        // CHECK-NEXT:                 kernel_3();
        // CHECK-NEXT:             });
        // CHECK-NEXT:     });
        // CHECK-NEXT: kernelEvent[i].wait();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        // CHECK: kernelEvent[i] = streams[i]->submit(
        // CHECK-NEXT:     [&](sycl::handler &cgh) {
        // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class kernel_4_{{[a-z0-9]+}}>>(
        // CHECK-NEXT:             sycl::nd_range<3>(grid * block, block),
        // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
        // CHECK-NEXT:                 kernel_4();
        // CHECK-NEXT:             });
        // CHECK-NEXT:     });
        // CHECK-NEXT: kernelEvent[i].wait();
        kernel_4<<<grid, block, 0, streams[i]>>>();

        // CHECK: kernelEvent_ct1_i = std::chrono::steady_clock::now();
        cudaEventRecord(kernelEvent[i], streams[i]);
        // CHECK: kernelEvent[i].wait();
        cudaStreamWaitEvent(streams[n_streams - 1], kernelEvent[i], 0);
    }

    // record stop event
    // CHECK: stop_ct1 = std::chrono::steady_clock::now();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Measured time for parallel execution = %.3fs\n",
           elapsed_time / 1000.0f);

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

#define CHECK(ARG) ARG

void foo() {
  float et;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  CHECK(cudaEventRecord(start));
  // 1CHECK: stop = dpct::get_default_queue().submit(
  // 1CHECK-NEXT:     [&](sycl::handler &cgh) {
  // 1CHECK-NEXT:         cgh.parallel_for(
  // 1CHECK-NEXT:             sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // 1CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // 1CHECK-NEXT:                 kernel_1();
  // 1CHECK-NEXT:             });
  // 1CHECK-NEXT:     });
  // 1CHECK-NEXT: stop.wait();
  kernel_1<<<1, 1>>>();
  CHECK(cudaEventRecord(stop));
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&et, start, stop);
}
