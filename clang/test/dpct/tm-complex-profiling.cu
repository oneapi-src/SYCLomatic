// RUN: dpct --enable-profiling  -out-root %T/tm-complex-profiling %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/tm-complex-profiling/tm-complex-profiling.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/tm-complex-profiling/tm-complex-profiling.dp.cpp -o %T/tm-complex-profiling/tm-complex-profiling.dp.o %}

// CHECK: #define DPCT_PROFILING_ENABLED
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <stdio.h>
// CHECK-NEXT: #include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 300
#define NSTREAM 4

__global__ void kernel_1(){}
__global__ void kernel_2(){}
__global__ void kernel_3(){}
__global__ void kernel_4(){}

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

    // set up max connection
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
    // CHECK:     dpct::sync_barrier(start, &dpct::get_in_order_queue());
    cudaEventRecord(start, 0);

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();

        // CHECK:        dpct::sync_barrier(kernelEvent[i], streams[i]);
        cudaEventRecord(kernelEvent[i], streams[i]);
        // CHECK:         streams[n_streams - 1]->ext_oneapi_submit_barrier({*kernelEvent[i]});
        cudaStreamWaitEvent(streams[n_streams - 1], kernelEvent[i], 0);
    }

    // record stop event
    // CHECK:    dpct::sync_barrier(stop, &dpct::get_in_order_queue());
    // CHECK-NEXT:    stop->wait_and_throw();
    // CHECK-NEXT:    elapsed_time = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
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

// CHECK:    dpct::get_in_order_queue().parallel_for<dpct_kernel_name<class kernel_1_{{[a-z0-9]+}}>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:            kernel_1();
// CHECK-NEXT:        });
// CHECK-NEXT:  /*
// CHECK-NEXT:    DPCT1024:{{[0-9]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT:    */
// CHECK-NEXT:  CHECK_FOO(DPCT_CHECK_ERROR(dpct::sync_barrier(stop)));
// CHECK-NEXT:  stop->wait_and_throw();
// CHECK-NEXT:  MY_ERROR_CHECKER(DPCT_CHECK_ERROR(stop->wait_and_throw()));
// CHECK-NEXT:  MY_CHECKER(DPCT_CHECK_ERROR(stop->wait_and_throw()));
// CHECK-NEXT:  (DPCT_CHECK_ERROR(stop->wait_and_throw()));
// CHECK-NEXT:  int ret;
// CHECK-NEXT:  ret = (DPCT_CHECK_ERROR(stop->wait_and_throw()));
// CHECK-NEXT:  int a = (DPCT_CHECK_ERROR(stop->wait_and_throw()));
// CHECK-NEXT:  et = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
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

    cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(
                                cudaStream_t));

    dim3 block (iblock);
    dim3 grid  (isize / iblock);

    // creat events
// CHECK:    dpct::event_ptr start, stop;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

// CHECK:    dpct::event_ptr *kernelEvent;
    cudaEvent_t *kernelEvent;
    kernelEvent = (cudaEvent_t *) malloc(n_streams * sizeof(cudaEvent_t));

    // record start event
// CHECK:    /*
// CHECK-NEXT:    DPCT1024:{{[0-9]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK(DPCT_CHECK_ERROR(dpct::sync_barrier(start, &dpct::get_in_order_queue())));
    CHECK(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {

        foo_kernel_1<<<grid, block, 0, streams[i]>>>();
        foo_kernel_2<<<grid, block, 0, streams[i]>>>();
        foo_kernel_3<<<grid, block, 0, streams[i]>>>();
        foo_kernel_4<<<grid, block, 0, streams[i]>>>();

// CHECK:        CHECK(DPCT_CHECK_ERROR(*kernelEvent[i] = streams[i]->ext_oneapi_submit_barrier()));
// CHECK-NEXT:        streams[n_streams - 1]->ext_oneapi_submit_barrier({*kernelEvent[i]});
        CHECK(cudaEventRecord(kernelEvent[i], streams[i]));
        cudaStreamWaitEvent(streams[n_streams - 1], kernelEvent[i], 0);
    }

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
    CHECK(cudaMallocHost((void **)&h_a, nbytes));

    float *d_a = 0;
    CHECK(cudaMalloc((void **)&d_a, nbytes));
    CHECK(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    dim3 block;
    dim3 grid;

    // create cuda event handles
// CHECK:    dpct::event_ptr stop;
// CHECK:    CHECK(DPCT_CHECK_ERROR(stop = new sycl::event()));
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&stop));

    // asynchronously issue work to the GPU (all to stream 0)
// CHECK:    CHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(d_a, h_a, nbytes)));
// CHECK-NEXT:    dpct::get_in_order_queue().parallel_for<dpct_kernel_name<class kernel_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                sycl::nd_range<3>(grid * block, block),
// CHECK-NEXT:                [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                    kernel(d_a, value);
// CHECK-NEXT:                });
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1124:{{[0-9]+}}: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the origin API might be synchronous, it depends on the type of operand memory, so you may need to call wait() on event return by memcpy API to ensure synchronization behavior.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(h_a, d_a, nbytes)));
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1024:{{[0-9]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT:    */
// CHECK-NEXT:    CHECK(DPCT_CHECK_ERROR(dpct::sync_barrier(stop)));
    CHECK(cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    kernel<<<grid, block>>>(d_a, value);
    CHECK(cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop));

    unsigned long int counter = 0;
// CHECK:    while (dpct::sycl_event_query(stop) == 1) {
// CHECK-NEXT:        counter++;
// CHECK-NEXT:    }
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    // release resources

// CHECK:    CHECK(DPCT_CHECK_ERROR(dpct::destroy_event(stop)));
// CHECK-NEXT:    CHECK(DPCT_CHECK_ERROR(sycl::free(h_a, dpct::get_in_order_queue())));
// CHECK-NEXT:    CHECK(DPCT_CHECK_ERROR(dpct::dpct_free(d_a, dpct::get_in_order_queue())));
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
  // CHECK-NEXT:  DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  {
  // CHECK-NEXT:    dpct::get_device(dpct::get_device_id(dpct::get_in_order_queue().get_device())).has_capability_or_fail({sycl::aspect::fp64});
  // CHECK-EMPTY:
  // CHECK-NEXT:    dpct::get_in_order_queue().parallel_for<dpct_kernel_name<class set_array_{{[a-z0-9]+}}>>(
  // CHECK-NEXT:                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
  // CHECK-NEXT:                [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:                    set_array(d_a, 2., N);
  // CHECK-NEXT:                });
  // CHECK-NEXT:  }
  set_array<<<dimGrid, dimBlock>>>(d_a, 2., N);

  // CHECK:  dpct::event_ptr start, stop;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (k = 0; k < NTIMES; k++) {
    // CHECK:    dpct::sync_barrier(start, &dpct::get_in_order_queue());
    // CHECK-NEXT:    /*
    // CHECK-NEXT:    DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    // CHECK-NEXT:    */
    // CHECK-NEXT:    {
    // CHECK-NEXT:        dpct::get_device(dpct::get_device_id(dpct::get_in_order_queue().get_device())).has_capability_or_fail({sycl::aspect::fp64});
    // CHECK-EMPTY:
    // CHECK-NEXT:        dpct::get_in_order_queue().parallel_for<dpct_kernel_name<class STREAM_Copy_{{[a-z0-9]+}}>>(
    // CHECK-NEXT:                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
    // CHECK-NEXT:                    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:                        STREAM_Copy(d_a, d_c, N);
    // CHECK-NEXT:                    });
    // CHECK-NEXT:    }
    // CHECK-NEXT:    dpct::sync_barrier(stop, &dpct::get_in_order_queue());
    // CHECK-NEXT:    stop->wait_and_throw();
    // CHECK-NEXT:    times[0][k] = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
    cudaEventRecord(start, 0);
    STREAM_Copy<<<dimGrid, dimBlock>>>(d_a, d_c, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[0][k], start, stop);

    // CHECK:    dpct::sync_barrier(start, &dpct::get_in_order_queue());
    // CHECK-NEXT:    /*
    // CHECK-NEXT:    DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    // CHECK-NEXT:    */
    // CHECK-NEXT:    {
    // CHECK-NEXT:        dpct::get_device(dpct::get_device_id(dpct::get_in_order_queue().get_device())).has_capability_or_fail({sycl::aspect::fp64});
    // CHECK-EMPTY:
    // CHECK-NEXT:        dpct::get_in_order_queue().parallel_for<dpct_kernel_name<class STREAM_Copy_Optimized_{{[a-z0-9]+}}>>(
    // CHECK-NEXT:                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
    // CHECK-NEXT:                    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:                        STREAM_Copy_Optimized(d_a, d_c, N);
    // CHECK-NEXT:                    });
    // CHECK-NEXT:    }
    // CHECK-NEXT:    dpct::sync_barrier(stop, &dpct::get_in_order_queue());
    // CHECK-NEXT:    stop->wait_and_throw();
    // CHECK-NEXT:    times[1][k] = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
    cudaEventRecord(start, 0);
    STREAM_Copy_Optimized<<<dimGrid, dimBlock>>>(d_a, d_c, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[1][k], start, stop);
  }
}

__global__ void kernel_test_2184() {}

void foo_test_2184() {
  int nbytes;
  float value = 10.0f;
  float gpu_time = 0.0f;

  float *h_a = 0;
  float *d_a = 0;

  // CHECK: dpct::event_ptr stop, start;
  // CHECK:  CHECK(DPCT_CHECK_ERROR(start = new sycl::event()));
  // CHECK:  CHECK(DPCT_CHECK_ERROR(stop = new sycl::event()));
  cudaEvent_t stop, start;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  // CHECK:  CHECK(DPCT_CHECK_ERROR(dpct::sync_barrier(start)));
  CHECK(cudaEventRecord(start));
  CHECK(cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
  kernel_test_2184<<<1, 1>>>();
  CHECK(cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

  // CHECK: CHECK(DPCT_CHECK_ERROR(dpct::sync_barrier(stop)));
  CHECK(cudaEventRecord(stop));

  unsigned long int counter = 0;
  while (cudaEventQuery(stop) == cudaErrorNotReady) {
    counter++;
  }
  CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
}

struct foo_node {
  cudaEvent_t readyEvent;
  struct foo_node *next;
};

void cudaEventQuery_foo_test() {
  struct foo_node *next;
  struct foo_node *curr;

  while (next != NULL) {
    curr = next;
    next = curr->next;

    // CHECK:    if (!curr->readyEvent != 0) {
    // CHECK-NEXT:      dpct::err0 e = dpct::sycl_event_query(curr->readyEvent);
    // CHECK-NEXT:      if (e == 0) {
    // CHECK-NEXT:        // to do some thing.
    // CHECK-NEXT:      } else if (e != 1) {
    // CHECK-NEXT:        // to do error handling.
    // CHECK-NEXT:      }
    // CHECK-NEXT:    }
    if (!curr->readyEvent != 0) {
      cudaError_t e = cudaEventQuery(curr->readyEvent);
      if (e == cudaSuccess) {
        // to do some thing.
      } else if (e != cudaErrorNotReady) {
        // to do error handling.
      }
    }
  }
}
