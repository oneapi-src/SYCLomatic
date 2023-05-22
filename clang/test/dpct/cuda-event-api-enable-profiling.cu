// RUN: dpct --enable-profiling --format-range=none --usm-level=none -out-root %T/cuda-event-api-enable-profiling %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda-event-api-enable-profiling/cuda-event-api-enable-profiling.dp.cpp --match-full-lines %s

// CHECK:#define DPCT_PROFILING_ENABLED
// CHECK-NEXT: #define DPCT_USM_LEVEL_NONE
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <stdio.h>
#include <stdio.h>

template <typename T>
void my_error_checker(T ReturnValue, char const *const FuncName) {
}

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)

//CHECK: #define CudaEvent(X) X = new sycl::event()
#define CudaEvent(X) cudaEventCreate(&X)

#define MY_CHECKER(CALL) do {                           \
  cudaError_t Error = CALL;                             \
  if (Error != cudaSuccess) {                           \
  }                                                     \
} while(0)

__global__ void kernelFunc()
{
}

int main(int argc, char* argv[]) {
// CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK: dpct::event_ptr start, stop;
// CHECK-EMPTY:
// CHECK: start = new sycl::event();
// CHECK: stop = new sycl::event();
// CHECK-EMPTY:
// CHECK-NEXT: float elapsed_time;
// CHECK-EMPTY:
// CHECK-NEXT: dev_ct1.queues_wait_and_throw();
// CHECK-EMPTY:
// CHECK-NEXT: int blocks = 32, threads = 32;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float elapsed_time;

  cudaDeviceSynchronize();

  int blocks = 32, threads = 32;

// CHECK: printf("<<<\n");
// CHECK-NEXT: CudaEvent(start);
// CHECK-NEXT: printf(">>>\n");
  printf("<<<\n");
  CudaEvent(start);
  printf(">>>\n");


// CHECK: MY_ERROR_CHECKER(CHECK_SYCL_ERROR(start = new sycl::event()));
// CHECK: dpct::err0 et = CHECK_SYCL_ERROR(stop = new sycl::event());
  MY_ERROR_CHECKER(cudaEventCreate(&start));
  cudaError_t et = cudaEventCreate(&stop);


  // kernel call without sync
// CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
// CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:           kernelFunc();
// CHECK-NEXT:         });
  kernelFunc<<<blocks,threads>>>();

// CHECK:   *start = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(start, 0);

  // kernel call without sync
// CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
// CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:           kernelFunc();
// CHECK-NEXT:         });
  kernelFunc<<<blocks,threads>>>();


// CHECK: *start = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(start, 0);

// CHECK: MY_ERROR_CHECKER(CHECK_SYCL_ERROR(*start = q_ct1.ext_oneapi_submit_barrier()));
  MY_ERROR_CHECKER(cudaEventRecord(start, 0));

// CHECK: if (0)
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT:   */
// CHECK-NEXT:   MY_ERROR_CHECKER(CHECK_SYCL_ERROR(*start = q_ct1.ext_oneapi_submit_barrier()));
  if (0)
    MY_ERROR_CHECKER(cudaEventRecord(start, 0));

  // kernel call with sync
// CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
// CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:           kernelFunc();
// CHECK-NEXT:         });
  kernelFunc<<<blocks,threads>>>();
// CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
// CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:           kernelFunc();
// CHECK-NEXT:         });
  kernelFunc<<<blocks,threads>>>();

// CHECK:   *stop = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(stop, 0);

// CHECK: MY_ERROR_CHECKER(CHECK_SYCL_ERROR(*stop = q_ct1.ext_oneapi_submit_barrier()));
  MY_ERROR_CHECKER(cudaEventRecord(stop, 0));


// CHECK:   MY_ERROR_CHECKER(CHECK_SYCL_ERROR(*stop = q_ct1.ext_oneapi_submit_barrier()));
  if (1)
    MY_ERROR_CHECKER(cudaEventRecord(stop, 0));

  // kernel call without sync
// CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
// CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:           kernelFunc();
// CHECK-NEXT:         });
  kernelFunc<<<blocks,threads>>>();

// CHECK: *stop = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(stop, 0);

// CHECK: /*
// CHECK-NEXT: DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT: */
// CHECK-NEXT: MY_ERROR_CHECKER(CHECK_SYCL_ERROR(*stop = q_ct1.ext_oneapi_submit_barrier()));
  MY_ERROR_CHECKER(cudaEventRecord(stop, 0));

// CHECK: if (0)
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
// CHECK-NEXT:   */
// CHECK-NEXT:   MY_ERROR_CHECKER(CHECK_SYCL_ERROR(*start = q_ct1.ext_oneapi_submit_barrier()));
  if (0)
    MY_ERROR_CHECKER(cudaEventRecord(start, 0));

// CHECK:  MY_ERROR_CHECKER(CHECK_SYCL_ERROR(*start = q_ct1.ext_oneapi_submit_barrier()));
  MY_ERROR_CHECKER(cudaEventRecord(start));

  // kernel call without sync
// CHECK:  DPCT1049:{{[0-9a-f]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
// CHECK-NEXT:  */
// CHECK-NEXT:  q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          kernelFunc();
// CHECK-NEXT:        });
  kernelFunc<<<blocks,threads>>>();

// CHECK:  *stop = q_ct1.ext_oneapi_submit_barrier();
// CHECK-NEXT:  stop->wait_and_throw();
// CHECK-NEXT:  elapsed_time = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);

// CHECK: MY_ERROR_CHECKER(CHECK_SYCL_ERROR(elapsed_time = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f));
  MY_ERROR_CHECKER(cudaEventElapsedTime(&elapsed_time, start, stop));

}

void foo() {
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float elapsed_time;

  cudaDeviceSynchronize();

  int blocks = 32, threads = 32;

// CHECK:  MY_CHECKER(CHECK_SYCL_ERROR(*start = q_ct1.ext_oneapi_submit_barrier()));
  MY_CHECKER(cudaEventRecord(start, 0));
  kernelFunc<<<blocks,threads>>>();
// CHECK: MY_CHECKER(CHECK_SYCL_ERROR(*stop = q_ct1.ext_oneapi_submit_barrier()));
  MY_CHECKER(cudaEventRecord(stop, 0));

  cudaEventSynchronize(stop);

// CHECK: MY_CHECKER(CHECK_SYCL_ERROR(elapsed_time = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f));
  MY_CHECKER(cudaEventElapsedTime(&elapsed_time, start, stop));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  {
  // CHECK: dpct::err0 err = CHECK_SYCL_ERROR(*start = q_ct1.ext_oneapi_submit_barrier());
    cudaError_t err = cudaEventRecord(start, 0);
  // CHECK: err = CHECK_SYCL_ERROR(*stop = q_ct1.ext_oneapi_submit_barrier());
    err = cudaEventRecord(stop, 0);
    if (cudaSuccess != err) {
      printf("%s\n", cudaGetErrorString( err));
    }
  }
}

void fun(int) {}

void bar() {
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float elapsed_time;

  cudaDeviceSynchronize();

  int blocks = 32, threads = 32;

// CHECK: fun(CHECK_SYCL_ERROR(*start = q_ct1.ext_oneapi_submit_barrier()));
  fun(cudaEventRecord(start, 0));
  kernelFunc<<<blocks,threads>>>();
// CHECK: fun(CHECK_SYCL_ERROR(*stop = q_ct1.ext_oneapi_submit_barrier()));
  fun(cudaEventRecord(stop, 0));

  cudaEventSynchronize(stop);
// CHECK: fun(CHECK_SYCL_ERROR(elapsed_time = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f));
  fun(cudaEventElapsedTime(&elapsed_time, start, stop));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

struct Node {
 // CHECK: dpct::event_ptr start;
 cudaEvent_t start;
 // CHECK: dpct::event_ptr end;
 cudaEvent_t end;
 // CHECK: dpct::event_ptr *ev[100];
 cudaEvent_t *ev[100];
 // CHECK: dpct::event_ptr events[100];
 cudaEvent_t events[100];
 // CHECK: dpct::event_ptr *p_events;
 cudaEvent_t *p_events;
};

void foo2(Node *n) {
  float elapsed_time;

// CHECK: *n->start = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(n->start, 0);
// CHECK: *n->start = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(n->start, 0);
  // do something
// CHECK: *n->end = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(n->end, 0);
// CHECK: *n->end = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(n->end, 0);
// CHECK: elapsed_time = (n->end->get_profiling_info<sycl::info::event_profiling::command_end>() - n->start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
  cudaEventElapsedTime(&elapsed_time, n->start, n->end);
  {
    int errorCode;
  // CHECK: MY_CHECKER(CHECK_SYCL_ERROR(*n->start = q_ct1.ext_oneapi_submit_barrier()));
    MY_CHECKER(cudaEventRecord(n->start, 0));
  // CHECK: errorCode = CHECK_SYCL_ERROR(*n->start = q_ct1.ext_oneapi_submit_barrier());
    errorCode = cudaEventRecord(n->start, 0);
  }

  Node node;
// CHECK: *node.start = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(node.start, 0);
// CHECK: *node.start = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(node.start, 0);
  // do something
// CHECK: *node.end = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(node.end, 0);
// CHECK: *node.end = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(node.end, 0);
// CHECK: elapsed_time = (node.end->get_profiling_info<sycl::info::event_profiling::command_end>() - node.start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
  cudaEventElapsedTime(&elapsed_time, node.start, node.end);
  {
    int errorCode;
  // CHECK: MY_CHECKER(CHECK_SYCL_ERROR(*node.start = q_ct1.ext_oneapi_submit_barrier()));
    MY_CHECKER(cudaEventRecord(node.start, 0));
  // CHECK: errorCode = CHECK_SYCL_ERROR(*node.start = q_ct1.ext_oneapi_submit_barrier());
    errorCode = cudaEventRecord(node.start, 0);
  }

  {
  // CHECK: *node.events[0] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(node.events[0]);
  // CHECK: *node.events[0] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(node.events[0]);
  // CHECK: *node.events[23] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(node.events[23]);
  // CHECK: *node.events[23] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(node.events[23]);
  // CHECK: elapsed_time = (node.events[23]->get_profiling_info<sycl::info::event_profiling::command_end>() - node.events[0]->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
    cudaEventElapsedTime(&elapsed_time, node.events[0], node.events[23]);
  }

  {
  // CHECK: **node.ev[0] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(*node.ev[0]);
  // CHECK: **node.ev[0] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(*node.ev[0]);
  // CHECK: **node.ev[23] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(*node.ev[23]);
  // CHECK: **node.ev[23] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(*node.ev[23]);
  // CHECK: elapsed_time = (*node.ev[23]->get_profiling_info<sycl::info::event_profiling::command_end>() - *node.ev[0]->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
    cudaEventElapsedTime(&elapsed_time, *node.ev[0], *node.ev[23]);
  }

  {
  // CHECK: **(&node)->ev[0] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(*(&node)->ev[0]);
  // CHECK: **(&node)->ev[0] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(*(&node)->ev[0]);
  // CHECK: **(&node)->ev[23] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(*(&node)->ev[23]);
  // CHECK: **(&node)->ev[23] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(*(&node)->ev[23]);
  // CHECK:  elapsed_time = (*(&node)->ev[23]->get_profiling_info<sycl::info::event_profiling::command_end>() - *(&node)->ev[0]->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
    cudaEventElapsedTime(&elapsed_time, *(&node)->ev[0], *(&node)->ev[23]);
  }

  {
  // CHECK: *n->p_events[0] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(n->p_events[0]);
  // CHECK: *n->p_events[1] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(n->p_events[1]);
  // CHECK:  *n->p_events[2] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(n->p_events[2]);
  // CHECK:  *n->p_events[3] = q_ct1.ext_oneapi_submit_barrier();
    cudaEventRecord(n->p_events[3]);
  }
}
