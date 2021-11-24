// RUN: dpct --format-range=none --usm-level=none -out-root %T/cuda-event-api %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda-event-api/cuda-event-api.dp.cpp --match-full-lines %s

#include <stdio.h>

template <typename T>
// CHECK: void my_error_checker(T ReturnValue, char const *const FuncName) {
void my_error_checker(T ReturnValue, char const *const FuncName) {
}

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)

//CHECK: /*
//CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaEventCreate was removed because this call is redundant in DPC++.
//CHECK-NEXT: */
//CHECK-NEXT: #define CudaEvent(X)
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
  // CHECK: sycl::event start, stop;
  // CHECK: std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  // CHECK: std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  // CHECK-EMPTY:
  // CHECK: /*
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaEventCreate was removed because this call is redundant in DPC++.
  // CHECK: */
  // CHECK: /*
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaEventCreate was removed because this call is redundant in DPC++.
  // CHECK: */
  // CHECK-EMPTY:
  // CHECK-NEXT: float elapsed_time;
  // CHECK-EMPTY:
  // CHECK-NEXT: dev_ct1.queues_wait_and_throw();
  // CHECK-EMPTY:
  // CHECK-NEXT: int blocks = 32, threads = 32;
  cudaEvent_t start, stop;

  cudaEventCreate(&start)  
    ;   
  cudaEventCreate(&stop)  ;   

  float elapsed_time;

  cudaDeviceSynchronize();

  int blocks = 32, threads = 32;

  // CHECK: printf("<<<\n");
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1061:{{[0-9]+}}: Call to CudaEvent macro was removed because it only contains code, which is unnecessary in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: printf(">>>\n");
  printf("<<<\n");
  CudaEvent(start);
  printf(">>>\n");

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0 because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER(0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0 because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: int et = 0;
  MY_ERROR_CHECKER(cudaEventCreate(&start));
  cudaError_t et = cudaEventCreate(&stop);


  // kernel call without sync
  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: start_ct1 = std::chrono::steady_clock::now();
  cudaEventRecord(start, 0);

  // kernel call without sync
  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: start_ct1 = std::chrono::steady_clock::now();
  // CHECK-NEXT: start = q_ct1.ext_oneapi_submit_barrier();
  cudaEventRecord(start, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[0-9a-f]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT: */
  // CHECK-NEXT: start_ct1 = std::chrono::steady_clock::now();
  // CHECK-NEXT: MY_ERROR_CHECKER((start = q_ct1.ext_oneapi_submit_barrier(), 0));
  MY_ERROR_CHECKER(cudaEventRecord(start, 0));

  // CHECK: if (0)
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   start_ct1 = std::chrono::steady_clock::now();
  // CHECK-NEXT:   MY_ERROR_CHECKER((start = q_ct1.ext_oneapi_submit_barrier(), 0));
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

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: stop_ct1 = std::chrono::steady_clock::now();
  cudaEventRecord(stop, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT: */
  // CHECK-NEXT: stop_ct1 = std::chrono::steady_clock::now();
  // CHECK-NEXT: MY_ERROR_CHECKER((stop = q_ct1.ext_oneapi_submit_barrier(), 0));
  MY_ERROR_CHECKER(cudaEventRecord(stop, 0));

  // CHECK: if (1)
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   stop_ct1 = std::chrono::steady_clock::now();
  // CHECK-NEXT:   MY_ERROR_CHECKER((stop = q_ct1.ext_oneapi_submit_barrier(), 0));
  if (1)
    MY_ERROR_CHECKER(cudaEventRecord(stop, 0));

  // kernel call without sync
  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: stop_ct1 = std::chrono::steady_clock::now();
  cudaEventRecord(stop, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT: */
  // CHECK-NEXT: stop_ct1 = std::chrono::steady_clock::now();
  // CHECK-NEXT: MY_ERROR_CHECKER((stop = q_ct1.ext_oneapi_submit_barrier(), 0));
  MY_ERROR_CHECKER(cudaEventRecord(stop, 0));

  // CHECK: if (0)
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   start_ct1 = std::chrono::steady_clock::now();
  // CHECK-NEXT:   MY_ERROR_CHECKER((start = q_ct1.ext_oneapi_submit_barrier(), 0));
  if (0)
    MY_ERROR_CHECKER(cudaEventRecord(start, 0));

  // CHECK:  MY_ERROR_CHECKER(0);
  MY_ERROR_CHECKER(cudaEventRecord(start));

  // kernel call without sync
  // CHECK:  DPCT1049:{{[0-9a-f]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), 
  // CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:          kernelFunc();
  // CHECK-NEXT:        });
  kernelFunc<<<blocks,threads>>>();

  // CHECK:  dpct::get_current_device().queues_wait_and_throw();
  // CHECK-NEXT:  stop_ct1 = std::chrono::steady_clock::now();
  // CHECK-NEXT:  elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9a-z]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER((elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count(), 0));
  MY_ERROR_CHECKER(cudaEventElapsedTime(&elapsed_time, start, stop));

  // kernel call without sync
  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  kernelFunc<<<blocks,threads>>>();

  // CHECK:/*
  // CHECK-NEXT:  DPCT1026:{{[0-9a-z]+}}: The call to cudaEventCreate was removed because this call is redundant in DPC++.
  // CHECK-NEXT:  */
  cudaEventCreate(&stop);

  // CHECK: int e = (int)stop.get_info<sycl::info::event::command_execution_status>();
  // CHECK-NEXT: MY_ERROR_CHECKER(e);
  // CHECK-NEXT: MY_ERROR_CHECKER((int)stop.get_info<sycl::info::event::command_execution_status>());
  // CHECK-NEXT: if (0 == (int)stop.get_info<sycl::info::event::command_execution_status>()){}
  // CHECK-NEXT: while((int)stop.get_info<sycl::info::event::command_execution_status>() != 0){}
  // CHECK-NEXT: for(;0 != (int)stop.get_info<sycl::info::event::command_execution_status>();){}
  // CHECK-NEXT: do{}while((int)stop.get_info<sycl::info::event::command_execution_status>() == 0);
  // CHECK-NEXT: {
  // CHECK-NEXT:   int *a;
  // CHECK-NEXT:   sycl::info::event_command_status e1;
  // CHECK-NEXT:   e1 = stop.get_info<sycl::info::event::command_execution_status>();
  // CHECK-NEXT:   if (sycl::info::event_command_status::complete != stop.get_info<sycl::info::event::command_execution_status>()) {}
  // CHECK-NEXT:   if (e1 == sycl::info::event_command_status::complete){}
  // CHECK-NEXT:   while(e1 != sycl::info::event_command_status::complete) {
  // CHECK-NEXT:     e1 = stop.get_info<sycl::info::event::command_execution_status>();
  // CHECK-NEXT:   }
  // CHECK-NEXT:   for(;e1 != sycl::info::event_command_status::complete;){
  // CHECK-NEXT:     e1 = stop.get_info<sycl::info::event::command_execution_status>();
  // CHECK-NEXT:   }
  // CHECK-NEXT:   {
  // CHECK-NEXT:     int e;
  // CHECK-NEXT:     e = (int)stop.get_info<sycl::info::event::command_execution_status>();
  // CHECK-NEXT:     /*
  // CHECK-NEXT:     DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:     */
  // CHECK-NEXT:     e = (a = (int *)dpct::dpct_malloc(sizeof(int)), 0);
  // CHECK-NEXT:   }
  // CHECK-NEXT:   int et1, et2;
  // CHECK-NEXT:   et1 = (int)stop.get_info<sycl::info::event::command_execution_status>();
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   et2 = (a = (int *)dpct::dpct_malloc(sizeof(int)), 0);
  // CHECK-NEXT: }

  cudaError_t e = cudaEventQuery(stop);
  MY_ERROR_CHECKER(e);
  MY_ERROR_CHECKER(cudaEventQuery(stop));
  if (cudaErrorNotReady != cudaEventQuery(stop)){}
  while(cudaEventQuery(stop) == cudaErrorNotReady){}
  for(;cudaErrorNotReady == cudaEventQuery(stop);){}
  do{}while(cudaEventQuery(stop) != cudaErrorNotReady);
  {
    int *a;
    cudaError_t e1;
    e1 = cudaEventQuery(stop);
    if (cudaSuccess != cudaEventQuery(stop)) {}
    if (e1 == cudaSuccess){}
    while(e1 != cudaSuccess) {
      e1 = cudaEventQuery(stop);
    }
    for(;e1 != cudaSuccess;){
      e1 = cudaEventQuery(stop);
    }
    {
      cudaError_t e;
      e = cudaEventQuery(stop);
      e = cudaMalloc(&a, sizeof(int));
    }
    cudaError_t et1, et2;
    et1 = cudaEventQuery(stop);
    et2 = cudaMalloc(&a, sizeof(int));
  }

  // CHECK: dev_ct1.queues_wait_and_throw();
  // CHECK-EMPTY:
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaEventDestroy was removed because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaEventDestroy was removed because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaEventDestroy was replaced with 0 because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER(0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaEventDestroy was replaced with 0 because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: et = 0;
  // CHECK-NEXT: }
  cudaDeviceSynchronize();

  cudaEventDestroy(start)  ;   
  cudaEventDestroy(stop)  
    ;   
  MY_ERROR_CHECKER(cudaEventDestroy(start));
  et = cudaEventDestroy(stop);
}

void foo() {
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float elapsed_time;

  cudaDeviceSynchronize();

  int blocks = 32, threads = 32;

  // CHECK: start_ct1 = std::chrono::steady_clock::now();
  // CHECK: MY_CHECKER(0);
  MY_CHECKER(cudaEventRecord(start, 0));
  kernelFunc<<<blocks,threads>>>();
  // CHECK: stop_ct1 = std::chrono::steady_clock::now();
  // CHECK: MY_CHECKER(0);
  MY_CHECKER(cudaEventRecord(stop, 0));

  cudaEventSynchronize(stop);

  // CHECK: MY_CHECKER((elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count(), 0));
  MY_CHECKER(cudaEventElapsedTime(&elapsed_time, start, stop));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  {
    // CHECK: start_ct1 = std::chrono::steady_clock::now();
    // CHECK-NEXT: int err = (start = q_ct1.ext_oneapi_submit_barrier(), 0);
    cudaError_t err = cudaEventRecord(start, 0);
    // CHECK: stop_ct1 = std::chrono::steady_clock::now();
    // CHECK-NEXT: err = (stop = q_ct1.ext_oneapi_submit_barrier(), 0);
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

  // CHECK: start_ct1 = std::chrono::steady_clock::now();
  // CHECK-NEXT: fun(0);
  fun(cudaEventRecord(start, 0));
  kernelFunc<<<blocks,threads>>>();
  // CHECK: stop_ct1 = std::chrono::steady_clock::now();
  // CHECK: fun(0);
  fun(cudaEventRecord(stop, 0));

  cudaEventSynchronize(stop);
  // CHECK: fun((elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count(), 0));
  fun(cudaEventElapsedTime(&elapsed_time, start, stop));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

struct Node {
 // CHECK: sycl::event start;
 // CHECK: std::chrono::time_point<std::chrono::steady_clock> start_ct1;
 cudaEvent_t start;
 // CHECK: sycl::event end;
 // CHECK: std::chrono::time_point<std::chrono::steady_clock> end_ct1;
 cudaEvent_t end;
 // CHECK: sycl::event *ev[100];
 // CHECK: std::chrono::time_point<std::chrono::steady_clock> ev_ct1[100];
 cudaEvent_t *ev[100];
 // CHECK: sycl::event events[100];
 // CHECK: std::chrono::time_point<std::chrono::steady_clock> events_ct1[100];
 cudaEvent_t events[100];
 // CHECK: sycl::event *p_events;
 // CHECK: std::chrono::time_point<std::chrono::steady_clock> p_events_ct1_0;
 // CHECK: std::chrono::time_point<std::chrono::steady_clock> p_events_ct1_1;
 // CHECK: std::chrono::time_point<std::chrono::steady_clock> p_events_ct1_2;
 // CHECK: std::chrono::time_point<std::chrono::steady_clock> p_events_ct1_3;
 cudaEvent_t *p_events;
};

void foo2(Node *n) {
  float elapsed_time;

  // CHECK: n->start_ct1 = std::chrono::steady_clock::now();
  cudaEventRecord(n->start, 0);
  // CHECK: n->start_ct1 = std::chrono::steady_clock::now();
  cudaEventRecord(n->start, 0);
  // do something
  // CHECK: n->end_ct1 = std::chrono::steady_clock::now();
  cudaEventRecord(n->end, 0);
  // CHECK: n->end_ct1 = std::chrono::steady_clock::now();
  cudaEventRecord(n->end, 0);
  // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(n->end_ct1 - n->start_ct1).count();
  cudaEventElapsedTime(&elapsed_time, n->start, n->end);
  {
    int errorCode;
    // CHECK: n->start_ct1 = std::chrono::steady_clock::now();
    // CHECK: MY_CHECKER((n->start = q_ct1.ext_oneapi_submit_barrier(), 0));
    MY_CHECKER(cudaEventRecord(n->start, 0));
    // CHECK: n->start_ct1 = std::chrono::steady_clock::now();
    // CHECK: errorCode = (n->start = q_ct1.ext_oneapi_submit_barrier(), 0);
    errorCode = cudaEventRecord(n->start, 0);
  }

  Node node;
  // CHECK: node.start_ct1 = std::chrono::steady_clock::now();
  cudaEventRecord(node.start, 0);
  // CHECK: node.start_ct1 = std::chrono::steady_clock::now();
  cudaEventRecord(node.start, 0);
  // do something
  // CHECK: node.end_ct1 = std::chrono::steady_clock::now();
  cudaEventRecord(node.end, 0);
  // CHECK: node.end_ct1 = std::chrono::steady_clock::now();
  cudaEventRecord(node.end, 0);
  // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(node.end_ct1 - node.start_ct1).count();
  cudaEventElapsedTime(&elapsed_time, node.start, node.end);
  {
    int errorCode;
    // CHECK: node.start_ct1 = std::chrono::steady_clock::now();
    // CHECK: MY_CHECKER((node.start = q_ct1.ext_oneapi_submit_barrier(), 0));
    MY_CHECKER(cudaEventRecord(node.start, 0));
    // CHECK: node.start_ct1 = std::chrono::steady_clock::now();
    // CHECK: errorCode = (node.start = q_ct1.ext_oneapi_submit_barrier(), 0);
    errorCode = cudaEventRecord(node.start, 0);
  }

  {
    // CHECK: node.events_ct1[0] = std::chrono::steady_clock::now();
    cudaEventRecord(node.events[0]);
    // CHECK: node.events_ct1[0] = std::chrono::steady_clock::now();
    cudaEventRecord(node.events[0]);
    // CHECK: node.events_ct1[23] = std::chrono::steady_clock::now();
    cudaEventRecord(node.events[23]);
    // CHECK: node.events_ct1[23] = std::chrono::steady_clock::now();
    cudaEventRecord(node.events[23]);
    // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(node.events_ct1[23] - node.events_ct1[0]).count();
    cudaEventElapsedTime(&elapsed_time, node.events[0], node.events[23]);
  }

  {
    // CHECK: node.ev_ct1[0] = std::chrono::steady_clock::now();
    cudaEventRecord(*node.ev[0]);
    // CHECK: node.ev_ct1[0] = std::chrono::steady_clock::now();
    cudaEventRecord(*node.ev[0]);
    // CHECK: node.ev_ct1[23] = std::chrono::steady_clock::now();
    cudaEventRecord(*node.ev[23]);
    // CHECK: node.ev_ct1[23] = std::chrono::steady_clock::now();
    cudaEventRecord(*node.ev[23]);
    // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(node.ev_ct1[23] - node.ev_ct1[0]).count();
    cudaEventElapsedTime(&elapsed_time, *node.ev[0], *node.ev[23]);
  }

  {
    // CHECK: (&node)->ev_ct1[0] = std::chrono::steady_clock::now();
    cudaEventRecord(*(&node)->ev[0]);
    // CHECK: (&node)->ev_ct1[0] = std::chrono::steady_clock::now();
    cudaEventRecord(*(&node)->ev[0]);
    // CHECK: (&node)->ev_ct1[23] = std::chrono::steady_clock::now();
    cudaEventRecord(*(&node)->ev[23]);
    // CHECK: (&node)->ev_ct1[23] = std::chrono::steady_clock::now();
    cudaEventRecord(*(&node)->ev[23]);
    // CHECK: elapsed_time = std::chrono::duration<float, std::milli>((&node)->ev_ct1[23] - (&node)->ev_ct1[0]).count();
    cudaEventElapsedTime(&elapsed_time, *(&node)->ev[0], *(&node)->ev[23]);
  }

  {
    // CHECK: n->p_events_ct1_0 = std::chrono::steady_clock::now();
    cudaEventRecord(n->p_events[0]);
    // CHECK: n->p_events_ct1_1 = std::chrono::steady_clock::now();
    cudaEventRecord(n->p_events[1]);
    // CHECK: n->p_events_ct1_2 = std::chrono::steady_clock::now();
    cudaEventRecord(n->p_events[2]);
    // CHECK: n->p_events_ct1_3 = std::chrono::steady_clock::now();
    cudaEventRecord(n->p_events[3]);
  }
}

class C {
  // CHECK: sycl::event start, stop;
  // CHECK-NEXT: std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  // CHECK-NEXT: std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  cudaEvent_t start, stop;
  float elapsed_time;
  void a() {
    // CHECK: start_ct1 = std::chrono::steady_clock::now();
    cudaEventRecord(start, 0);
  }
  void b() {
    // CHECK: stop_ct1 = std::chrono::steady_clock::now();
    cudaEventRecord(stop, 0);
    // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    cudaEventElapsedTime(&elapsed_time, start, stop);
  }
  void c() {
    cudaEventRecord(start, 0);
  }
  void d() {
    // CHECK: stop_ct1 = std::chrono::steady_clock::now();
    cudaEventRecord(stop, 0);
    // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    cudaEventElapsedTime(&elapsed_time, start, stop);
  }
};

struct S {
  cudaEvent_t *events;
  // CHECK: std::chrono::time_point<std::chrono::steady_clock> events_ct1_0;
  // CHECK: std::chrono::time_point<std::chrono::steady_clock> events_ct1_1;
  // CHECK: std::chrono::time_point<std::chrono::steady_clock> events_ct1_2;
  // CHECK: std::chrono::time_point<std::chrono::steady_clock> events_ct1_3;
};

void foo(int n) {
  // CHECK: sycl::event *events = new sycl::event[n];
  cudaEvent_t *events = new cudaEvent_t[n];
  // CHECK: std::chrono::time_point<std::chrono::steady_clock> events_ct1_0;
  // CHECK: std::chrono::time_point<std::chrono::steady_clock> events_ct1_1;
  // CHECK: std::chrono::time_point<std::chrono::steady_clock> events_ct1_2;
  // CHECK: std::chrono::time_point<std::chrono::steady_clock> events_ct1_3;

  // CHECK: events_ct1_0 = std::chrono::steady_clock::now();
  cudaEventRecord(events[0]);
  // CHECK: events_ct1_1 = std::chrono::steady_clock::now();
  cudaEventRecord(events[1]);
  // CHECK: events_ct1_2 = std::chrono::steady_clock::now();
  cudaEventRecord(events[2]);
  // CHECK: events_ct1_3 = std::chrono::steady_clock::now();
  cudaEventRecord(events[3]);

  S s;
  // CHECK: s.events_ct1_0 = std::chrono::steady_clock::now();
  cudaEventRecord(s.events[0]);
  // CHECK: s.events_ct1_1 = std::chrono::steady_clock::now();
  cudaEventRecord(s.events[1]);
  // CHECK: s.events_ct1_2 = std::chrono::steady_clock::now();
  cudaEventRecord(s.events[2]);
  // CHECK: s.events_ct1_3 = std::chrono::steady_clock::now();
  cudaEventRecord(s.events[3]);

  S *s2 = new S;
  // CHECK: s2->events_ct1_0 = std::chrono::steady_clock::now();
  cudaEventRecord(s2->events[0]);
  // CHECK: s2->events_ct1_1 = std::chrono::steady_clock::now();
  cudaEventRecord(s2->events[1]);
  // CHECK: s2->events_ct1_2 = std::chrono::steady_clock::now();
  cudaEventRecord(s2->events[2]);
  // CHECK: s2->events_ct1_3 = std::chrono::steady_clock::now();
  cudaEventRecord(s2->events[3]);
}

void barr(int maxCalls) {
  cudaEvent_t evtStart[maxCalls];
  cudaEvent_t evtEnd[maxCalls];
  float time[maxCalls];
  for (int i = 0; i < maxCalls; i++) {
    cudaEventCreate( &(evtStart[i]) );
    cudaEventCreate( &(evtEnd[i]) );
    time[i] = 0.0;
  }

  // CHECK: evtStart_ct1[0] = std::chrono::steady_clock::now();
  cudaEventRecord( evtStart[0], 0 );
  // CHECK: evtEnd[0].wait();
  kernelFunc<<<1, 1>>>();
  // CHECK: evtEnd_ct1[0] = std::chrono::steady_clock::now();
  cudaEventRecord( evtEnd[0], 0 );

  // CHECK: evtStart_ct1[1] = std::chrono::steady_clock::now();
  cudaEventRecord( evtStart[1], 0 );
  // CHECK: evtEnd[1].wait();
  kernelFunc<<<1, 1>>>();
  // CHECK: evtEnd_ct1[1] = std::chrono::steady_clock::now();
  cudaEventRecord( evtEnd[1], 0 );

  // CHECK: evtStart_ct1[2] = std::chrono::steady_clock::now();
  cudaEventRecord( evtStart[2], 0 );
  // CHECK: evtEnd[2].wait();
  kernelFunc<<<1, 1>>>();
  // CHECK: evtEnd_ct1[2] = std::chrono::steady_clock::now();
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

