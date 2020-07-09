// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda-event-api.dp.cpp --match-full-lines %s

#include <stdio.h>

template <typename T>
// CHECK: void check(T result, char const *const func) {
void check(T result, char const *const func) {
}

#define checkCudaErrors(val) check((val), #val)

//CHECK: /*
//CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaEventCreate was removed, because this call is redundant in DPC++.
//CHECK-NEXT: */
#define CudaEvent(X)\
  cudaEventCreate(&X)

#define cudaCheck(stmt) do {                         \
  cudaError_t err = stmt;                            \
  if (err != cudaSuccess) {                          \
    char msg[256];                                   \
    sprintf(msg, "%s in file %s, function %s, line %d\n", #stmt,__FILE__,__FUNCTION__,__LINE__); \
  }                                                  \
} while(0)

__global__ void kernelFunc()
{
}

int main(int argc, char* argv[]) {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: sycl::event start, stop;
  // CHECK-EMPTY:
  // CHECK: /*
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaEventCreate was removed, because this call is redundant in DPC++.
  // CHECK: */
  // CHECK: /*
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaEventCreate was removed, because this call is redundant in DPC++.
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

  CudaEvent(start);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0, because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors(0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaEventCreate was replaced with 0, because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: int et = 0;
  checkCudaErrors(cudaEventCreate(&start));
  cudaError_t et = cudaEventCreate(&stop);


  // kernel call without sync
  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: auto start_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(start, 0);

  // kernel call without sync
  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: start_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(start, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[0-9a-f]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT: */
  // CHECK-NEXT: start_ct1 = std::chrono::high_resolution_clock::now();
  // CHECK-NEXT: checkCudaErrors(0);
  checkCudaErrors(cudaEventRecord(start, 0));

  // CHECK: if (0)
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   start_ct1 = std::chrono::high_resolution_clock::now();
  // CHECK-NEXT:   checkCudaErrors(0);
  if (0)
    checkCudaErrors(cudaEventRecord(start, 0));

  // kernel call with sync
  // CHECK:   stop = q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: stop.wait();
  kernelFunc<<<blocks,threads>>>();
  // CHECK:   stop = q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: stop.wait();
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: auto stop_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(stop, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT: */
  // CHECK-NEXT: stop_ct1 = std::chrono::high_resolution_clock::now();
  // CHECK-NEXT: checkCudaErrors(0);
  checkCudaErrors(cudaEventRecord(stop, 0));

  // CHECK: if (1)
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   stop_ct1 = std::chrono::high_resolution_clock::now();
  // CHECK-NEXT:   checkCudaErrors(0);
  if (1)
    checkCudaErrors(cudaEventRecord(stop, 0));

  // kernel call without sync
  // CHECK:   stop = q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: stop.wait();
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: stop_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(stop, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT: */
  // CHECK-NEXT: stop_ct1 = std::chrono::high_resolution_clock::now();
  // CHECK-NEXT: checkCudaErrors(0);
  checkCudaErrors(cudaEventRecord(stop, 0));

  // CHECK: if (0)
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   stop_ct1 = std::chrono::high_resolution_clock::now();
  // CHECK-NEXT:   checkCudaErrors(0);
  if (0)
    checkCudaErrors(cudaEventRecord(stop, 0));

  // CHECK: stop.wait_and_throw();
  cudaEventSynchronize(stop);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9a-z]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((stop.wait_and_throw(), 0));
  checkCudaErrors(cudaEventSynchronize(stop));

  // kernel call without sync
  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernelFunc<<<blocks,threads>>>();

  // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  cudaEventElapsedTime(&elapsed_time, start, stop);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9a-z]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count(), 0));
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));

  // kernel call without sync
  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernelFunc<<<blocks,threads>>>();


  // CHECK: int e = (int)stop.get_info<sycl::info::event::command_execution_status>();
  // CHECK-NEXT: checkCudaErrors((int)stop.get_info<sycl::info::event::command_execution_status>());
  // CHECK-NEXT: if (0 == (int)stop.get_info<sycl::info::event::command_execution_status>()){}
  // CHECK-NEXT: while((int)stop.get_info<sycl::info::event::command_execution_status>() != 0){}
  // CHECK-NEXT: for(;0 != (int)stop.get_info<sycl::info::event::command_execution_status>();){}
  // CHECK-NEXT: do{}while((int)stop.get_info<sycl::info::event::command_execution_status>() == 0);
  cudaError_t e = cudaEventQuery(stop);
  checkCudaErrors(cudaEventQuery(stop));
  if (cudaErrorNotReady != cudaEventQuery(stop)){}
  while(cudaEventQuery(stop) == cudaErrorNotReady){}
  for(;cudaErrorNotReady == cudaEventQuery(stop);){}
  do{}while(cudaEventQuery(stop) != cudaErrorNotReady);

  // CHECK: dev_ct1.queues_wait_and_throw();
  // CHECK-EMPTY:
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaEventDestroy was removed, because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaEventDestroy was removed, because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaEventDestroy was replaced with 0, because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors(0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaEventDestroy was replaced with 0, because this call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: et = 0;
  // CHECK-NEXT: }
  cudaDeviceSynchronize();

  cudaEventDestroy(start)  ;   
  cudaEventDestroy(stop)  
    ;   
  checkCudaErrors(cudaEventDestroy(start));
  et = cudaEventDestroy(stop);
}

void foo() {
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float elapsed_time;

  cudaDeviceSynchronize();

  int blocks = 32, threads = 32;

  // CHECK: auto start_ct1 = std::chrono::high_resolution_clock::now();
  // CHECK: cudaCheck(0);
  cudaCheck(cudaEventRecord(start, 0));
  kernelFunc<<<blocks,threads>>>();
  // CHECK: auto stop_ct1 = std::chrono::high_resolution_clock::now();
  // CHECK: cudaCheck(0);
  cudaCheck(cudaEventRecord(stop, 0));

  cudaEventSynchronize(stop);

  // CHECK: cudaCheck((elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count(), 0));
  cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  {
    // CHECK: auto start_ct1 = std::chrono::high_resolution_clock::now();
    // CHECK-NEXT: int err = 0;
    cudaError_t err = cudaEventRecord(start, 0);
    // CHECK: auto stop_ct1 = std::chrono::high_resolution_clock::now();
    // CHECK-NEXT: err = 0;
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

  // CHECK: auto start_ct1 = std::chrono::high_resolution_clock::now();
  // CHECK: fun(0);
  fun(cudaEventRecord(start, 0));
  kernelFunc<<<blocks,threads>>>();
  // CHECK: auto stop_ct1 = std::chrono::high_resolution_clock::now();
  // CHECK: fun(0);
  fun(cudaEventRecord(stop, 0));

  cudaEventSynchronize(stop);

  // CHECK: fun((elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count(), 0));
  fun(cudaEventElapsedTime(&elapsed_time, start, stop));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

struct Node {
 cudaEvent_t start;
 cudaEvent_t end;
 struct {
  cudaEvent_t *ev[100];
  cudaEvent_t events[100];
 };
};

void foo2(Node *n) {
  float elapsed_time;

  // CHECK: auto n_start_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(n->start, 0);
  // CHECK: n_start_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(n->start, 0);
  // do something
  // CHECK: auto n_end_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(n->end, 0);
  // CHECK: n_end_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(n->end, 0);
  // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(n_end_ct1 - n_start_ct1).count();
  cudaEventElapsedTime(&elapsed_time, n->start, n->end);
  {
    int errorCode;
    // CHECK: auto n_start_ct1 = std::chrono::high_resolution_clock::now();
    // CHECK: cudaCheck(0);
    cudaCheck(cudaEventRecord(n->start, 0));
    // CHECK: n_start_ct1 = std::chrono::high_resolution_clock::now();
    // CHECK: errorCode = 0;
    errorCode = cudaEventRecord(n->start, 0);
  }

  Node node;
  // CHECK: auto node_start_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(node.start, 0);
  // CHECK: node_start_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(node.start, 0);
  // do something
  // CHECK: auto node_end_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(node.end, 0);
  // CHECK: node_end_ct1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(node.end, 0);
  // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(node_end_ct1 - node_start_ct1).count();
  cudaEventElapsedTime(&elapsed_time, node.start, node.end);
  {
    int errorCode;
    // CHECK: auto node_start_ct1 = std::chrono::high_resolution_clock::now();
    // CHECK: cudaCheck(0);
    cudaCheck(cudaEventRecord(node.start, 0));
    // CHECK: node_start_ct1 = std::chrono::high_resolution_clock::now();
    // CHECK: errorCode = 0;
    errorCode = cudaEventRecord(node.start, 0);
  }

  {
    // CHECK: auto node_events_0_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(node.events[0]);
    // CHECK: node_events_0_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(node.events[0]);
    // CHECK: auto node_events_23_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(node.events[23]);
    // CHECK: node_events_23_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(node.events[23]);
    // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(node_events_23_ct1 - node_events_0_ct1).count();
    cudaEventElapsedTime(&elapsed_time, node.events[0], node.events[23]);
  }

  {
    // CHECK: auto node_ev_0_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(*node.ev[0]);
    // CHECK: node_ev_0_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(*node.ev[0]);
    // CHECK: auto node_ev_23_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(*node.ev[23]);
    // CHECK: node_ev_23_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(*node.ev[23]);
    // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(node_ev_23_ct1 - node_ev_0_ct1).count();
    cudaEventElapsedTime(&elapsed_time, *node.ev[0], *node.ev[23]);
  }

  {
    // CHECK: auto node_ev_0_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(*(&node)->ev[0]);
    // CHECK: node_ev_0_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(*(&node)->ev[0]);
    // CHECK: auto node_ev_23_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(*(&node)->ev[23]);
    // CHECK: node_ev_23_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(*(&node)->ev[23]);
    // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(node_ev_23_ct1 - node_ev_0_ct1).count();
    cudaEventElapsedTime(&elapsed_time, *(&node)->ev[0], *(&node)->ev[23]);
  }
}

class C {
  // CHECK: sycl::event start, stop;
  // CHECK-NEXT: std::chrono::time_point<std::chrono::high_resolution_clock> start_ct1;
  // CHECK-NEXT: std::chrono::time_point<std::chrono::high_resolution_clock> stop_ct1;
  cudaEvent_t start, stop;
  float elapsed_time;
  void a() {
    // CHECK: start_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start, 0);
  }
  void b() {
    // CHECK: stop_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(stop, 0);
    // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    cudaEventElapsedTime(&elapsed_time, start, stop);
  }
  void c() {
    cudaEventRecord(start, 0);
  }
  void d() {
    // CHECK: stop_ct1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(stop, 0);
    // CHECK: elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    cudaEventElapsedTime(&elapsed_time, start, stop);
  }
};
