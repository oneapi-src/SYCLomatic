// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda-event-api.dp.cpp --match-full-lines %s

#include <stdio.h>

template <typename T>
// CHECK: void check(T result, char const *const func) {
void check(T result, char const *const func) {
}

#define checkCudaErrors(val) check((val), #val)

#define CudaEvent(X)\
  cudaEventCreate(&X)

__global__ void kernelFunc()
{
}

int main(int argc, char* argv[]) {
  // CHECK: cl::sycl::event start, stop;
  // CHECK-EMPTY:
  // CHECK-EMPTY:
  // CHECK-NEXT: float elapsed_time;
  // CHECK-EMPTY:
  // CHECK-NEXT: dpct::get_device_manager().current_device().queues_wait_and_throw();
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

  // CHECK: checkCudaErrors(0);
  // CHECK-NEXT: int et = 0;
  checkCudaErrors(cudaEventCreate(&start));
  cudaError_t et = cudaEventCreate(&stop);


  // kernel call without sync
  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, blocks) * cl::sycl::range<3>(1, 1, threads), cl::sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: auto start_ct1 = clock();
  cudaEventRecord(start, 0);

  // kernel call without sync
  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, blocks) * cl::sycl::range<3>(1, 1, threads), cl::sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: start_ct1 = clock();
  cudaEventRecord(start, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[0-9a-f]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT: */
  // CHECK-NEXT: start_ct1 = clock(), checkCudaErrors(0);
  checkCudaErrors(cudaEventRecord(start, 0));

  // CHECK: if (0)
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   start_ct1 = clock(), checkCudaErrors(0);
  if (0)
    checkCudaErrors(cudaEventRecord(start, 0));

  // kernel call with sync
  // CHECK:   stop = dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, blocks) * cl::sycl::range<3>(1, 1, threads), cl::sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: stop.wait();
  kernelFunc<<<blocks,threads>>>();
  // CHECK:   stop = dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, blocks) * cl::sycl::range<3>(1, 1, threads), cl::sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: stop.wait();
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: auto stop_ct1 = clock();
  cudaEventRecord(stop, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT: */
  // CHECK-NEXT: stop_ct1 = clock(), checkCudaErrors(0);
  checkCudaErrors(cudaEventRecord(stop, 0));

  // CHECK: if (1)
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   stop_ct1 = clock(), checkCudaErrors(0);
  if (1)
    checkCudaErrors(cudaEventRecord(stop, 0));

  // kernel call without sync
  // CHECK:   stop = dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, blocks) * cl::sycl::range<3>(1, 1, threads), cl::sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: stop.wait();
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[a-f0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: stop_ct1 = clock();
  cudaEventRecord(stop, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT: */
  // CHECK-NEXT: stop_ct1 = clock(), checkCudaErrors(0);
  checkCudaErrors(cudaEventRecord(stop, 0));

  // CHECK: if (0)
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1012:{{[0-9a-z]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   /*
  // CHECK-NEXT:   DPCT1024:{{[0-9a-f]+}}: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   stop_ct1 = clock(), checkCudaErrors(0);
  if (0)
    checkCudaErrors(cudaEventRecord(stop, 0));

  // CHECK: stop.wait_and_throw();
  cudaEventSynchronize(stop);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9a-z]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((stop.wait_and_throw(), 0));
  checkCudaErrors(cudaEventSynchronize(stop));

  // kernel call without sync
  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, blocks) * cl::sycl::range<3>(1, 1, threads), cl::sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernelFunc<<<blocks,threads>>>();

  // CHECK: *(&elapsed_time) = (float)(stop_ct1 - start_ct1) / CLOCKS_PER_SEC * 1000;
  cudaEventElapsedTime(&elapsed_time, start, stop);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9a-z]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((*(&elapsed_time) = (float)(stop_ct1 - start_ct1) / CLOCKS_PER_SEC * 1000, 0));
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));

  // kernel call without sync
  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, blocks) * cl::sycl::range<3>(1, 1, threads), cl::sycl::range<3>(1, 1, threads)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernelFunc<<<blocks,threads>>>();

  // CHECK: dpct::get_device_manager().current_device().queues_wait_and_throw();
  // CHECK-EMPTY:
  // CHECK-NEXT: checkCudaErrors(0);
  // CHECK-NEXT: et = 0;
  // CHECK-NEXT: }
  cudaDeviceSynchronize();

  cudaEventDestroy(start)  ;   
  cudaEventDestroy(stop)  
    ;   
  checkCudaErrors(cudaEventDestroy(start));
  et = cudaEventDestroy(stop);
}
