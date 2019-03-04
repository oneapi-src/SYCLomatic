// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cuda-event-api.sycl.cpp --match-full-lines %s

#include <stdio.h>

__global__ void kernelFunc()
{
}

int main(int argc, char* argv[]) {
  // CHECK: cl::sycl::event start, stop;
  // CHECK-EMPTY:
  // CHECK-EMPTY:
  // CHECK-NEXT: float elapsed_time;
  // CHECK-EMPTY:
  // CHECK-NEXT: syclct::get_device_manager().current_device().queues_wait_and_throw();
  // CHECK-EMPTY:
  // CHECK-NEXT: int blocks = 32, threads = 32;
  cudaEvent_t start, stop;

  cudaEventCreate(&start)  
    ;   
  cudaEventCreate(&stop)  ;   

  float elapsed_time;

  cudaDeviceSynchronize();

  int blocks = 32, threads = 32;

  // kernel call without sync
  // CHECK: {
  // CHECK-NEXT:   syclct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(blocks, 1, 1) * cl::sycl::range<3>(threads, 1, 1)), cl::sycl::range<3>(threads, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: };
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1012:{{[a-f0-9]+}}: The tool detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. Depending on your goals you may want to change the way the time is measured.
  // CHECK-NEXT: */
  // CHECK-NEXT: auto syclct_start_{{[a-f0-9]+}} = clock();
  cudaEventRecord(start, 0);

  // kernel call without sync
  // CHECK: {
  // CHECK-NEXT:   syclct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(blocks, 1, 1) * cl::sycl::range<3>(threads, 1, 1)), cl::sycl::range<3>(threads, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: };
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1012:{{[a-f0-9]+}}: The tool detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. Depending on your goals you may want to change the way the time is measured.
  // CHECK-NEXT: */
  // CHECK-NEXT: syclct_start_{{[a-f0-9]+}} = clock();
  cudaEventRecord(start, 0);

  // kernel call with sync
  // CHECK: {
  // CHECK-NEXT:   syclct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(blocks, 1, 1) * cl::sycl::range<3>(threads, 1, 1)), cl::sycl::range<3>(threads, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     }).wait();
  // CHECK-NEXT: };
  kernelFunc<<<blocks,threads>>>();
  // CHECK: {
  // CHECK-NEXT:   syclct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(blocks, 1, 1) * cl::sycl::range<3>(threads, 1, 1)), cl::sycl::range<3>(threads, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     }).wait();
  // CHECK-NEXT: };
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1012:{{[a-f0-9]+}}: The tool detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. Depending on your goals you may want to change the way the time is measured.
  // CHECK-NEXT: */
  // CHECK-NEXT: auto syclct_stop_{{[a-f0-9]+}} = clock();
  cudaEventRecord(stop, 0);

  // kernel call without sync
  // CHECK: {
  // CHECK-NEXT:   syclct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(blocks, 1, 1) * cl::sycl::range<3>(threads, 1, 1)), cl::sycl::range<3>(threads, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     }).wait();
  // CHECK-NEXT: };
  kernelFunc<<<blocks,threads>>>();

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1012:{{[a-f0-9]+}}: The tool detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. Depending on your goals you may want to change the way the time is measured.
  // CHECK-NEXT: */
  // CHECK-NEXT: syclct_stop_{{[a-f0-9]+}} = clock();
  cudaEventRecord(stop, 0);

  // CHECK: stop.wait_and_throw();
  cudaEventSynchronize(stop);

  // kernel call without sync
  // CHECK: {
  // CHECK-NEXT:   syclct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(blocks, 1, 1) * cl::sycl::range<3>(threads, 1, 1)), cl::sycl::range<3>(threads, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: };
  kernelFunc<<<blocks,threads>>>();

  // CHECK: *(&elapsed_time) = (float)(syclct_stop_{{[a-f0-9]+}} - syclct_start_{{[a-f0-9]+}}) / CLOCKS_PER_SEC * 1000;
  cudaEventElapsedTime(&elapsed_time, start, stop);

  // kernel call without sync
  // CHECK: {
  // CHECK-NEXT:   syclct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(blocks, 1, 1) * cl::sycl::range<3>(threads, 1, 1)), cl::sycl::range<3>(threads, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: };
  kernelFunc<<<blocks,threads>>>();

  // CHECK: syclct::get_device_manager().current_device().queues_wait_and_throw();
  // CHECK-EMPTY:
  // CHECK-NEXT: }
  cudaDeviceSynchronize();

  cudaEventDestroy(start)  ;   
  cudaEventDestroy(stop)  
    ;   
}
