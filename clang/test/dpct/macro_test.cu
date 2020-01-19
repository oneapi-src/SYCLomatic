// RUN: cat %s > %T/macro_test.cu
// RUN: cd %T
// RUN: dpct -out-root %T macro_test.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/macro_test.dp.cpp --match-full-lines macro_test.cu
#define CUDA_NUM_THREADS 1024+32
#define GET_BLOCKS(n,t)  1+n+t-1
#define GET_BLOCKS2(n,t) 1+n+t
#define GET_BLOCKS3(n,t) n+t-1
#define GET_BLOCKS4(n,t) n+t

__global__ void foo_kernel() {}

void foo() {
  int outputThreadCount = 512;

  // CHECK: dpct::get_default_queue_wait().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS(outputThreadCount, outputThreadCount)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 2),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS(outputThreadCount, outputThreadCount), 2, 0>>>();

  // CHECK: dpct::get_default_queue_wait().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS2(CUDA_NUM_THREADS, CUDA_NUM_THREADS)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 0),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 0)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS2(CUDA_NUM_THREADS, CUDA_NUM_THREADS), 0, 0>>>();

  // CHECK: dpct::get_default_queue_wait().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS3(CUDA_NUM_THREADS, outputThreadCount)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 0),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 0)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS3(CUDA_NUM_THREADS, outputThreadCount), 0, 0>>>();

  // CHECK: dpct::get_default_queue_wait().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS4(outputThreadCount, CUDA_NUM_THREADS)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 2),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS4(outputThreadCount, CUDA_NUM_THREADS), 2, 0>>>();
}