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

  // CHECK: dpct::get_default_queue().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS(outputThreadCount, outputThreadCount)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 2),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS(outputThreadCount, outputThreadCount), 2, 0>>>();

  // CHECK: dpct::get_default_queue().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS2(CUDA_NUM_THREADS, CUDA_NUM_THREADS)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 0),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 0)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS2(CUDA_NUM_THREADS, CUDA_NUM_THREADS), 0, 0>>>();

  // CHECK: dpct::get_default_queue().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS3(CUDA_NUM_THREADS, outputThreadCount)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 0),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 0)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS3(CUDA_NUM_THREADS, outputThreadCount), 0, 0>>>();

  // CHECK: dpct::get_default_queue().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS4(outputThreadCount, CUDA_NUM_THREADS)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 2),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS4(outputThreadCount, CUDA_NUM_THREADS), 2, 0>>>();

  // Test if SIGABRT.
  // No check here because the generated code need further fine tune.
  #define MACRO_CALL(a, b) foo_kernel<<<a, b, 0>>>();
  MACRO_CALL(0,0)

// CHECK: #define HANDLE_GPU_ERROR(err) \
// CHECK-NEXT: do \
// CHECK-NEXT: { \
// CHECK-NEXT:     if (err != 0) \
// CHECK-NEXT:     { \
// CHECK-NEXT:         int currentDevice; \
// CHECK-NEXT:         currentDevice = dpct::dev_mgr::instance().current_device_id(); \
// CHECK-NEXT:     } \
// CHECK-NEXT: } while (0)
#define HANDLE_GPU_ERROR(err) \
do \
{ \
    if(err != cudaSuccess) \
    { \
        int currentDevice; \
        cudaGetDevice(&currentDevice); \
    } \
} \
while(0)

HANDLE_GPU_ERROR(0);

// CHECK: #define cbrt(x) pow((double)x,(double)(1.0/3.0))
// CHECK-NEXT: double DD = sqrt(cbrt(5.9)) / sqrt(cbrt(3.2));
#define cbrt(x) pow((double)x,(double)(1.0/3.0))
  double DD = sqrt(cbrt(5.9)) / sqrt(cbrt(3.2));

// CHECK: #define NNBI(x) floor(x+0.5)
// CHECK-NEXT: NNBI(3.0);
#define NNBI(x) floor(x+0.5)
NNBI(3.0);

// CHECK: #define PI acos(-1)
#define PI acos(-1)
// CHECK: double cosine = cos(2 * PI);
double cosine = cos(2 * PI);
}

__global__ void foo2(){
  // CHECK: #define IMUL(a, b) sycl::mul24(a, b)
  // CHECK-NEXT: int vectorBase = IMUL(1, 2);
  #define IMUL(a, b) __mul24(a, b)
  int vectorBase = IMUL(1, 2);
}
