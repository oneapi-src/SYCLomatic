// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cuda-stream-api.sycl.cpp --match-full-lines %s

template <typename T>
// CHECK: void check(T result, char const *const func) try {
void check(T result, char const *const func) {
}

#define checkCudaErrors(val) check((val), #val)

__global__ void kernelFunc() {
}

template<typename FloatN, typename Float>
static void func()
{
  // CHECK: cl::sycl::queue s0, &s1 = s0;
  // CHECK-NEXT: cl::sycl::queue s2, *s3 = &s2;
  // CHECK-NEXT: cl::sycl::queue s4, s5;
  // CHECK-EMPTY:
  cudaStream_t s0, &s1 = s0;
  cudaStream_t s2, *s3 = &s2;
  cudaStream_t s4, s5;

  // CHECK: *(&s0) = cl::sycl::queue{};
  cudaStreamCreate(&s0);
  // CHECK: checkCudaErrors((*(&s1) = cl::sycl::queue{}, 0));
  checkCudaErrors(cudaStreamCreate(&s1));

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1014:{{[0-9]+}}: Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(&s2) = cl::sycl::queue{};
  cudaStreamCreateWithFlags(&s2, cudaStreamDefault);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1014:{{[0-9]+}}: Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((*(s3) = cl::sycl::queue{}, 0));
  checkCudaErrors(cudaStreamCreateWithFlags(s3, cudaStreamNonBlocking));

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1014:{{[0-9]+}}: Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(&s4) = cl::sycl::queue{};
  cudaStreamCreateWithPriority(&s4, cudaStreamDefault, 2);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1014:{{[0-9]+}}: Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((*(&s5) = cl::sycl::queue{}, 0));
  checkCudaErrors(cudaStreamCreateWithPriority(&s5, cudaStreamNonBlocking, 3));

  int priority_low;
  int priority_hi;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1014:{{[0-9]+}}: Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(&priority_low) = 0, *(&priority_hi) = 0;
  cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi);

  int priority;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1014:{{[0-9]+}}: Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors(*(&priority) = 0);
  checkCudaErrors(cudaStreamGetPriority(s0, &priority));

  kernelFunc<<<16, 32, 0, s0>>>();
  kernelFunc<<<16, 32, 0, s1>>>();
  kernelFunc<<<16, 32, 0, s2>>>();
  kernelFunc<<<16, 32, 0, *s3>>>();

  // CHECK: s0.wait();
  cudaStreamSynchronize(s0);
  // CHECK: checkCudaErrors((s1.wait(), 0));
  checkCudaErrors(cudaStreamSynchronize(s1));
  // CHECK: s2.wait();
  cudaStreamSynchronize(s2);
  // CHECK: checkCudaErrors((*s3.wait(), 0));
  checkCudaErrors(cudaStreamSynchronize(*s3));

  // CHECK: s0 = cl::sycl::queue{};
  cudaStreamDestroy(s0);
  // CHECK: checkCudaErrors((s1 = cl::sycl::queue{}, 0));
  checkCudaErrors(cudaStreamDestroy(s1));
  // CHECK: s2 = cl::sycl::queue{};
  cudaStreamDestroy(s2);
  // CHECK: checkCudaErrors((*s3 = cl::sycl::queue{}, 0));
  checkCudaErrors(cudaStreamDestroy(*s3));
}
