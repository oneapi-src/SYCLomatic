// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cuda-stream-api.sycl.cpp --match-full-lines %s

#include <list>

template <typename T>
// CHECK: void check(T result, char const *const func) try {
void check(T result, char const *const func) {
}

#define checkCudaErrors(val) check((val), #val)

__global__ void kernelFunc() {
}

void process(cudaStream_t st, char *data, cudaError_t status) {}

template<typename T>
void callback(cudaStream_t st, cudaError_t status, void *vp) {
  T *data = static_cast<T *>( vp);
  process(st, data, status);
}

template<typename FloatN, typename Float>
static void func()
{
  // TODO: 1CHECK: std::list<cl::sycl::queue> streams;
  std::list<cudaStream_t> streams;
  for (auto Iter = streams.begin(); Iter != streams.end(); ++Iter)
    // CHECK: (*Iter = cl::sycl::queue{}, 0);
    cudaStreamDestroy(*Iter);

  // CHECK: cl::sycl::queue s0, &s1 = s0;
  // CHECK-NEXT: cl::sycl::queue s2, *s3 = &s2;
  // CHECK-NEXT: cl::sycl::queue s4, s5;
  // CHECK-EMPTY:
  // CHECK-NEXT: {
  // CHECK-NEXT:   syclct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1)), cl::sycl::range<3>(32, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  cudaStream_t s0, &s1 = s0;
  cudaStream_t s2, *s3 = &s2;
  cudaStream_t s4, s5;

  cudaStreamCreate(&s0);
  kernelFunc<<<16, 32, 0>>>();

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((0, 0));
  checkCudaErrors(cudaStreamCreate(&s1));

  // CHECK: {
  // CHECK-NEXT:   s0.submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1)), cl::sycl::range<3>(32, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  kernelFunc<<<16, 32, 0, s0>>>();

  // CHECK: {
  // CHECK-NEXT:   s1.submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1)), cl::sycl::range<3>(32, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  kernelFunc<<<16, 32, 0, s1>>>();

  {
    // CHECK: /*
    // CHECK-NEXT: SYCLCT1014:{{[0-9]+}}: Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: s2 = cl::sycl::queue{};
    cudaStreamCreateWithFlags(&s2, cudaStreamDefault);

    // CHECK: /*
    // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: SYCLCT1014:{{[0-9]+}}: Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((*(s3) = cl::sycl::queue{}, 0));
    checkCudaErrors(cudaStreamCreateWithFlags(s3, cudaStreamNonBlocking));

    // CHECK: {
    // CHECK-NEXT:   s2.submit(
    // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
    // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
    // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1)), cl::sycl::range<3>(32, 1, 1)),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
    // CHECK-NEXT:           kernelFunc();
    // CHECK-NEXT:         });
    // CHECK-NEXT:     });
    // CHECK-NEXT: }
    kernelFunc<<<16, 32, 0, s2>>>();

    // CHECK: {
    // CHECK-NEXT:   (*s3).submit(
    // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
    // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
    // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1)), cl::sycl::range<3>(32, 1, 1)),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
    // CHECK-NEXT:           kernelFunc();
    // CHECK-NEXT:         });
    // CHECK-NEXT:     });
    // CHECK-NEXT: }
    kernelFunc<<<16, 32, 0, *s3>>>();

    // CHECK: s2 = cl::sycl::queue{};
    cudaStreamDestroy(s2);
    // CHECK: /*
    // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((*s3 = cl::sycl::queue{}, 0));
    checkCudaErrors(cudaStreamDestroy(*s3));
  }

  {
    {
      // CHECK: /*
      // CHECK-NEXT: SYCLCT1014:{{[0-9]+}}: Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.
      // CHECK-NEXT: */
      // CHECK-NEXT: s4 = cl::sycl::queue{};
      cudaStreamCreateWithPriority(&s4, cudaStreamDefault, 2);

      // CHECK: /*
      // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
      // CHECK-NEXT: */
      // CHECK-NEXT: /*
      // CHECK-NEXT: SYCLCT1014:{{[0-9]+}}: Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.
      // CHECK-NEXT: */
      // CHECK-NEXT: checkCudaErrors((s5 = cl::sycl::queue{}, 0));
      checkCudaErrors(cudaStreamCreateWithPriority(&s5, cudaStreamNonBlocking, 3));

      // CHECK: {
      // CHECK-NEXT:   s4.submit(
      // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
      // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
      // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1)), cl::sycl::range<3>(32, 1, 1)),
      // CHECK-NEXT:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
      // CHECK-NEXT:           kernelFunc();
      // CHECK-NEXT:         });
      // CHECK-NEXT:     });
      // CHECK-NEXT: }
      kernelFunc<<<16, 32, 0, s4>>>();
      // CHECK: {
      // CHECK-NEXT:   s5.submit(
      // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
      // CHECK-NEXT:       cgh.parallel_for<syclct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
      // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1)), cl::sycl::range<3>(32, 1, 1)),
      // CHECK-NEXT:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
      // CHECK-NEXT:           kernelFunc();
      // CHECK-NEXT:         });
      // CHECK-NEXT:     });
      // CHECK-NEXT: }
      kernelFunc<<<16, 32, 0, s5>>>();

      // CHECK: s4 = cl::sycl::queue{};
      cudaStreamDestroy(s4);
      // CHECK: /*
      // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
      // CHECK-NEXT: */
      // CHECK-NEXT: checkCudaErrors((s5 = cl::sycl::queue{}, 0));
      checkCudaErrors(cudaStreamDestroy(s5));
    }
  }

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

  char str[256];

  unsigned int flags = 0;
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: int status = (s0.wait(), callback<char *>(s0, 0, str), 0);
  // CHECK-NEXT: s1.wait(), callback<char*>(s1, 0, str);
  cudaError_t status = cudaStreamAddCallback(s0, callback<char *>, str, flags);
  cudaStreamAddCallback(s1, callback<char*>, str, flags);

  // CHECK: s0.wait();
  cudaStreamSynchronize(s0);
  // CHECK: checkCudaErrors((s1.wait(), 0));
  // CHECK-EMPTY:
  checkCudaErrors(cudaStreamSynchronize(s1));

  cudaStreamDestroy(s0);
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((0, 0));
  checkCudaErrors(cudaStreamDestroy(s1));
}
