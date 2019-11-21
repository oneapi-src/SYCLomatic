// FIXME:
// UNSUPPORTED: -windows-
// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda-stream-api.dp.cpp --match-full-lines %s

#include <list>
// CHECK: using queue_p = cl::sycl::queue *;

template <typename T>
// CHECK: void check(T result, char const *const func) {
void check(T result, char const *const func) {
}

#define checkCudaErrors(val) check((val), #val)

__global__ void kernelFunc() {
}

// CHECK: void process(queue_p st, char *data, int status) {}
void process(cudaStream_t st, char *data, cudaError_t status) {}

template<typename T>
// CHECK: void callback(queue_p st, int status, void *vp) {
void callback(cudaStream_t st, cudaError_t status, void *vp) {
  T *data = static_cast<T *>( vp);
  process(st, data, status);
}

template<typename FloatN, typename Float>
static void func()
{
  // CHECK: std::list<queue_p> streams;
  std::list<cudaStream_t> streams;
  for (auto Iter = streams.begin(); Iter != streams.end(); ++Iter)
    // CHECK: *Iter = new cl::sycl::queue{};
    cudaStreamCreate(&*Iter);
  for (auto Iter = streams.begin(); Iter != streams.end(); ++Iter)
    // CHECK: delete *Iter;
    cudaStreamDestroy(*Iter);

  // CHECK: queue_p s0, &s1 = s0;
  // CHECK-NEXT: queue_p s2, *s3 = &s2;
  // CHECK-NEXT: queue_p s4, s5;
  // CHECK-EMPTY:
  cudaStream_t s0, &s1 = s0;
  cudaStream_t s2, *s3 = &s2;
  cudaStream_t s4, s5;

  // CHECK: if (1)
  // CHECK-NEXT: s0 = new cl::sycl::queue{};
  if (1)
    cudaStreamCreate(&s0);

  // CHECK: while (0)
  // CHECK-NEXT: s0 = new cl::sycl::queue{};
  while (0)
    cudaStreamCreate(&s0);

  // CHECK: do
  // CHECK-NEXT: s0 = new cl::sycl::queue{};
  // CHECK: while (0);
  do
    cudaStreamCreate(&s0);
  while (0);

  // CHECK: for (; 0; )
  // CHECK-NEXT: s0 = new cl::sycl::queue{};
  for (; 0; )
    cudaStreamCreate(&s0);

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(32, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernelFunc<<<16, 32, 0>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((s1 = new cl::sycl::queue{}, 0));
  checkCudaErrors(cudaStreamCreate(&s1));

  // CHECK:   s0->submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(32, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernelFunc<<<16, 32, 0, s0>>>();

  // CHECK:   s1->submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(32, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernelFunc<<<16, 32, 0, s1>>>();

  {
    // CHECK: /*
    // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag/priority options.
    // CHECK-NEXT: */
    // CHECK-NEXT: s2 = new cl::sycl::queue{};
    cudaStreamCreateWithFlags(&s2, cudaStreamDefault);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag/priority options.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((*(s3) = new cl::sycl::queue{}, 0));
    checkCudaErrors(cudaStreamCreateWithFlags(s3, cudaStreamNonBlocking));

    // CHECK:   s2->submit(
    // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
    // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1);
    // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(32, 1, 1);
    // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernelFunc();
    // CHECK-NEXT:         });
    // CHECK-NEXT:     });
    kernelFunc<<<16, 32, 0, s2>>>();

    // CHECK:   (*s3)->submit(
    // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
    // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1);
    // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(32, 1, 1);
    // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernelFunc();
    // CHECK-NEXT:         });
    // CHECK-NEXT:     });
    kernelFunc<<<16, 32, 0, *s3>>>();

    // CHECK: delete s2;
    cudaStreamDestroy(s2);
    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: checkCudaErrors((delete *s3, 0));
    checkCudaErrors(cudaStreamDestroy(*s3));
  }

  {
    {
      // CHECK: /*
      // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag/priority options.
      // CHECK-NEXT: */
      // CHECK-NEXT: s4 = new cl::sycl::queue{};
      cudaStreamCreateWithPriority(&s4, cudaStreamDefault, 2);

      // CHECK: /*
      // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      // CHECK-NEXT: */
      // CHECK-NEXT: /*
      // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag/priority options.
      // CHECK-NEXT: */
      // CHECK-NEXT: checkCudaErrors((s5 = new cl::sycl::queue{}, 0));
      checkCudaErrors(cudaStreamCreateWithPriority(&s5, cudaStreamNonBlocking, 3));

      // CHECK:   s4->submit(
      // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
      // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1);
      // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(32, 1, 1);
      // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
      // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
      // CHECK-NEXT:           kernelFunc();
      // CHECK-NEXT:         });
      // CHECK-NEXT:     });
      kernelFunc<<<16, 32, 0, s4>>>();
      // CHECK:   s5->submit(
      // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
      // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(16, 1, 1) * cl::sycl::range<3>(32, 1, 1);
      // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(32, 1, 1);
      // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
      // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
      // CHECK-NEXT:           kernelFunc();
      // CHECK-NEXT:         });
      // CHECK-NEXT:     });
      kernelFunc<<<16, 32, 0, s5>>>();

      // CHECK: delete s4;
      cudaStreamDestroy(s4);
      // CHECK: /*
      // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      // CHECK-NEXT: */
      // CHECK-NEXT: checkCudaErrors((delete s5, 0));
      checkCudaErrors(cudaStreamDestroy(s5));
    }
  }

  int priority_low;
  int priority_hi;
  // CHECK: /*
  // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The flag/priority options are not supported for SYCL queues; the output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(&priority_low) = 0, *(&priority_hi) = 0;
  cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi);
  // CHECK: /*
  // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The flag/priority options are not supported for SYCL queues; the output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((*(&priority_low) = 0, *(&priority_hi) = 0, 0));
  checkCudaErrors(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi));

  int priority;
  // CHECK: /*
  // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The flag/priority options are not supported for SYCL queues; the output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((*(&priority) = 0, 0));
  checkCudaErrors(cudaStreamGetPriority(s0, &priority));

  char str[256];

  unsigned int flags = 0;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: int status = (std::async([&]() { s0->wait(); callback<char *>(s0, 0, str); }), 0);
  // CHECK-NEXT: std::async([&]() { s1->wait(); callback<char*>(s1, 0, str); });
  cudaError_t status = cudaStreamAddCallback(s0, callback<char *>, str, flags);
  cudaStreamAddCallback(s1, callback<char*>, str, flags);

  // CHECK: /*
  // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The flag/priority options are not supported for SYCL queues; the output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((*(&flags) = 0, 0));
  checkCudaErrors(cudaStreamGetFlags(s0, &flags));

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaStreamAttachMemAsync was removed. DPC++ currently doesn't support associating USM with a specific queue.
  // CHECK-NEXT: */
  cudaStreamAttachMemAsync(s0, nullptr);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaStreamAttachMemAsync was replaced with 0. DPC++ currently doesn't support associating USM with a specific queue.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors(0);
  checkCudaErrors(cudaStreamAttachMemAsync(s0, nullptr));

  cudaEvent_t e;
  // CHECK; e.wait();
  cudaStreamWaitEvent(s0, e, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaStreamQuery was removed. DPC++ currently doesn't support query operations on queues.
  // CHECK-NEXT: */
  cudaStreamQuery(s0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaStreamQuery was replaced with 0. DPC++ currently doesn't support query operations on queues.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors(0);
  checkCudaErrors(cudaStreamQuery(s0));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((e.wait(), 0));
  checkCudaErrors(cudaStreamWaitEvent(s0, e, 0));

  // CHECK: s0->wait();
  cudaStreamSynchronize(s0);
  // CHECK: checkCudaErrors((s1->wait(), 0));
  // CHECK-EMPTY:
  checkCudaErrors(cudaStreamSynchronize(s1));

  // CHECK: delete s0;
  cudaStreamDestroy(s0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkCudaErrors((delete s1, 0));
  checkCudaErrors(cudaStreamDestroy(s1));
}
