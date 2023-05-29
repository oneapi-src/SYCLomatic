// FIXME:
// UNSUPPORTED: system-windows
// RUN: dpct --usm-level=none -out-root %T/cuda-stream-api %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda-stream-api/cuda-stream-api.dp.cpp --match-full-lines %s

#include <list>
#include <functional>

template <typename T>
// CHECK: void my_error_checker(T ReturnValue, char const *const FuncName) {
void my_error_checker(T ReturnValue, char const *const FuncName) {
}

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)

__global__ void kernelFunc() {
}

// CHECK: void process(dpct::queue_ptr st, char *data, dpct::err0 status) {}
void process(cudaStream_t st, char *data, cudaError_t status) {}

template<typename T>
// CHECK: void callback(dpct::queue_ptr st, dpct::err0 status, void *vp) {
void callback(cudaStream_t st, cudaError_t status, void *vp) {
  T *data = static_cast<T *>( vp);
  process(st, data, status);
}

template<typename FloatN, typename Float>
static void func()
{
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK: std::list<dpct::queue_ptr> streams;
  std::list<cudaStream_t> streams;
  for (auto Iter = streams.begin(); Iter != streams.end(); ++Iter)
    // CHECK: *Iter = dev_ct1.create_queue();
    cudaStreamCreate(&*Iter);
  for (auto Iter = streams.begin(); Iter != streams.end(); ++Iter)
    // CHECK: dev_ct1.destroy_queue(*Iter);
    cudaStreamDestroy(*Iter);

  // CHECK: dpct::queue_ptr s0, &s1 = s0;
  // CHECK-NEXT: dpct::queue_ptr s2, *s3 = &s2;
  // CHECK-NEXT: dpct::queue_ptr s4, s5;
  // CHECK-EMPTY:
  cudaStream_t s0, &s1 = s0;
  cudaStream_t s2, *s3 = &s2;
  cudaStream_t s4, s5;

  // CHECK: if (1)
  // CHECK-NEXT: s0 = dev_ct1.create_queue();
  if (1)
    cudaStreamCreate(&s0);

  // CHECK: while (0)
  // CHECK-NEXT: s0 = dev_ct1.create_queue();
  while (0)
    cudaStreamCreate(&s0);

  // CHECK: do
  // CHECK-NEXT: s0 = dev_ct1.create_queue();
  // CHECK: while (0);
  do
    cudaStreamCreate(&s0);
  while (0);

  // CHECK: for (; 0; )
  // CHECK-NEXT: s0 = dev_ct1.create_queue();
  for (; 0; )
    cudaStreamCreate(&s0);

  // CHECK:   q_ct1.parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  kernelFunc<<<16, 32, 0>>>();

  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(s1 = dev_ct1.create_queue()));
  MY_ERROR_CHECKER(cudaStreamCreate(&s1));

  // CHECK:   s0->parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  kernelFunc<<<16, 32, 0, s0>>>();

  // CHECK:   s1->parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  kernelFunc<<<16, 32, 0, s1>>>();

  {
    // CHECK: /*
    // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag and priority options.
    // CHECK-NEXT: */
    // CHECK-NEXT: s2 = dev_ct1.create_queue();
    cudaStreamCreateWithFlags(&s2, cudaStreamDefault);

    // CHECK: /*
    // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag and priority options.
    // CHECK-NEXT: */
    // CHECK-NEXT: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(*(s3) = dev_ct1.create_queue()));
    MY_ERROR_CHECKER(cudaStreamCreateWithFlags(s3, cudaStreamNonBlocking));

    // CHECK:   s2->parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
    // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
    // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernelFunc();
    // CHECK-NEXT:         });
    kernelFunc<<<16, 32, 0, s2>>>();

    // CHECK:   (*s3)->parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
    // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
    // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernelFunc();
    // CHECK-NEXT:         });
    kernelFunc<<<16, 32, 0, *s3>>>();

    // CHECK: dev_ct1.destroy_queue(s2);
    cudaStreamDestroy(s2);
    // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dev_ct1.destroy_queue(*s3)));
    MY_ERROR_CHECKER(cudaStreamDestroy(*s3));
  }

  {
    {
      // CHECK: /*
      // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag and priority options.
      // CHECK-NEXT: */
      // CHECK-NEXT: s4 = dev_ct1.create_queue();
      cudaStreamCreateWithPriority(&s4, cudaStreamDefault, 2);

      // CHECK: /*
      // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag and priority options.
      // CHECK-NEXT: */
      // CHECK-NEXT: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(s5 = dev_ct1.create_queue()));
      MY_ERROR_CHECKER(cudaStreamCreateWithPriority(&s5, cudaStreamNonBlocking, 3));

      // CHECK:   s4->parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
      // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
      // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
      // CHECK-NEXT:           kernelFunc();
      // CHECK-NEXT:         });
      kernelFunc<<<16, 32, 0, s4>>>();
      // CHECK:   s5->parallel_for<dpct_kernel_name<class kernelFunc_{{[a-f0-9]+}}>>(
      // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
      // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
      // CHECK-NEXT:           kernelFunc();
      // CHECK-NEXT:         });
      kernelFunc<<<16, 32, 0, s5>>>();

      // CHECK: dev_ct1.destroy_queue(s4);
      cudaStreamDestroy(s4);
      // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dev_ct1.destroy_queue(s5)));
      MY_ERROR_CHECKER(cudaStreamDestroy(s5));
    }
  }

  int priority_low;
  int priority_hi;
  // CHECK: /*
  // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The flag and priority options are not supported for SYCL queues. The output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(&priority_low) = 0, *(&priority_hi) = 0;
  cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi);
  // CHECK: /*
  // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The flag and priority options are not supported for SYCL queues. The output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(*(&priority_low) = 0, *(&priority_hi) = 0));
  MY_ERROR_CHECKER(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi));

  int priority;
  // CHECK: /*
  // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The flag and priority options are not supported for SYCL queues. The output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(*(&priority) = 0));
  MY_ERROR_CHECKER(cudaStreamGetPriority(s0, &priority));

  char str[256];

  unsigned int flags = 0;
  // CHECK: dpct::err0 status = DPCT_CHECK_ERROR(std::async([&]() { s0->wait(); callback<char *>(s0, 0, str); }));
  // CHECK-NEXT: std::async([&]() { s1->wait(); callback<char*>(s1, 0, str); });
  cudaError_t status = cudaStreamAddCallback(s0, callback<char *>, str, flags);
  cudaStreamAddCallback(s1, callback<char*>, str, flags);

  // CHECK: /*
  // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The flag and priority options are not supported for SYCL queues. The output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(*(&flags) = 0));
  MY_ERROR_CHECKER(cudaStreamGetFlags(s0, &flags));

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaStreamAttachMemAsync was removed because SYCL currently does not support associating USM with a specific queue.
  // CHECK-NEXT: */
  cudaStreamAttachMemAsync(s0, nullptr);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaStreamAttachMemAsync was replaced with 0 because SYCL currently does not support associating USM with a specific queue.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER(0);
  MY_ERROR_CHECKER(cudaStreamAttachMemAsync(s0, nullptr));

  cudaEvent_t e;
  // CHECK:  s0->ext_oneapi_submit_barrier({*e});
  cudaStreamWaitEvent(s0, e, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaStreamQuery was removed because SYCL currently does not support query operations on queues.
  // CHECK-NEXT: */
  cudaStreamQuery(s0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaStreamQuery was replaced with 0 because SYCL currently does not support query operations on queues.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER(0);
  MY_ERROR_CHECKER(cudaStreamQuery(s0));

  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(s0->ext_oneapi_submit_barrier({*e})));
  MY_ERROR_CHECKER(cudaStreamWaitEvent(s0, e, 0));

  // CHECK: s0->wait();
  cudaStreamSynchronize(s0);
  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(s1->wait()));
  // CHECK-EMPTY:
  MY_ERROR_CHECKER(cudaStreamSynchronize(s1));

  // CHECK: q_ct1.wait();
  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(q_ct1.wait()));
  // CHECK-NEXT: q_ct1.wait();
  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(q_ct1.wait()));
  // CHECK-NEXT: q_ct1.wait();
  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(q_ct1.wait()));
  cudaStreamSynchronize(cudaStreamDefault);
  MY_ERROR_CHECKER(cudaStreamSynchronize(cudaStreamDefault));
  cudaStreamSynchronize(cudaStreamLegacy);
  MY_ERROR_CHECKER(cudaStreamSynchronize(cudaStreamLegacy));
  cudaStreamSynchronize(cudaStreamPerThread);
  MY_ERROR_CHECKER(cudaStreamSynchronize(cudaStreamPerThread));

  // CHECK: dev_ct1.destroy_queue(s0);
  cudaStreamDestroy(s0);
  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(dev_ct1.destroy_queue(s1)));
  MY_ERROR_CHECKER(cudaStreamDestroy(s1));
}

template <typename T>
class S {
  static const unsigned int MAX_NUM_PATHS = 8;
  // CHECK: dpct::queue_ptr streams[MAX_NUM_PATHS];
  cudaStream_t streams[MAX_NUM_PATHS];
};

template <int I>
class S2 {
  // CHECK: dpct::queue_ptr streams[I];
  cudaStream_t streams[I];
};

void foo(int size) {
  // CHECK: dpct::queue_ptr streams[size];
  cudaStream_t streams[size];
}

// CHECK: void gFunc0(dpct::err0 e, dpct::queue_ptr s) {}
void gFunc0(cudaError_t e, cudaStream_t s) {}
// CHECK: void gFunc1(dpct::queue_ptr s, dpct::err0 e) {}
void gFunc1(cudaStream_t s ,cudaError_t e) {}

// CHECK: void gFunc2(dpct::err0 e, dpct::queue_ptr const s) {}
void gFunc2(cudaError_t e, cudaStream_t const s) {}
// CHECK: void gFunc3(dpct::queue_ptr const s, dpct::err0 e) {}
void gFunc3(cudaStream_t const s, cudaError_t e) {}

// CHECK: void gFunc4(dpct::err0 e, const dpct::queue_ptr s) {}
void gFunc4(cudaError_t e, const cudaStream_t s) {}
// CHECK: void gFunc5(const dpct::queue_ptr s, dpct::err0 e) {}
void gFunc5(const cudaStream_t s ,cudaError_t e) {}

void bar() {
   // CHECK: std::function<void(dpct::err0, dpct::queue_ptr)> f0 = gFunc0;
   std::function<void(cudaError_t, cudaStream_t)> f0 = gFunc0;
   // CHECK: std::function<void(dpct::queue_ptr, dpct::err0)> f1 = gFunc1;
   std::function<void(cudaStream_t, cudaError_t)> f1 = gFunc1;

   // CHECK: std::function<void(dpct::err0, dpct::queue_ptr const)> f2 = gFunc2;
   std::function<void(cudaError_t, cudaStream_t const)> f2 = gFunc2;
   // CHECK: std::function<void(dpct::queue_ptr const, dpct::err0)> f3 = gFunc3;
   std::function<void(cudaStream_t const, cudaError_t)> f3 = gFunc3;

   // CHECK: std::function<void(dpct::err0, const dpct::queue_ptr)> f4 = gFunc4;
   std::function<void(cudaError_t, const cudaStream_t)> f4 = gFunc4;
   // CHECK: std::function<void(const dpct::queue_ptr, dpct::err0)> f5 = gFunc5;
   std::function<void(const cudaStream_t, cudaError_t)> f5 = gFunc5;

   // CHECK: sizeof(dpct::queue_ptr);
   sizeof(cudaStream_t);
}

// CHECK: #define INIT_STREAM (dpct::queue_ptr *)malloc(10 * sizeof(dpct::queue_ptr))
#define INIT_STREAM (cudaStream_t *) malloc(10 * sizeof(cudaStream_t))

int stream_initializers() {
  // CHECK: dpct::queue_ptr *streams = (dpct::queue_ptr *)malloc(10 * sizeof(dpct::queue_ptr));
  cudaStream_t *streams = (cudaStream_t *) malloc(10 * sizeof(cudaStream_t));
  // CHECK: (dpct::queue_ptr *)malloc(10 * sizeof(dpct::queue_ptr));
  (cudaStream_t *) malloc(10 * sizeof(cudaStream_t));
  // CHECK: sizeof(dpct::queue_ptr);
  sizeof(cudaStream_t);

  // CHECK: dpct::queue_ptr *streams2 = INIT_STREAM;
  cudaStream_t *streams2 = INIT_STREAM;
}

class C {
  // CHECK: dpct::queue_ptr *streams = (dpct::queue_ptr *)malloc(10 * sizeof(dpct::queue_ptr));
  cudaStream_t *streams = (cudaStream_t *) malloc(10 * sizeof(cudaStream_t));
  cudaStream_t *streams2 = INIT_STREAM;
  void foo() {
    // CHECK: streams = (dpct::queue_ptr *)malloc(10 * sizeof(dpct::queue_ptr));
    streams = (cudaStream_t *) malloc(10 * sizeof(cudaStream_t));
    streams2 = INIT_STREAM;
  }
};

