// FIXME:
// UNSUPPORTED: -windows-
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

// CHECK: void process(sycl::queue *st, char *data, int status) {}
void process(cudaStream_t st, char *data, cudaError_t status) {}

template<typename T>
// CHECK: void callback(sycl::queue *st, int status, void *vp) {
void callback(cudaStream_t st, cudaError_t status, void *vp) {
  T *data = static_cast<T *>( vp);
  process(st, data, status);
}

template<typename FloatN, typename Float>
static void func()
{
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK: std::list<sycl::queue *> streams;
  std::list<cudaStream_t> streams;
  for (auto Iter = streams.begin(); Iter != streams.end(); ++Iter)
    // CHECK: *Iter = dev_ct1.create_queue();
    cudaStreamCreate(&*Iter);
  for (auto Iter = streams.begin(); Iter != streams.end(); ++Iter)
    // CHECK: dev_ct1.destroy_queue(*Iter);
    cudaStreamDestroy(*Iter);

  // CHECK: sycl::queue *s0, *&s1 = s0;
  // CHECK-NEXT: sycl::queue *s2, **s3 = &s2;
  // CHECK-NEXT: sycl::queue *s4, *s5;
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

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER((s1 = dev_ct1.create_queue(), 0));
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
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag and priority options.
    // CHECK-NEXT: */
    // CHECK-NEXT: MY_ERROR_CHECKER((*(s3) = dev_ct1.create_queue(), 0));
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
    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: MY_ERROR_CHECKER((dev_ct1.destroy_queue(*s3), 0));
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
      // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      // CHECK-NEXT: */
      // CHECK-NEXT: /*
      // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag and priority options.
      // CHECK-NEXT: */
      // CHECK-NEXT: MY_ERROR_CHECKER((s5 = dev_ct1.create_queue(), 0));
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
      // CHECK: /*
      // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      // CHECK-NEXT: */
      // CHECK-NEXT: MY_ERROR_CHECKER((dev_ct1.destroy_queue(s5), 0));
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
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER((*(&priority_low) = 0, *(&priority_hi) = 0, 0));
  MY_ERROR_CHECKER(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi));

  int priority;
  // CHECK: /*
  // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The flag and priority options are not supported for SYCL queues. The output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER((*(&priority) = 0, 0));
  MY_ERROR_CHECKER(cudaStreamGetPriority(s0, &priority));

  char str[256];

  unsigned int flags = 0;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: int status = (std::async([&]() { s0->wait(); callback<char *>(s0, 0, str); }), 0);
  // CHECK-NEXT: std::async([&]() { s1->wait(); callback<char*>(s1, 0, str); });
  cudaError_t status = cudaStreamAddCallback(s0, callback<char *>, str, flags);
  cudaStreamAddCallback(s1, callback<char*>, str, flags);

  // CHECK: /*
  // CHECK-NEXT: DPCT1014:{{[0-9]+}}: The flag and priority options are not supported for SYCL queues. The output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER((*(&flags) = 0, 0));
  MY_ERROR_CHECKER(cudaStreamGetFlags(s0, &flags));

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaStreamAttachMemAsync was removed because DPC++ currently does not support associating USM with a specific queue.
  // CHECK-NEXT: */
  cudaStreamAttachMemAsync(s0, nullptr);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaStreamAttachMemAsync was replaced with 0 because DPC++ currently does not support associating USM with a specific queue.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER(0);
  MY_ERROR_CHECKER(cudaStreamAttachMemAsync(s0, nullptr));

  cudaEvent_t e;
  // CHECK:  e = s0->ext_oneapi_submit_barrier({e});
  cudaStreamWaitEvent(s0, e, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaStreamQuery was removed because DPC++ currently does not support query operations on queues.
  // CHECK-NEXT: */
  cudaStreamQuery(s0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaStreamQuery was replaced with 0 because DPC++ currently does not support query operations on queues.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER(0);
  MY_ERROR_CHECKER(cudaStreamQuery(s0));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER((e = s0->ext_oneapi_submit_barrier({e}), 0));
  MY_ERROR_CHECKER(cudaStreamWaitEvent(s0, e, 0));

  // CHECK: s0->wait();
  cudaStreamSynchronize(s0);
  // CHECK: MY_ERROR_CHECKER((s1->wait(), 0));
  // CHECK-EMPTY:
  MY_ERROR_CHECKER(cudaStreamSynchronize(s1));

  // CHECK: q_ct1.wait();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER((q_ct1.wait(), 0));
  // CHECK-NEXT: q_ct1.wait();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER((q_ct1.wait(), 0));
  // CHECK-NEXT: q_ct1.wait();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER((q_ct1.wait(), 0));
  cudaStreamSynchronize(cudaStreamDefault);
  MY_ERROR_CHECKER(cudaStreamSynchronize(cudaStreamDefault));
  cudaStreamSynchronize(cudaStreamLegacy);
  MY_ERROR_CHECKER(cudaStreamSynchronize(cudaStreamLegacy));
  cudaStreamSynchronize(cudaStreamPerThread);
  MY_ERROR_CHECKER(cudaStreamSynchronize(cudaStreamPerThread));

  // CHECK: dev_ct1.destroy_queue(s0);
  cudaStreamDestroy(s0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER((dev_ct1.destroy_queue(s1), 0));
  MY_ERROR_CHECKER(cudaStreamDestroy(s1));
}

template <typename T>
class S {
  static const unsigned int MAX_NUM_PATHS = 8;
  // CHECK: sycl::queue *streams[MAX_NUM_PATHS];
  cudaStream_t streams[MAX_NUM_PATHS];
};

template <int I>
class S2 {
  // CHECK: sycl::queue *streams[I];
  cudaStream_t streams[I];
};

void foo(int size) {
  // CHECK: sycl::queue *streams[size];
  cudaStream_t streams[size];
}

// CHECK: void gFunc0(int e, sycl::queue *s) {}
void gFunc0(cudaError_t e, cudaStream_t s) {}
// CHECK: void gFunc1(sycl::queue *s, int e) {}
void gFunc1(cudaStream_t s ,cudaError_t e) {}

// CHECK: void gFunc2(int e, sycl::queue *const s) {}
void gFunc2(cudaError_t e, cudaStream_t const s) {}
// CHECK: void gFunc3(sycl::queue *const s, int e) {}
void gFunc3(cudaStream_t const s ,cudaError_t e) {}

// CHECK: void gFunc4(int e, sycl::queue *const s) {}
void gFunc4(cudaError_t e, const cudaStream_t s) {}
// CHECK: void gFunc5(sycl::queue *const s, int e) {}
void gFunc5(const cudaStream_t s ,cudaError_t e) {}

void bar() {
   // CHECK: std::function<void(int, sycl::queue *)> f0 = gFunc0;
   std::function<void(cudaError_t, cudaStream_t)> f0 = gFunc0;
   // CHECK: std::function<void(sycl::queue *, int)> f1 = gFunc1;
   std::function<void(cudaStream_t, cudaError_t)> f1 = gFunc1;

   // CHECK: std::function<void(int, sycl::queue *const)> f2 = gFunc2;
   std::function<void(cudaError_t, cudaStream_t const)> f2 = gFunc2;
   // CHECK: std::function<void(sycl::queue *const, int)> f3 = gFunc3;
   std::function<void(cudaStream_t const, cudaError_t)> f3 = gFunc3;

   // CHECK: std::function<void(int, sycl::queue *const)> f4 = gFunc4;
   std::function<void(cudaError_t, const cudaStream_t)> f4 = gFunc4;
   // CHECK: std::function<void(sycl::queue *const, int)> f5 = gFunc5;
   std::function<void(const cudaStream_t, cudaError_t)> f5 = gFunc5;

   // CHECK: sizeof(sycl::queue *);
   sizeof(cudaStream_t);
}

// CHECK: #define INIT_STREAM (sycl::queue **)malloc(10 * sizeof(sycl::queue *))
#define INIT_STREAM (cudaStream_t *) malloc(10 * sizeof(cudaStream_t))

int stream_initializers() {
  // CHECK: sycl::queue **streams = (sycl::queue **)malloc(10 * sizeof(sycl::queue *));
  cudaStream_t *streams = (cudaStream_t *) malloc(10 * sizeof(cudaStream_t));
  // CHECK: (sycl::queue **)malloc(10 * sizeof(sycl::queue *));
  (cudaStream_t *) malloc(10 * sizeof(cudaStream_t));
  // CHECK: sizeof(sycl::queue *);
  sizeof(cudaStream_t);

  // CHECK: sycl::queue **streams2 = INIT_STREAM;
  cudaStream_t *streams2 = INIT_STREAM;
}

class C {
  // CHECK: sycl::queue **streams = (sycl::queue **)malloc(10 * sizeof(sycl::queue *));
  cudaStream_t *streams = (cudaStream_t *) malloc(10 * sizeof(cudaStream_t));
  cudaStream_t *streams2 = INIT_STREAM;
  void foo() {
    // CHECK: streams = (sycl::queue **)malloc(10 * sizeof(sycl::queue *));
    streams = (cudaStream_t *) malloc(10 * sizeof(cudaStream_t));
    streams2 = INIT_STREAM;
  }
};

