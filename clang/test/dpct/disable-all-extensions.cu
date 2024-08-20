// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5
// RUN: dpct --format-range=none -usm-level=none -out-root %T/disable-all-extensions %s --cuda-include-path="%cuda-path/include" --no-dpcpp-extensions=all
// RUN: FileCheck --input-file %T/disable-all-extensions/disable-all-extensions.dp.cpp --match-full-lines %s

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <functional>
#include <list>
#include <stdio.h>

// bfloat16

__global__ void f() {
  // CHECK: __nv_bfloat16 bf16;
  __nv_bfloat16 bf16;
  // CHECK: __nv_bfloat162 bf162;
  __nv_bfloat162 bf162;
  float f;
  float2 f2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __hadd_sat is not supported.
  // CHECK-NEXT: */
  bf16 = __hadd_sat(bf16, bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __hfma_sat is not supported.
  // CHECK-NEXT: */
  bf16 = __hfma_sat(bf16, bf16, bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __hmul_sat is not supported.
  // CHECK-NEXT: */
  bf16 = __hmul_sat(bf16, bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __hsub_sat is not supported.
  // CHECK-NEXT: */
  bf16 = __hsub_sat(bf16, bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __habs2 is not supported.
  // CHECK-NEXT: */
  bf162 = __habs2(bf162);
  // CHECK: f2 = sycl::float2(bf162[0], bf162[1]);
  f2 = __bfloat1622float2(bf162);
  // CHECK: f = static_cast<float>(bf16);
  f = __bfloat162float(bf16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __float22bfloat162_rn is not supported.
  // CHECK-NEXT: */
  bf162 = __float22bfloat162_rn(f2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of __float2bfloat16 is not supported.
  // CHECK-NEXT: */
  bf16 = __float2bfloat16(f);
}

// device_info
void device_info() {
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1090:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with pciDeviceID. It was not migrated. You need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: const int id = properties.pciDeviceID;
  const int id = properties.pciDeviceID;
  // CHECK: /*
  // CHECK-NEXT: DPCT1090:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with uuid. It was not migrated. You need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: const std::array<unsigned char, 16> uuid = properties.uuid;
  const cudaUUID_t uuid = properties.uuid;
  // CHECK: /*
  // CHECK-NEXT: DPCT1090:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with pciDeviceID. It was not migrated. You need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: properties.pciDeviceID = id;
  properties.pciDeviceID = id;
  // CHECK: /*
  // CHECK-NEXT: DPCT1090:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with uuid. It was not migrated. You need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: properties.uuid = uuid;
  properties.uuid = uuid;
}

// enqueued_barriers

#define N 1000

#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess) {                              \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n", error,       \
              cudaGetErrorString(error));                    \
    }                                                        \
  }

__global__ void add(int *a, int *b) {
  int i = blockIdx.x;
  if (i < N) {
    b[i] = 2 * a[i];
  }
}

int enqueued_barriers() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
  cudaStream_t stream;

  int ha[N], hb[N];
  cudaEvent_t start, stop;

  int *da, *db;
  float elapsedTime;

  cudaMalloc((void **)&da, N * sizeof(int));
  cudaMalloc((void **)&db, N * sizeof(int));

  for (int i = 0; i < N; ++i) {
    ha[i] = i;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // CHECK:   dpct::get_current_device().queues_wait_and_throw();
  // CHECK-NEXT:    *start = q_ct1.single_task([=](){});
  // CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
  cudaEventRecord(start, 0);

  cudaMemcpyAsync(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(da, ha, N * sizeof(int), cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(da, ha, N * sizeof(int), cudaMemcpyHostToDevice, stream);

  // CHECK:    dpct::get_current_device().queues_wait_and_throw();
  // CHECK-NEXT:    *stop = q_ct1.single_task([=](){});
  // CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
  // CHECK-NEXT:     stop->wait_and_throw();
  // CHECK-NEXT:    elapsedTime = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  add<<<N, 1>>>(da, db);

  // CHECK: dpct::async_dpct_memcpy(hb, db, N * sizeof(int), dpct::device_to_host);
  cudaMemcpyAsync(hb, db, N * sizeof(int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  for (int i = 0; i < N; ++i) {
    printf("%d\n", hb[i]);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(da);
  cudaFree(db);

  return 0;
}

// peer_access
int peer_access() {
  int r;
  // CHECK:  /*
  // CHECK:  DPCT1031:{{[0-9]+}}: Memory access across peer devices is an implementation-specific feature which may not be supported by some SYCL backends and compilers. The output parameter(s) are set to 0. You can migrate the code with peer access extension if you do not specify -no-dpcpp-extensions=peer_access.
  // CHECK:  */
  // CHECK:  r = 0;
  cudaDeviceCanAccessPeer(&r, 0, 0);
  // CHECK:  /*
  // CHECK:  DPCT1026:{{[0-9]+}}: The call to cudaDeviceEnablePeerAccess was removed because SYCL currently does not support memory access across peer devices. You can migrate the code with peer access extension by not specifying -no-dpcpp-extensions=peer_access.
  // CHECK:  */
  cudaDeviceEnablePeerAccess(0, 0);
  // CHECK:  /*
  // CHECK:  DPCT1026:{{[0-9]+}}: The call to cudaDeviceDisablePeerAccess was removed because SYCL currently does not support memory access across peer devices. You can migrate the code with peer access extension by not specifying -no-dpcpp-extensions=peer_access.
  // CHECK:  */
  cudaDeviceDisablePeerAccess(0);
  // CHECK:  /*
  // CHECK:  DPCT1026:{{[0-9]+}}: The call to cuCtxEnablePeerAccess was removed because SYCL currently does not support memory access across peer devices. You can migrate the code with peer access extension by not specifying -no-dpcpp-extensions=peer_access.
  // CHECK:  */
  cuCtxEnablePeerAccess(0, 0);

  return 0;
}

__global__ void kernel() {
  // CHECK:  /*
  // CHECK:  DPCT1028:{{[0-9]+}}: The __trap was not migrated because assert extension is disabled. You can migrate the code with assert extension by not specifying --no-dpcpp-extensions=assert.
  // CHECK:  */
  // CHECK:  __trap();
  __trap();
}

// queue_empty

template <typename T>
// CHECK: void my_error_checker(T ReturnValue, char const *const FuncName) {
void my_error_checker(T ReturnValue, char const *const FuncName) {
}

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)

__global__ void kernelFunc() {
}

// CHECK: void process(dpct::queue_ptr st, char *data, dpct::err0 status) {}
void process(cudaStream_t st, char *data, cudaError_t status) {}

template <typename T>
// CHECK: void callback(dpct::queue_ptr st, dpct::err0 status, void *vp) {
void callback(cudaStream_t st, cudaError_t status, void *vp) {
  T *data = static_cast<T *>(vp);
  process(st, data, status);
}

template <typename FloatN, typename Float>
static void func() {
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

  // CHECK: for (; 0;)
  // CHECK-NEXT: s0 = dev_ct1.create_queue();
  for (; 0;)
    cudaStreamCreate(&s0);

  // CHECK:   q_ct1.parallel_for(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  kernelFunc<<<16, 32, 0>>>();

  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(s1 = dev_ct1.create_queue()));
  MY_ERROR_CHECKER(cudaStreamCreate(&s1));

  // CHECK:   s0->parallel_for(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernelFunc();
  // CHECK-NEXT:         });
  kernelFunc<<<16, 32, 0, s0>>>();

  // CHECK:   s1->parallel_for(
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

    // CHECK:   s2->parallel_for(
    // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
    // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernelFunc();
    // CHECK-NEXT:         });
    kernelFunc<<<16, 32, 0, s2>>>();

    // CHECK:   (*s3)->parallel_for(
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

      // CHECK:   s4->parallel_for(
      // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
      // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
      // CHECK-NEXT:           kernelFunc();
      // CHECK-NEXT:         });
      kernelFunc<<<16, 32, 0, s4>>>();
      // CHECK:   s5->parallel_for(
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
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaDeviceGetStreamPriorityRange was removed because SYCL currently does not support get queue priority range.
  // CHECK-NEXT: */
  cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi);
  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaDeviceGetStreamPriorityRange was replaced with 0 because SYCL currently does not support get queue priority range.
  // CHECK-NEXT: */
  // CHECK: MY_ERROR_CHECKER(0);
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
  // CHECK-NEXT: std::async([&]() { s1->wait(); callback<char *>(s1, 0, str); });
  cudaError_t status = cudaStreamAddCallback(s0, callback<char *>, str, flags);
  cudaStreamAddCallback(s1, callback<char *>, str, flags);

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

  // CHECK: dpct::event_ptr e;
  // CHECK-NEXT: e->wait();
  cudaEvent_t e;
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

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuStreamQuery was removed because SYCL currently does not support query operations on queues.
  // CHECK-NEXT: */
  cuStreamQuery(s0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cuStreamQuery was replaced with 0 because SYCL currently does not support query operations on queues.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_ERROR_CHECKER(0);
  MY_ERROR_CHECKER(cuStreamQuery(s0));

  // CHECK: MY_ERROR_CHECKER(DPCT_CHECK_ERROR(e->wait()));
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
