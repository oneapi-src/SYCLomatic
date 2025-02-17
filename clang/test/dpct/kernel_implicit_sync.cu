// RUN: dpct --format-range=none --use-experimental-features=in_order_queue_events -out-root %T/kernel_implicit_sync %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel_implicit_sync/kernel_implicit_sync.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/kernel_implicit_sync/kernel_implicit_sync.dp.cpp -o %T/kernel_implicit_sync/kernel_implicit_sync.dp.o %}
#include<cuda_runtime.h>
#include<iostream>
__global__ void kernel(int *a){

}

void hostCallback(void *userData) {
  const char *msg = static_cast<const char*>(userData);
  std::cout << "Host callback executed. Message: " << msg << std::endl;
}

int main() {
  int *a, *b;
  cudaStream_t s1;
  cudaStreamCreate(&s1);
  cudaMallocManaged(&a, 100);
  cudaMallocManaged(&b, 100);

// CHECK:  q_ct1.submit(
// CHECK:      [&](sycl::handler &cgh) {
// CHECK:        cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());
// CHECK:        cgh.parallel_for(
// CHECK:          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK:          [=](sycl::nd_item<3> item_ct1) {
// CHECK:            kernel(a);
// CHECK:          });
// CHECK:      });
  kernel<<<1,1>>>(a);

// CHECK:  s1->submit(
// CHECK:      [&](sycl::handler &cgh) {
// CHECK:        cgh.depends_on(dpct::get_default_queue().ext_oneapi_get_last_event());
// CHECK:        cgh.parallel_for(
// CHECK:          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK:          [=](sycl::nd_item<3> item_ct1) {
// CHECK:            kernel(b);
// CHECK:          });
// CHECK:      });  
  kernel<<<1, 1, 0, s1>>>(b);

  cudaError_t err;
  const char *message = "Kernel execution finished.";
  cudaStream_t stream;
// CHECK:  err = DPCT_CHECK_ERROR(stream->submit([&](sycl::handler &cgh) {
// CHECK:    cgh.depends_on(stream->ext_oneapi_get_last_event());
// CHECK:    cgh.host_task([=](){
// CHECK:      hostCallback((void*)message);
// CHECK:    });
// CHECK:  }));

  err = cudaLaunchHostFunc(stream, hostCallback, (void*)message);
// CHECK: stream->submit([&](sycl::handler &cgh) {
// CHECK:   cgh.depends_on(stream->ext_oneapi_get_last_event());
// CHECK:   cgh.host_task([=](){
// CHECK:     hostCallback((void*)message);
// CHECK: });
  cudaLaunchHostFunc(stream, hostCallback, (void*)message);

  return 0;
}
