// RUN: dpct --format-range=none --enable-default-queue-synchronization -out-root %T/default_stream_sync %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/default_stream_sync/default_stream_sync.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/default_stream_sync/default_stream_sync.dp.cpp -o %T/default_stream_sync/default_stream_sync.dp.o %}
#include<cuda_runtime.h>

__global__ void kernel(int *a){

}

int main() {
  int *a, *b;
  cudaStream_t s1;
  cudaStreamCreate(&s1);
  cudaMallocManaged(&a, 100);
  cudaMallocManaged(&b, 100);

  cudaMemcpyAsync(a,b, 1, cudaMemcpyHostToDevice, s1);
//CHECK:  q_ct1.submit(
//CHECK:      [&](sycl::handler &cgh) {
//CHECK:        dpct::get_current_device().none_default_queues_wait();
//CHECK:        cgh.parallel_for(
//CHECK:          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK:          [=](sycl::nd_item<3> item_ct1) {
//CHECK:            kernel(a);
//CHECK:          });
//CHECK:      });
  kernel<<<1,1>>>(a);
  cudaMemcpyAsync(b, a, 1, cudaMemcpyDeviceToHost, s1);
  cudaDeviceSynchronize();

  return 0;
}
