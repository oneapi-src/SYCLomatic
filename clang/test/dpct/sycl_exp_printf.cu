// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --use-experimental-features=non-stdandard-sycl-builtins --format-range=none -out-root %T/sycl_exp_printf %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sycl_exp_printf/sycl_exp_printf.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/sycl_exp_printf/sycl_exp_printf.dp.cpp -o %T/sycl_exp_printf/sycl_exp_printf.dp.o %}

#include <cub/cub.cuh>

__global__ void ExampleKernel() {
  int thread_data[4] = {1, 2, 3, 4};
  // CHECK: sycl::ext::oneapi::experimental::printf("%3d: [%d, %d, %d, %d]\n", item_ct1.get_local_id(2), thread_data[0], thread_data[1], thread_data[2], thread_data[3]);
  // CHECK-NEXT: sycl::ext::oneapi::experimental::printf("%3d: Hello %s\n", item_ct1.get_local_id(2), "World");
  printf("%3d: [%d, %d, %d, %d]\n", threadIdx.x, thread_data[0], thread_data[1], thread_data[2], thread_data[3]);
  printf("%3d: Hello %s\n", threadIdx.x, "World");
}

int main() {
  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:     ExampleKernel(item_ct1);
  // CHECK-NEXT:   });
  ExampleKernel<<<1, 128>>>();
  cudaStreamSynchronize(0);
  return 0;
}
