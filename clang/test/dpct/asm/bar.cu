// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/bar %s --cuda-include-path="%cuda-path/include" --use-experimental-features=non-uniform-groups -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --input-file %T/bar/bar.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/bar/bar.dp.cpp -o %T/bar/bar.dp.o %}

// clang-format off

#include <stdio.h>

__device__ void bar(int *arr, int *brr) {
  arr[threadIdx.x] = threadIdx.x + 10;
  if (threadIdx.x % 2 == 0) {
    for (int i = 0; i < 1000; ++i)
      arr[threadIdx.x] += arr[threadIdx.x] - 1 * arr[threadIdx.x] - 3;
    if (arr[threadIdx.x] < 0)
      arr[threadIdx.x] = 0;
  }
  
  // CHECK: sycl::group_barrier(sycl::ext::oneapi::experimental::get_ballot_group(item_ct1.get_sub_group(), 0b1010101010 & (1 << item_ct1.get_local_linear_id())));
  asm volatile ("bar.warp.sync %0;" :: "r"(0b1010101010));
  if (threadIdx.x == 1) {
    for (int i = 0; i < 10; ++i) {
      brr[i] = arr[i];
    }
  }
}

__global__ void kernel(int *arr, int *brr) {
  bar(arr, brr);
}
int main() {
  int *arr, *brr;
  cudaMallocManaged(&arr, sizeof(int) * 10);
  cudaMemset(arr, 0, sizeof(int) * 10);
  cudaMallocManaged(&brr, sizeof(int) * 10);
  cudaMemset(brr, 0, sizeof(int) * 10);

  // CHECK: intel::reqd_sub_group_size(32)
  kernel<<<1, 10>>>(arr, brr);
  cudaDeviceSynchronize();
  cudaFree(arr);
  for (int i = 0; i < 10; ++i)
    printf("%d%c", brr[i], (i == 9 ? '\n' : ' '));
  return 0;
}
// clang-format on
