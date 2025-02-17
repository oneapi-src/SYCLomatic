// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -format-range=none -in-root %S -out-root %T/blocklevel/blockshuffle %S/blockshuffle.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockshuffle/blockshuffle.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/blocklevel/blockshuffle/blockshuffle.dp.cpp -o %T/blocklevel/blockshuffle/blockshuffle.dp.o %}

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <cub/block/block_shuffle.cuh>

// CHECK: void BlockShuffleKernel(const sycl::nd_item<3> &item_ct1,
// CHECK:       uint8_t *temp_storage) {
// CHECK:   int d[4];
// CHECK:   typedef dpct::group::group_shuffle<int, 128, 1> BS;
// CHECK:   int a;
// CHECK:   BS(temp_storage).select(item_ct1, a, a, 2);
// CHECK:   BS(temp_storage).select2(item_ct1, a, a, 2);
// CHECK:   BS(temp_storage).shuffle_right(item_ct1, d, d, a);
// CHECK:   BS(temp_storage).shuffle_left(item_ct1, d, d);
// CHECK: }

__global__ void BlockShuffleKernel() {
  int d[4];
  typedef cub::BlockShuffle<int, 128, 1> BS;
  __shared__ typename BS::TempStorage temp_storage;
  int a;
  BS(temp_storage).Offset(a, a, 2);
  BS(temp_storage).Rotate(a, a, 2);
  BS(temp_storage).Up<4>(d, d, a);
  BS(temp_storage).Down<4>(d, d);
}

bool test_striped_to_blocked() {
// CHECK: dpct::get_in_order_queue().submit(
// CHECK:       [&](sycl::handler &cgh) {
// CHECK:         sycl::local_accessor<uint8_t, 1> temp_storage_acc(dpct::group::group_shuffle<int, 128>::get_local_memory_size(sycl::range<3>(1, 1, 128).size()), cgh);
// CHECK:         cgh.parallel_for(
// CHECK:           sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
// CHECK:           [=](sycl::nd_item<3> item_ct1) {
// CHECK:             BlockShuffleKernel(item_ct1, &temp_storage_acc[0]);
// CHECK:           });
// CHECK:       });
  BlockShuffleKernel<<<1, 128>>>();
  cudaDeviceSynchronize();

  return true;
}

int main() {
  test_striped_to_blocked();
  return 0;
};
