// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/warplevel/shuffle %S/shuffle.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/warplevel/shuffle/shuffle.dp.cpp --match-full-lines %s

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32

void init_data(int* data, int num) {
  for(int i = 0; i < num; i++)
    data[i] = 1;
}
void verify_data(int* data, int num) {
  return;
}
void print_data(int* data, int num) {
  for (int i = 0; i < num; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

// CHECK: void ShuffleIndexKernel1(int* data,
// CHECK-NEXT: const sycl::nd_item<3> &item_ct1) {
// CHECK-EMPTY:
// CHECK-NEXT:  int threadid = item_ct1.get_local_id(2);
// CHECK-EMPTY:
// CHECK-NEXT:  int input = data[threadid];
// CHECK-NEXT:  int output = 0;
// CHECK-NEXT:  output = item_ct1.get_sub_group().shuffle(input, 0);
// CHECK-NEXT:  data[threadid] = output;
// CHECK-NEXT:}
__global__ void ShuffleIndexKernel1(int* data) {

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  output = cub::ShuffleIndex<32>(input, 0, 0xffffffff);
  data[threadid] = output;
}

__global__ void ShuffleIndexKernel2(int* data) {

  int threadid = threadIdx.x;
  int input = data[threadid];
  int output = 0;
// CHECK: /*
// CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cub::ShuffleUp is not supported.
// CHECK-NEXT: */
// CHECK-NEXT: output = cub::ShuffleUp<32>(input, 0, 0, 0xaaaaaaaa);
  output = cub::ShuffleUp<32>(input, 0, 0, 0xaaaaaaaa);
  data[threadid] = output;
}

int main() {
  int* dev_data = nullptr;

  dim3 GridSize(2);
  dim3 BlockSize(1 , 1, 128);
  int TotalThread = GridSize.x * BlockSize.x * BlockSize.y * BlockSize.z;

  cudaMallocManaged(&dev_data, TotalThread * sizeof(int));

  init_data(dev_data, TotalThread);
//CHECK:  q_ct1.parallel_for(
//CHECK-NEXT:            sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:            [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:              ShuffleIndexKernel1(dev_data, item_ct1);
//CHECK-NEXT:            });
  ShuffleIndexKernel1<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK:    q_ct1.parallel_for(
//CHECK-NEXT:            sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:            [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:              ShuffleIndexKernel2(dev_data, item_ct1);
//CHECK-NEXT:            });
  ShuffleIndexKernel2<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);
  return 0;
}
