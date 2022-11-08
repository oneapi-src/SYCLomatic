// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/blocklevel/blockscan %S/blockscan.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockscan/blockscan.dp.cpp --match-full-lines %s

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
//CHECK: void ExclusiveScanKernel(int* data,
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT:  int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT:  int input = data[threadid];
//CHECK-NEXT:  int output = 0;
//CHECK-NEXT:  output = sycl::exclusive_scan_over_group(item_ct1.get_group(), input, 0, sycl::plus<>());
//CHECK-NEXT:  data[threadid] = output;
//CHECK-NEXT:}
__global__ void ExclusiveScanKernel(int* data) {
  typedef cub::BlockScan<int, 4> BlockScan;

  __shared__ typename BlockScan::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  BlockScan(temp1).ExclusiveScan(input, output, 0, cub::Sum());
  data[threadid] = output;
}

//CHECK: void ExclusiveSumKernel(int* data,
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT:  int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT:  int input = data[threadid];
//CHECK-NEXT:  int output = 0;
//CHECK-NEXT:  output = sycl::exclusive_scan_over_group(item_ct1.get_group(), input, 0, sycl::plus<>());
//CHECK-NEXT:  data[threadid] = output;
//CHECK-NEXT:}
__global__ void ExclusiveSumKernel(int* data) {
  typedef cub::BlockScan<int, 4> BlockScan;

  __shared__ typename BlockScan::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  BlockScan(temp1).ExclusiveSum(input, output);
  data[threadid] = output;
}

//CHECK: void InclusiveScanKernel(int* data,
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT:  int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT:  int input = data[threadid];
//CHECK-NEXT:  int output = 0;
//CHECK-NEXT:  output = sycl::inclusive_scan_over_group(item_ct1.get_group(), input, sycl::plus<>());
//CHECK-NEXT:  data[threadid] = output;
//CHECK-NEXT:}
__global__ void InclusiveScanKernel(int* data) {
  typedef cub::BlockScan<int, 4> BlockScan;

  __shared__ typename BlockScan::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  BlockScan(temp1).InclusiveScan(input, output, cub::Sum());
  data[threadid] = output;
}

//CHECK: void InclusiveSumKernel(int* data,
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT:  int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT:  int input = data[threadid];
//CHECK-NEXT:  int output = 0;
//CHECK-NEXT:  output = sycl::inclusive_scan_over_group(item_ct1.get_group(), input, sycl::plus<>());
//CHECK-NEXT:  data[threadid] = output;
//CHECK-NEXT:}
__global__ void InclusiveSumKernel(int* data) {
  typedef cub::BlockScan<int, 4> BlockScan;

  __shared__ typename BlockScan::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  BlockScan(temp1).InclusiveSum(input, output);
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
//CHECK-NEXT:        sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:          ExclusiveScanKernel(dev_data, item_ct1);
//CHECK-NEXT:        });
  ExclusiveScanKernel<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK:  q_ct1.parallel_for(
//CHECK-NEXT:        sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:          ExclusiveSumKernel(dev_data, item_ct1);
//CHECK-NEXT:        });
  ExclusiveSumKernel<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK:  q_ct1.parallel_for(
//CHECK-NEXT:        sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:          InclusiveScanKernel(dev_data, item_ct1);
//CHECK-NEXT:        });
  InclusiveScanKernel<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK:  q_ct1.parallel_for(
//CHECK-NEXT:        sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:          InclusiveSumKernel(dev_data, item_ct1);
//CHECK-NEXT:        });
  InclusiveSumKernel<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  return 0;
}
