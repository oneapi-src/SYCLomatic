// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/warplevel/warpreduce %S/warpreduce.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/warplevel/warpreduce/warpreduce.dp.cpp --match-full-lines %s
// clang-format off
// CHECK: #include <oneapi/dpl/execution>
// CHECK: #include <oneapi/dpl/algorithm>
// CHECK: #include <dpct/dpl_utils.hpp>
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

//CHECK: void SumKernel(int* data, const sycl::nd_item<3> &item_ct1) {
//CHECK:  int threadid = item_ct1.get_local_id(2);
//CHECK:  int input = data[threadid];
//CHECK:  int output = 0;
//CHECK:  output = sycl::reduce_over_group(item_ct1.get_sub_group(), input, sycl::plus<>());
//CHECK:  data[threadid] = output;
//CHECK:}
__global__ void SumKernel(int* data) {
  typedef cub::WarpReduce<int> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  output = WarpReduce(temp1).Sum(input);
  data[threadid] = output;
}

//CHECK: void ReduceKernel(int* data, const sycl::nd_item<3> &item_ct1) {
//CHECK:  int threadid = item_ct1.get_local_id(2);
//CHECK:  int input = data[threadid];
//CHECK:  int output = 0;
//CHECK:  output = sycl::reduce_over_group(item_ct1.get_sub_group(), input, sycl::plus<>());
//CHECK:  data[threadid] = output;
//CHECK:}
__global__ void ReduceKernel(int* data) {
  typedef cub::WarpReduce<int> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  output = WarpReduce(temp1).Reduce(input, cub::Sum());
  data[threadid] = output;
}


//CHECK: void ReduceKernel_Max(int* data, const sycl::nd_item<3> &item_ct1) {
//CHECK:  int threadid = item_ct1.get_local_id(2);
//CHECK:  int input = data[threadid];
//CHECK:  int output = 0;
//CHECK:  output = sycl::reduce_over_group(item_ct1.get_sub_group(), input, sycl::maximum<>());
//CHECK:  data[threadid] = output;
//CHECK:}
__global__ void ReduceKernel_Max(int* data) {
  typedef cub::WarpReduce<int> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  output = WarpReduce(temp1).Reduce(input, cub::Max());
  data[threadid] = output;
}


//CHECK: void ReduceKernel_Min(int* data, const sycl::nd_item<3> &item_ct1) {
//CHECK:  int threadid = item_ct1.get_local_id(2);
//CHECK:  int input = data[threadid];
//CHECK:  int output = 0;
//CHECK:  output = sycl::reduce_over_group(item_ct1.get_sub_group(), input, sycl::minimum<>());
//CHECK:  data[threadid] = output;
//CHECK:}
__global__ void ReduceKernel_Min(int* data) {
  typedef cub::WarpReduce<int> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  output = WarpReduce(temp1).Reduce(input, cub::Min());
  data[threadid] = output;
}


//CHECK: void ReduceKernel2(int* data, int valid_items, const sycl::nd_item<3> &item_ct1) {
//CHECK:  int threadid = item_ct1.get_local_id(2);
//CHECK:  int input = data[threadid];
//CHECK:  int output = 0;
//CHECK:  output = dpct::group::reduce_over_partial_group(item_ct1, input, valid_items, sycl::plus<>());
//CHECK:  data[threadid] = output;
//CHECK:}
__global__ void ReduceKernel2(int* data, int valid_items) {
  typedef cub::WarpReduce<int> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  output = WarpReduce(temp1).Reduce(input, cub::Sum(), valid_items);
  data[threadid] = output;
}

// CHECK: void SumKernel2(int* data, int valid_items, const sycl::nd_item<3> &item_ct1) {
// CHECK:  int threadid = item_ct1.get_local_id(2);
// CHECK:  int input = data[threadid];
// CHECK:  int output = 0;
// CHECK:  output = dpct::group::reduce_over_partial_group(item_ct1, input, valid_items, sycl::plus<>());
// CHECK:  data[threadid] = output;
// CHECK: }
__global__ void SumKernel2(int* data, int valid_items) {
  typedef cub::WarpReduce<int> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  output = WarpReduce(temp1).Sum(input, valid_items);
  data[threadid] = output;
}

int main() {
  int* dev_data = nullptr;

  dim3 GridSize(2);
  dim3 BlockSize(1 , 1, 128);
  int TotalThread = GridSize.x * BlockSize.x * BlockSize.y * BlockSize.z;

  cudaMallocManaged(&dev_data, TotalThread * sizeof(int));

  init_data(dev_data, TotalThread);
//CHECK: q_ct1.parallel_for(
//CHECK-NEXT:       sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:         SumKernel(dev_data, item_ct1);
//CHECK-NEXT:       });
  SumKernel<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK: q_ct1.parallel_for(
//CHECK-NEXT:       sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:         ReduceKernel(dev_data, item_ct1);
//CHECK-NEXT:       });
  ReduceKernel<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK: q_ct1.parallel_for(
//CHECK-NEXT:       sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:         ReduceKernel_Max(dev_data, item_ct1);
//CHECK-NEXT:       });
  ReduceKernel_Max<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);  
  
  init_data(dev_data, TotalThread);
//CHECK: q_ct1.parallel_for(
//CHECK-NEXT:       sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:         ReduceKernel_Min(dev_data, item_ct1);
//CHECK-NEXT:       });
  ReduceKernel_Min<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);  

  init_data(dev_data, TotalThread);
//CHECK: q_ct1.parallel_for(
//CHECK-NEXT:       sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:         ReduceKernel2(dev_data, 4, item_ct1);
//CHECK-NEXT:       });
  ReduceKernel2<<<GridSize, BlockSize>>>(dev_data, 4);
  cudaDeviceSynchronize();
  verify_data(dev_data, 4);

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:         SumKernel2(dev_data, 4, item_ct1);
  // CHECK-NEXT:       });
  SumKernel2<<<GridSize, BlockSize>>>(dev_data, 4);
  cudaDeviceSynchronize();
  verify_data(dev_data, 4);

  return 0;
}
// clang-format off
