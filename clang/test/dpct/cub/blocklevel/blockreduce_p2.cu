// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/blocklevel/blockreduce_p2 %S/blockreduce_p2.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockreduce_p2/blockreduce_p2.dp.cpp --match-full-lines %s

#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32
#define DATA_NUM 256


template<typename T = int>
void init_data(T* data, int num) {
  for(int i = 0; i < num; i++)
    data[i] = i;
}

template<typename T = int>
bool verify_data(T* data, T* expect, int num, int step = 1) {
  for(int i = 0; i < num; i = i + step) {
    if(data[i] != expect[i]) {
      return false;
    }
  }
  return true;
}

template<typename T = int>
void print_data(T* data, int num) {
  for (int i = 0; i < num; i++) {
    std::cout << data[i] << ", ";
    if((i+1)%32 == 0)
        std::cout << std::endl;
  }
  std::cout << std::endl;
}
//CHECK:  void SumKernel(int* data, sycl::nd_item<3> item_ct1) {

//CHECK:    int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                        + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                        + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                        + item_ct1.get_local_id(2);;

//CHECK:    int input[4];
//CHECK:    input[0] = data[4 * threadid];
//CHECK:    input[1] = data[4 * threadid + 1];
//CHECK:    input[2] = data[4 * threadid + 2];
//CHECK:    input[3] = data[4 * threadid + 3];
//CHECK:    int output = 0;
//CHECK:    output = dpct::group::reduce(item_ct1, input, sycl::plus<>());
//CHECK:    data[4 * threadid] = output;
//CHECK:    data[4 * threadid + 1] = 0;
//CHECK:    data[4 * threadid + 2] = 0;
//CHECK:    data[4 * threadid + 3] = 0;
//CHECK:  }
__global__ void SumKernel(int* data) {
  typedef cub::BlockReduce<int, 8, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 4, 1> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp1;

  int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;

  int input[4];
  input[0] = data[4 * threadid];
  input[1] = data[4 * threadid + 1];
  input[2] = data[4 * threadid + 2];
  input[3] = data[4 * threadid + 3];
  int output = 0;
  output = BlockReduce(temp1).Sum(input);
  data[4 * threadid] = output;
  data[4 * threadid + 1] = 0;
  data[4 * threadid + 2] = 0;
  data[4 * threadid + 3] = 0;
  
}

//CHECK:  void ReduceKernel(int* data, sycl::nd_item<3> item_ct1) {

//CHECK:    int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                        + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                        + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                        + item_ct1.get_local_id(2);;

//CHECK:    int input[4];
//CHECK:    input[0] = data[4 * threadid];
//CHECK:    input[1] = data[4 * threadid + 1];
//CHECK:    input[2] = data[4 * threadid + 2];
//CHECK:    input[3] = data[4 * threadid + 3];
//CHECK:    int output = 0;
//CHECK:    output = dpct::group::reduce(item_ct1, input, sycl::plus<>());
//CHECK:    data[4 * threadid] = output;
//CHECK:    data[4 * threadid + 1] = 0;
//CHECK:    data[4 * threadid + 2] = 0;
//CHECK:    data[4 * threadid + 3] = 0;
//CHECK:  }
__global__ void ReduceKernel(int* data) {
  typedef cub::BlockReduce<int, 8, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 4, 1> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp1;

  int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;

  int input[4];
  input[0] = data[4 * threadid];
  input[1] = data[4 * threadid + 1];
  input[2] = data[4 * threadid + 2];
  input[3] = data[4 * threadid + 3];
  int output = 0;
  output = BlockReduce(temp1).Reduce(input, cub::Sum());
  data[4 * threadid] = output;
  data[4 * threadid + 1] = 0;
  data[4 * threadid + 2] = 0;
  data[4 * threadid + 3] = 0;
}

int main() {
  bool Result = true;
  int* dev_data = nullptr;
  int* dev_agg = nullptr;

  dim3 GridSize;
  dim3 BlockSize;
  cudaMallocManaged(&dev_data, DATA_NUM * sizeof(int));
  cudaMallocManaged(&dev_agg, DATA_NUM * sizeof(int));

  GridSize = {2};
  BlockSize = {8, 4, 1};
  int expect1[DATA_NUM] = {
    8128, 0, 0, 0, 8122, 0, 0, 0, 8100, 0, 0, 0, 8062, 0, 0, 0, 8008, 0, 0, 0, 7938, 0, 0, 0, 7852, 0, 0, 0, 7750, 0, 0, 0,
    7632, 0, 0, 0, 7498, 0, 0, 0, 7348, 0, 0, 0, 7182, 0, 0, 0, 7000, 0, 0, 0, 6802, 0, 0, 0, 6588, 0, 0, 0, 6358, 0, 0, 0,
    6112, 0, 0, 0, 5850, 0, 0, 0, 5572, 0, 0, 0, 5278, 0, 0, 0, 4968, 0, 0, 0, 4642, 0, 0, 0, 4300, 0, 0, 0, 3942, 0, 0, 0,
    3568, 0, 0, 0, 3178, 0, 0, 0, 2772, 0, 0, 0, 2350, 0, 0, 0, 1912, 0, 0, 0, 1458, 0, 0, 0, 988, 0, 0, 0, 502, 0, 0, 0,
    24512, 0, 0, 0, 23994, 0, 0, 0, 23460, 0, 0, 0, 22910, 0, 0, 0, 22344, 0, 0, 0, 21762, 0, 0, 0, 21164, 0, 0, 0, 20550, 0, 0, 0,
    19920, 0, 0, 0, 19274, 0, 0, 0, 18612, 0, 0, 0, 17934, 0, 0, 0, 17240, 0, 0, 0, 16530, 0, 0, 0, 15804, 0, 0, 0, 15062, 0, 0, 0,
    14304, 0, 0, 0, 13530, 0, 0, 0, 12740, 0, 0, 0, 11934, 0, 0, 0, 11112, 0, 0, 0, 10274, 0, 0, 0, 9420, 0, 0, 0, 8550, 0, 0, 0,
    7664, 0, 0, 0, 6762, 0, 0, 0, 5844, 0, 0, 0, 4910, 0, 0, 0, 3960, 0, 0, 0, 2994, 0, 0, 0, 2012, 0, 0, 0, 1014, 0, 0, 0
  };
  init_data(dev_data, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      SumKernel(dev_data, item_ct1);
  //CHECK:    });
  SumKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect1, DATA_NUM, 128)) {
    std::cout << "SumKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect1, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {8, 4, 1};
  int expect2[DATA_NUM] = {
    8128, 0, 0, 0, 8122, 0, 0, 0, 8100, 0, 0, 0, 8062, 0, 0, 0, 8008, 0, 0, 0, 7938, 0, 0, 0, 7852, 0, 0, 0, 7750, 0, 0, 0,
    7632, 0, 0, 0, 7498, 0, 0, 0, 7348, 0, 0, 0, 7182, 0, 0, 0, 7000, 0, 0, 0, 6802, 0, 0, 0, 6588, 0, 0, 0, 6358, 0, 0, 0,
    6112, 0, 0, 0, 5850, 0, 0, 0, 5572, 0, 0, 0, 5278, 0, 0, 0, 4968, 0, 0, 0, 4642, 0, 0, 0, 4300, 0, 0, 0, 3942, 0, 0, 0,
    3568, 0, 0, 0, 3178, 0, 0, 0, 2772, 0, 0, 0, 2350, 0, 0, 0, 1912, 0, 0, 0, 1458, 0, 0, 0, 988, 0, 0, 0, 502, 0, 0, 0,
    24512, 0, 0, 0, 23994, 0, 0, 0, 23460, 0, 0, 0, 22910, 0, 0, 0, 22344, 0, 0, 0, 21762, 0, 0, 0, 21164, 0, 0, 0, 20550, 0, 0, 0,
    19920, 0, 0, 0, 19274, 0, 0, 0, 18612, 0, 0, 0, 17934, 0, 0, 0, 17240, 0, 0, 0, 16530, 0, 0, 0, 15804, 0, 0, 0, 15062, 0, 0, 0,
    14304, 0, 0, 0, 13530, 0, 0, 0, 12740, 0, 0, 0, 11934, 0, 0, 0, 11112, 0, 0, 0, 10274, 0, 0, 0, 9420, 0, 0, 0, 8550, 0, 0, 0,
    7664, 0, 0, 0, 6762, 0, 0, 0, 5844, 0, 0, 0, 4910, 0, 0, 0, 3960, 0, 0, 0, 2994, 0, 0, 0, 2012, 0, 0, 0, 1014, 0, 0, 0
  };
  init_data(dev_data, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      ReduceKernel(dev_data, item_ct1);
  //CHECK:    });
  ReduceKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect2, DATA_NUM, 128)) {
    std::cout << "ReduceKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect2, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  if(Result)
    std::cout << "passed" << std::endl;
  return 0;
}
