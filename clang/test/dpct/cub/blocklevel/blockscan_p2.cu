// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/blocklevel/blockscan_p2 %S/blockscan_p2.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockscan_p2/blockscan_p2.dp.cpp --match-full-lines %s

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

//CHECK:  void ExclusiveSumKernel1(int* data, int* aggregate, sycl::nd_item<3> item_ct1) {

//CHECK:    int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                        + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                        + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                        + item_ct1.get_local_id(2);;
//CHECK:    int input = data[threadid];
//CHECK:    int output = 0;
//CHECK:    int agg = 0;

//CHECK:    output = dpct::group::exclusive_scan(item_ct1, input, 0, sycl::plus<>(), agg);

//CHECK:    data[threadid] = output;
//CHECK:    aggregate[threadid] = agg;
//CHECK:  }
__global__ void ExclusiveSumKernel1(int* data, int* aggregate) {
  typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;
  __shared__ typename BlockScan::TempStorage temp1;

  int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
  int input = data[threadid];
  int output = 0;
  int agg = 0;

  BlockScan(temp1).ExclusiveSum(input, output, agg);

  data[threadid] = output;
  aggregate[threadid] = agg;
}

//CHECK:  struct CallbackOp1
//CHECK:  {
//CHECK:      int value;
//CHECK:      CallbackOp1(int init_value) : value(init_value) {}
//CHECK:      int operator()(int aggregate)
//CHECK:      {
//CHECK:          int pre_value = value;
//CHECK:          value += aggregate;
//CHECK:          return pre_value;
//CHECK:      }
//CHECK:  };

//CHECK:  void ExclusiveSumKernel2(int* data, sycl::nd_item<3> item_ct1) {

//CHECK:      CallbackOp1 CB(0);
//CHECK:      int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                        + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                        + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                        + item_ct1.get_local_id(2);;
//CHECK:      int input = data[threadid];
//CHECK:      int output = 0;

//CHECK:      output = dpct::group::exclusive_scan(item_ct1, input, sycl::plus<>(), CB);

//CHECK:      data[threadid] = output;
//CHECK:  }

struct CallbackOp1
{
    int value;

    __device__ CallbackOp1(int init_value) : value(init_value) {}

    __device__ int operator()(int aggregate)
    {
        int pre_value = value;
        value += aggregate;
        return pre_value;
    }
};

__global__ void ExclusiveSumKernel2(int* data) {
    typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;
    __shared__ typename BlockScan::TempStorage temp1;
    CallbackOp1 CB(0);
    int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
    int input = data[threadid];
    int output = 0;

    BlockScan(temp1).ExclusiveSum(input, output, CB);

    data[threadid] = output;
}
//CHECK:  void ExclusiveSumKernel3(int* data, sycl::nd_item<3> item_ct1) {

//CHECK:    int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                      + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                      + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                      + item_ct1.get_local_id(2);;
//CHECK:    int input[4];
//CHECK:    input[0] = data[4 * threadid];
//CHECK:    input[1] = data[4 * threadid + 1];
//CHECK:    input[2] = data[4 * threadid + 2];
//CHECK:    input[3] = data[4 * threadid + 3];
//CHECK:    int output[4];

//CHECK:    dpct::group::exclusive_scan(item_ct1, input, output, 0, sycl::plus<>());

//CHECK:    data[4 * threadid] = output[0];
//CHECK:    data[4 * threadid + 1] = output[1];
//CHECK:    data[4 * threadid + 2] = output[2];
//CHECK:    data[4 * threadid + 3] = output[3];
//CHECK:  }
__global__ void ExclusiveSumKernel3(int* data) {
    typedef cub::BlockScan<int, 8, cub::BLOCK_SCAN_RAKING, 4, 1> BlockScan;
    __shared__ typename BlockScan::TempStorage temp1;
    int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
    int input[4];
    input[0] = data[4 * threadid];
    input[1] = data[4 * threadid + 1];
    input[2] = data[4 * threadid + 2];
    input[3] = data[4 * threadid + 3];
    int output[4];

    BlockScan(temp1).ExclusiveSum(input, output);

    data[4 * threadid] = output[0];
    data[4 * threadid + 1] = output[1];
    data[4 * threadid + 2] = output[2];
    data[4 * threadid + 3] = output[3];
}
//CHECK:  void ExclusiveScanKernel1(int* data, int* aggregate, sycl::nd_item<3> item_ct1) {

//CHECK:    int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                      + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                      + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                      + item_ct1.get_local_id(2);;
//CHECK:  int input = data[threadid];
//CHECK:  int output = 0;
//CHECK:  int agg = 0;

//CHECK:  output = dpct::group::exclusive_scan(item_ct1, input, 0, sycl::plus<>(), agg);

//CHECK:  data[threadid] = output;
//CHECK:  aggregate[threadid] = agg;
//CHECK:  }
__global__ void ExclusiveScanKernel1(int* data, int* aggregate) {
  typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;
  __shared__ typename BlockScan::TempStorage temp1;

    int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
  int input = data[threadid];
  int output = 0;
  int agg = 0;

  BlockScan(temp1).ExclusiveScan(input, output, 0, cub::Sum(), agg);

  data[threadid] = output;
  aggregate[threadid] = agg;
}

//CHECK:  struct CallbackOp2
//CHECK:  {
//CHECK:      int value;
//CHECK:      CallbackOp2(int init_value) : value(init_value) {}
//CHECK:      int operator()(int aggregate)
//CHECK:      {
//CHECK:          int pre_value = value;
//CHECK:          value += aggregate;
//CHECK:          return pre_value;
//CHECK:      }
//CHECK:  };

//CHECK:  void ExclusiveScanKernel2(int* data, sycl::nd_item<3> item_ct1) {

//CHECK:      CallbackOp2 CB(0);
//CHECK:      int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                        + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                        + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                        + item_ct1.get_local_id(2);;
//CHECK:      int input = data[threadid];
//CHECK:      int output = 0;

//CHECK:      output = dpct::group::exclusive_scan(item_ct1, input, sycl::plus<>(), CB);

//CHECK:      data[threadid] = output;
//CHECK:  }

struct CallbackOp2
{
    int value;

    __device__ CallbackOp2(int init_value) : value(init_value) {}

    __device__ int operator()(int aggregate)
    {
        int pre_value = value;
        value += aggregate;
        return pre_value;
    }
};

__global__ void ExclusiveScanKernel2(int* data) {
    typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;
    __shared__ typename BlockScan::TempStorage temp1;
    CallbackOp2 CB(0);
    int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
    int input = data[threadid];
    int output = 0;

    BlockScan(temp1).ExclusiveScan(input, output, cub::Sum(), CB);

    data[threadid] = output;
}
//CHECK:  void ExclusiveScanKernel3(int* data, sycl::nd_item<3> item_ct1) {
//CHECK:    int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                      + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                      + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                      + item_ct1.get_local_id(2);;
//CHECK:    int input[4];
//CHECK:    input[0] = data[4 * threadid];
//CHECK:    input[1] = data[4 * threadid + 1];
//CHECK:    input[2] = data[4 * threadid + 2];
//CHECK:    input[3] = data[4 * threadid + 3];
//CHECK:    int output[4];

//CHECK:    dpct::group::exclusive_scan(item_ct1, input, output, 0, sycl::plus<>());

//CHECK:    data[4 * threadid] = output[0];
//CHECK:    data[4 * threadid + 1] = output[1];
//CHECK:    data[4 * threadid + 2] = output[2];
//CHECK:    data[4 * threadid + 3] = output[3];
//CHECK:  }
__global__ void ExclusiveScanKernel3(int* data) {
    typedef cub::BlockScan<int, 8, cub::BLOCK_SCAN_RAKING, 4, 1> BlockScan;
    __shared__ typename BlockScan::TempStorage temp1;
    int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
    int input[4];
    input[0] = data[4 * threadid];
    input[1] = data[4 * threadid + 1];
    input[2] = data[4 * threadid + 2];
    input[3] = data[4 * threadid + 3];
    int output[4];

    BlockScan(temp1).ExclusiveScan(input, output, 0, cub::Sum());

    data[4 * threadid] = output[0];
    data[4 * threadid + 1] = output[1];
    data[4 * threadid + 2] = output[2];
    data[4 * threadid + 3] = output[3];
}

//CHECK:  void InclusiveSumKernel1(int* data, int* aggregate, sycl::nd_item<3> item_ct1) {

//CHECK:    int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                      + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                      + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                      + item_ct1.get_local_id(2);;
//CHECK:  int input = data[threadid];
//CHECK:  int output = 0;
//CHECK:  int agg = 0;

//CHECK:  output = dpct::group::inclusive_scan(item_ct1, input, sycl::plus<>(), agg);

//CHECK:  data[threadid] = output;
//CHECK:  aggregate[threadid] = agg;
//CHECK:  }

__global__ void InclusiveSumKernel1(int* data, int* aggregate) {
  typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;
  __shared__ typename BlockScan::TempStorage temp1;

    int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
  int input = data[threadid];
  int output = 0;
  int agg = 0;

  BlockScan(temp1).InclusiveSum(input, output, agg);

  data[threadid] = output;
  aggregate[threadid] = agg;
}

//CHECK:  struct CallbackOp3
//CHECK:  {
//CHECK:      int value;

//CHECK:      CallbackOp3(int init_value) : value(init_value) {}

//CHECK:      int operator()(int aggregate)
//CHECK:      {
//CHECK:          int pre_value = value;
//CHECK:          value += aggregate;
//CHECK:          return pre_value;
//CHECK:      }
//CHECK:  };

//CHECK:  void InclusiveSumKernel2(int* data, sycl::nd_item<3> item_ct1) {
//CHECK:      CallbackOp3 CB(0);
//CHECK:      int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                        + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                        + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                        + item_ct1.get_local_id(2);;
//CHECK:      int input = data[threadid];
//CHECK:      int output = 0;

//CHECK:      output = dpct::group::inclusive_scan(item_ct1, input, sycl::plus<>(), CB);

//CHECK:      data[threadid] = output;
//CHECK:  }

struct CallbackOp3
{
    int value;

    __device__ CallbackOp3(int init_value) : value(init_value) {}

    __device__ int operator()(int aggregate)
    {
        int pre_value = value;
        value += aggregate;
        return pre_value;
    }
};

__global__ void InclusiveSumKernel2(int* data) {
    typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;
    __shared__ typename BlockScan::TempStorage temp1;
    CallbackOp3 CB(0);
    int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
    int input = data[threadid];
    int output = 0;

    BlockScan(temp1).InclusiveSum(input, output, CB);

    data[threadid] = output;
}

//CHECK:  void InclusiveSumKernel3(int* data, sycl::nd_item<3> item_ct1) {

//CHECK:    int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                      + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                      + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                      + item_ct1.get_local_id(2);;
//CHECK:    int input[4];
//CHECK:    input[0] = data[4 * threadid];
//CHECK:    input[1] = data[4 * threadid + 1];
//CHECK:    input[2] = data[4 * threadid + 2];
//CHECK:    input[3] = data[4 * threadid + 3];
//CHECK:    int output[4];

//CHECK:    dpct::group::inclusive_scan(item_ct1, input, output, sycl::plus<>());

//CHECK:    data[4 * threadid] = output[0];
//CHECK:    data[4 * threadid + 1] = output[1];
//CHECK:    data[4 * threadid + 2] = output[2];
//CHECK:    data[4 * threadid + 3] = output[3];
//CHECK:  }

__global__ void InclusiveSumKernel3(int* data) {
    typedef cub::BlockScan<int, 8, cub::BLOCK_SCAN_RAKING, 4, 1> BlockScan;
    __shared__ typename BlockScan::TempStorage temp1;
    int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
    int input[4];
    input[0] = data[4 * threadid];
    input[1] = data[4 * threadid + 1];
    input[2] = data[4 * threadid + 2];
    input[3] = data[4 * threadid + 3];
    int output[4];

    BlockScan(temp1).InclusiveSum(input, output);

    data[4 * threadid] = output[0];
    data[4 * threadid + 1] = output[1];
    data[4 * threadid + 2] = output[2];
    data[4 * threadid + 3] = output[3];
}

//CHECK:  void InclusiveScanKernel1(int* data, int* aggregate, sycl::nd_item<3> item_ct1) {

//CHECK:    int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                      + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                      + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                      + item_ct1.get_local_id(2);;
//CHECK:  int input = data[threadid];
//CHECK:  int output = 0;
//CHECK:  int agg = 0;

//CHECK:  output = dpct::group::inclusive_scan(item_ct1, input, sycl::plus<>(), agg);

//CHECK:  data[threadid] = output;
//CHECK:  aggregate[threadid] = agg;
//CHECK:  }

__global__ void InclusiveScanKernel1(int* data, int* aggregate) {
  typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;
  __shared__ typename BlockScan::TempStorage temp1;

    int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
  int input = data[threadid];
  int output = 0;
  int agg = 0;

  BlockScan(temp1).InclusiveScan(input, output, cub::Sum(), agg);

  data[threadid] = output;
  aggregate[threadid] = agg;
}

//CHECK:  struct CallbackOp4
//CHECK:  {
//CHECK:      int value;

//CHECK:      CallbackOp4(int init_value) : value(init_value) {}

//CHECK:      int operator()(int aggregate)
//CHECK:      {
//CHECK:          int pre_value = value;
//CHECK:          value += aggregate;
//CHECK:          return pre_value;
//CHECK:      }
//CHECK:  };

//CHECK:  void InclusiveScanKernel2(int* data, sycl::nd_item<3> item_ct1) {
//CHECK:      CallbackOp4 CB(0);
//CHECK:      int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                        + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                        + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                        + item_ct1.get_local_id(2);;
//CHECK:      int input = data[threadid];
//CHECK:      int output = 0;

//CHECK:      output = dpct::group::inclusive_scan(item_ct1, input, sycl::plus<>(), CB);

//CHECK:      data[threadid] = output;
//CHECK:  }

struct CallbackOp4
{
    int value;

    __device__ CallbackOp4(int init_value) : value(init_value) {}

    __device__ int operator()(int aggregate)
    {
        int pre_value = value;
        value += aggregate;
        return pre_value;
    }
};

__global__ void InclusiveScanKernel2(int* data) {
    typedef cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 8, 1> BlockScan;
    __shared__ typename BlockScan::TempStorage temp1;
    CallbackOp4 CB(0);
    int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
    int input = data[threadid];
    int output = 0;

    BlockScan(temp1).InclusiveScan(input, output, cub::Sum(), CB);

    data[threadid] = output;
}

//CHECK:  void InclusiveScanKernel3(int* data, sycl::nd_item<3> item_ct1) {

//CHECK:    int threadid = item_ct1.get_group(2) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0))
//CHECK:                      + item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))
//CHECK:                      + item_ct1.get_local_id(1) * item_ct1.get_local_range(2)
//CHECK:                      + item_ct1.get_local_id(2);;
//CHECK:    int input[4];
//CHECK:    input[0] = data[4 * threadid];
//CHECK:    input[1] = data[4 * threadid + 1];
//CHECK:    input[2] = data[4 * threadid + 2];
//CHECK:    input[3] = data[4 * threadid + 3];
//CHECK:    int output[4];

//CHECK:    dpct::group::inclusive_scan(item_ct1, input, output, sycl::plus<>());

//CHECK:    data[4 * threadid] = output[0];
//CHECK:    data[4 * threadid + 1] = output[1];
//CHECK:    data[4 * threadid + 2] = output[2];
//CHECK:    data[4 * threadid + 3] = output[3];
//CHECK:  }

__global__ void InclusiveScanKernel3(int* data) {
    typedef cub::BlockScan<int, 8, cub::BLOCK_SCAN_RAKING, 4, 1> BlockScan;
    __shared__ typename BlockScan::TempStorage temp1;
    int threadid = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                      + threadIdx.z * (blockDim.x * blockDim.y)
                      + threadIdx.y * blockDim.x
                      + threadIdx.x;;
    int input[4];
    input[0] = data[4 * threadid];
    input[1] = data[4 * threadid + 1];
    input[2] = data[4 * threadid + 2];
    input[3] = data[4 * threadid + 3];
    int output[4];

    BlockScan(temp1).InclusiveScan(input, output, cub::Sum());

    data[4 * threadid] = output[0];
    data[4 * threadid + 1] = output[1];
    data[4 * threadid + 2] = output[2];
    data[4 * threadid + 3] = output[3];
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
  BlockSize = {16, 8, 1};
  int expect3[DATA_NUM] = {
    0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465,
    496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953,
    2016, 2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465,
    4560, 4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001,
    0, 128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433,
    4592, 4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017,
    10208, 10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625,
    16848, 17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257
  };
  int agg_expect3[DATA_NUM] = {
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512
  };
  init_data(dev_data, DATA_NUM);
  init_data(dev_agg, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      ExclusiveSumKernel1(dev_data, dev_agg, item_ct1);
  //CHECK:    });
  ExclusiveSumKernel1<<<GridSize, BlockSize>>>(dev_data, dev_agg);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect3, DATA_NUM)) {
    std::cout << "ExclusiveSumKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect3, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }
  if(!verify_data(dev_agg, agg_expect3, DATA_NUM)) {
    std::cout << "ExclusiveSumKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(agg_expect3, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_agg, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect4[DATA_NUM] = {
    0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465,
    496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953,
    2016, 2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465,
    4560, 4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001,
    0, 128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433,
    4592, 4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017,
    10208, 10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625,
    16848, 17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257
  };
  init_data(dev_data, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      ExclusiveSumKernel2(dev_data, item_ct1);
  //CHECK:    });
  ExclusiveSumKernel2<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect4, DATA_NUM)) {
    std::cout << "ExclusiveSumKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect4, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {8, 4, 1};
  int expect5[DATA_NUM] = {
    0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465,
    496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953,
    2016, 2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465,
    4560, 4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001,
    0, 128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433,
    4592, 4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017,
    10208, 10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625,
    16848, 17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257
  };
  init_data(dev_data, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      ExclusiveSumKernel3(dev_data, item_ct1);
  //CHECK:    });
  ExclusiveSumKernel3<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect5, DATA_NUM)) {
    std::cout << "ExclusiveSumKernel3" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect5, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }


  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect6[DATA_NUM] = {
    0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465,
    496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953,
    2016, 2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465,
    4560, 4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001,
    0, 128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433,
    4592, 4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017,
    10208, 10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625,
    16848, 17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257
  };
  int agg_expect6[DATA_NUM] = {
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512
  };
  init_data(dev_data, DATA_NUM);
  init_data(dev_agg, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      ExclusiveScanKernel1(dev_data, dev_agg, item_ct1);
  //CHECK:    });
  ExclusiveScanKernel1<<<GridSize, BlockSize>>>(dev_data, dev_agg);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect6, DATA_NUM)) {
    std::cout << "ExclusiveScanKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect6, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }
  if(!verify_data(dev_agg, agg_expect6, DATA_NUM)) {
    std::cout << "ExclusiveScanKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(agg_expect6, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_agg, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect7[DATA_NUM] = {
    0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465,
    496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953,
    2016, 2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465,
    4560, 4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001,
    0, 128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433,
    4592, 4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017,
    10208, 10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625,
    16848, 17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257
  };
  init_data(dev_data, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      ExclusiveScanKernel2(dev_data, item_ct1);
  //CHECK:    });
  ExclusiveScanKernel2<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect7, DATA_NUM)) {
    std::cout << "ExclusiveScanKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect7, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {8, 4, 1};
  int expect8[DATA_NUM] = {
    0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465,
    496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953,
    2016, 2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465,
    4560, 4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001,
    0, 128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433,
    4592, 4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017,
    10208, 10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625,
    16848, 17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257
  };
  init_data(dev_data, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      ExclusiveScanKernel3(dev_data, item_ct1);
  //CHECK:    });
  ExclusiveScanKernel3<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect8, DATA_NUM)) {
    std::cout << "ExclusiveScanKernel3" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect8, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }


  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect9[DATA_NUM] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496,
    528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016,
    2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465, 4560,
    4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001, 8128,
    128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433, 4592,
    4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017, 10208,
    10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625, 16848,
    17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257, 24512
  };
  int agg_expect9[DATA_NUM] = {
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512
  };
  init_data(dev_data, DATA_NUM);
  init_data(dev_agg, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      InclusiveSumKernel1(dev_data, dev_agg, item_ct1);
  //CHECK:    });
  InclusiveSumKernel1<<<GridSize, BlockSize>>>(dev_data, dev_agg);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect9, DATA_NUM)) {
    std::cout << "InclusiveSumKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect9, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }
  if(!verify_data(dev_agg, agg_expect9, DATA_NUM)) {
    std::cout << "InclusiveSumKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(agg_expect9, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_agg, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect10[DATA_NUM] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496,
    528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016,
    2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465, 4560,
    4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001, 8128,
    128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433, 4592,
    4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017, 10208,
    10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625, 16848,
    17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257, 24512
  };
  init_data(dev_data, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      InclusiveSumKernel2(dev_data, item_ct1);
  //CHECK:    });
  InclusiveSumKernel2<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect10, DATA_NUM)) {
    std::cout << "InclusiveSumKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect10, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {8, 4, 1};
  int expect11[DATA_NUM] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496,
    528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016,
    2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465, 4560,
    4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001, 8128,
    128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433, 4592,
    4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017, 10208,
    10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625, 16848,
    17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257, 24512
  };
  init_data(dev_data, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      InclusiveSumKernel3(dev_data, item_ct1);
  //CHECK:    });
  InclusiveSumKernel3<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect11, DATA_NUM)) {
    std::cout << "InclusiveSumKernel3" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect11, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect12[DATA_NUM] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496,
    528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016,
    2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465, 4560,
    4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001, 8128,
    128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433, 4592,
    4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017, 10208,
    10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625, 16848,
    17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257, 24512
  };
  int agg_expect12[DATA_NUM] = {
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128, 8128,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512,
    24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512, 24512
  };
  init_data(dev_data, DATA_NUM);
  init_data(dev_agg, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      InclusiveScanKernel1(dev_data, dev_agg, item_ct1);
  //CHECK:    });
  InclusiveScanKernel1<<<GridSize, BlockSize>>>(dev_data, dev_agg);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect12, DATA_NUM)) {
    std::cout << "InclusiveScanKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect12, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }
  if(!verify_data(dev_agg, agg_expect12, DATA_NUM)) {
    std::cout << "InclusiveScanKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(agg_expect12, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_agg, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect13[DATA_NUM] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496,
    528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016,
    2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465, 4560,
    4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001, 8128,
    128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433, 4592,
    4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017, 10208,
    10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625, 16848,
    17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257, 24512
  };
  init_data(dev_data, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      InclusiveScanKernel2(dev_data, item_ct1);
  //CHECK:    });
  InclusiveScanKernel2<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect13, DATA_NUM)) {
    std::cout << "InclusiveScanKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect13, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {8, 4, 1};
  int expect14[DATA_NUM] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496,
    528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016,
    2080, 2145, 2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926, 3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828, 3916, 4005, 4095, 4186, 4278, 4371, 4465, 4560,
    4656, 4753, 4851, 4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995, 6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260, 7381, 7503, 7626, 7750, 7875, 8001, 8128,
    128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433, 4592,
    4752, 4913, 5075, 5238, 5402, 5567, 5733, 5900, 6068, 6237, 6407, 6578, 6750, 6923, 7097, 7272, 7448, 7625, 7803, 7982, 8162, 8343, 8525, 8708, 8892, 9077, 9263, 9450, 9638, 9827, 10017, 10208,
    10400, 10593, 10787, 10982, 11178, 11375, 11573, 11772, 11972, 12173, 12375, 12578, 12782, 12987, 13193, 13400, 13608, 13817, 14027, 14238, 14450, 14663, 14877, 15092, 15308, 15525, 15743, 15962, 16182, 16403, 16625, 16848,
    17072, 17297, 17523, 17750, 17978, 18207, 18437, 18668, 18900, 19133, 19367, 19602, 19838, 20075, 20313, 20552, 20792, 21033, 21275, 21518, 21762, 22007, 22253, 22500, 22748, 22997, 23247, 23498, 23750, 24003, 24257, 24512
  };
  init_data(dev_data, DATA_NUM);
  //CHECK:  q_ct1.parallel_for(
  //CHECK:    sycl::nd_range<3>(GridSize * BlockSize, BlockSize), 
  //CHECK:    [=](sycl::nd_item<3> item_ct1) {
  //CHECK:      InclusiveScanKernel3(dev_data, item_ct1);
  //CHECK:    });
  InclusiveScanKernel3<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect14, DATA_NUM)) {
    std::cout << "InclusiveScanKernel3" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect14, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  if(Result)
    std::cout << "passed" << std::endl;
  return 0;
}
