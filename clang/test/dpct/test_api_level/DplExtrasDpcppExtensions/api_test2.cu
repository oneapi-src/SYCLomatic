// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: c2s --format-range=none  --use-custom-helper=api -out-root %T/DplExtrasDpcppExtensions/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: grep "IsCalled" %T/DplExtrasDpcppExtensions/api_test2_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasDpcppExtensions/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasDpcppExtensions/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasDpcppExtensions/api_test2_out

// CHECK: 3
// TEST_FEATURE: DplExtrasDpcppExtensions_reduce


#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define DATA_NUM 100

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