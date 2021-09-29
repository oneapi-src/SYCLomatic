// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none  --use-custom-helper=api -out-root %T/DplExtrasDpcppExtensions/api_test4_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: grep "IsCalled" %T/DplExtrasDpcppExtensions/api_test4_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasDpcppExtensions/api_test4_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasDpcppExtensions/api_test4_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasDpcppExtensions/api_test4_out

// CHECK: 4
// TEST_FEATURE: DplExtrasDpcppExtensions_inclusive_scan


#include <cuda_runtime.h>
#include <cub/cub.cuh>

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