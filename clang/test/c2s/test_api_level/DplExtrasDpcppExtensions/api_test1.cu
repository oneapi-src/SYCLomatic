// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: c2s --format-range=none  --use-custom-helper=api -out-root %T/DplExtrasDpcppExtensions/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: grep "IsCalled" %T/DplExtrasDpcppExtensions/api_test1_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasDpcppExtensions/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasDpcppExtensions/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasDpcppExtensions/api_test1_out

// CHECK: 19
// TEST_FEATURE: DplExtrasDpcppExtensions_segmented_reduce


#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define DATA_NUM 100

bool test_reduce(){
    int          num_segments = 10;
    int          *device_offsets;
    int          *device_in;
    int          *device_out;
    int          initial_value = INT_MAX;
    void     *temp_storage = NULL;
    size_t   temp_storage_size = 0;
  
    cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);
  
    cudaMalloc(&temp_storage, temp_storage_size);
  
    cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);
  
    return true;
}