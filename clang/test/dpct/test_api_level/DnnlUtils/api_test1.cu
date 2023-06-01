// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test1_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test1_out

// CHECK: 29

#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
// TEST_FEATURE: DnnlUtils_engine_ext

int main() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

  return 0;
}
