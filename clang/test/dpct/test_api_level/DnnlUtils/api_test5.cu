// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test5_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test5_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test5_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test5_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test5_out

// CHECK: 12

#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
// TEST_FEATURE: DnnlUtils_reorder

int main() {

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    int n = 1, c = 2, h = 5, w = 5;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    float alpha = 3.f, beta = 1.f;
    float *data, *out;
    cudnnTransformTensor(handle, &alpha, dataTensor, data, &beta, outTensor, out);

    return 0;
}
