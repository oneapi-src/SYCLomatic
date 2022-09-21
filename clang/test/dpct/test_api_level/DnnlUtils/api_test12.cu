// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test12_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test12_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test12_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test12_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test12_out

// CHECK: 13
// TEST_FEATURE: DnnlUtils_reduction
// TEST_FEATURE: DnnlUtils_reduction_op
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#define DT float
int main() {

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    cudaStream_t stream1;

    int in = 2, ic = 2, ih = 6, iw = 6;
    int on = 2, oc = 2, oh = 6, ow = 1;
    DT *data, *out;
    float alpha = 2.5f, beta = 1.5f;

    cudnnReduceTensorDescriptor_t reducedesc;
    cudnnCreateReduceTensorDescriptor(&reducedesc);

    cudnnSetReduceTensorDescriptor(
        reducedesc,
        CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS,
        CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES);

    void *ws;
    size_t ws_size;

    cudnnGetReductionWorkspaceSize(
        handle,
        reducedesc,
        dataTensor,
        outTensor,
        &ws_size);

    cudnnReduceTensor(
        handle,
        reducedesc,
        0,
        0,
        ws,
        ws_size,
        &alpha,
        dataTensor,
        data,
        &beta,
        outTensor,
        out);

    return 0;
}
