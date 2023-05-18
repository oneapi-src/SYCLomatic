// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test11_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test11_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test11_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test11_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test11_out

// CHECK: 15
// TEST_FEATURE: DnnlUtils_binary
// TEST_FEATURE: DnnlUtils_binary_op
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

#define DT float
int main() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;


    cudaStream_t stream1;

    int in = 1, ic = 1, ih = 5, iw = 5;
    int on = 1, oc = 1, oh = 5, ow = 5;

    DT *data, *out, *filter;

    cudnnOpTensorDescriptor_t OpDesc;
    cudnnCreateOpTensorDescriptor(&OpDesc);

    cudnnSetOpTensorDescriptor(OpDesc, CUDNN_OP_TENSOR_NOT, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);

    float alpha0 = 1.f, alpha1 = 1.f, beta = 0.f;

    auto status = cudnnOpTensor(
        handle, 
        OpDesc,
        &alpha0, 
        dataTensor, 
        data,
        &alpha1, 
        dataTensor, 
        data,
        &beta,
        outTensor,
        out
    );

    return 0;
}
