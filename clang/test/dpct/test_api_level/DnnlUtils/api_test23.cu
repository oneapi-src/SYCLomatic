// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test23_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test23_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test23_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test23_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test23_out

// CHECK: 30
// TEST_FEATURE: DnnlUtils_convolution_backward_weight
// TEST_FEATURE: DnnlUtils_convolution_desc

#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    int nDevices;
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    cudnnTensorDescriptor_t diffdataTensor, diffoutTensor;
    cudnnFilterDescriptor_t filterTensor, difffilterTensor;
    cudaStream_t stream1;

    int in = 1, ic = 4, ih = 5, iw = 5;
    int on = 1, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 2, fh = 2, fw = 2;
    int ele_num = in * ic * ih * iw;
    int oele_num = on * oc * oh * ow;
    int fele_num = fk *fc * fh * fw;

    int filterdim[4] = {fk, fc, fh, fw};
    cudnnSetFilterNdDescriptor(filterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterdim);
    cudnnSetFilterNdDescriptor(difffilterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterdim);

    float *data, *out, *filter, *diffdata, *diffout, *difffilter;

    cudnnConvolutionDescriptor_t covdes;
    cudnnCreateConvolutionDescriptor(&covdes);
    cudnnSetConvolution2dDescriptor(covdes, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(covdes, 2);

    int retCount;

    size_t size;
    void *workspacesize;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, 
        dataTensor,
        diffoutTensor, 
        covdes, 
        difffilterTensor, 
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, 
        &size);
    cudaMalloc(&workspacesize, size);

    float alpha = 1.0f, beta = 0.f;
    cudnnConvolutionBackwardFilter(
        handle, 
        &alpha,
        dataTensor,
        data,
        diffoutTensor,
        diffout, 
        covdes, 
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        workspacesize, 
        size, 
        &beta, 
        difffilterTensor, 
        difffilter);

    return 0;
}