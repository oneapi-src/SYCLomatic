// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test24_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test24_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test24_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test24_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test24_out

// CHECK: 31
// TEST_FEATURE: DnnlUtils_convolution_backward_bias
// TEST_FEATURE: DnnlUtils_convolution_desc

#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    int nDevices;
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, biasTensor;
    cudnnFilterDescriptor_t filterTensor;
    cudaStream_t stream1;

    int in = 1, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 1, fh = 2, fw = 2;
    int ele_num = in * ic * ih * iw;
    int oele_num = on * oc * oh * ow;
    int fele_num = fk *fc * fh * fw;
    std::vector<int> bias_dim = {1, oc, 1, 1};
    std::vector<int> bias_stride = {oc, 1, 1, 1};
    int bele_num = oc * 1;

    int filterdim[4] = {fk, fc, fh, fw};

    cudnnSetFilterNdDescriptor(filterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, 4, filterdim);

    float *data, *out, *filter, *z, *bias;

    cudnnConvolutionDescriptor_t covdes;
    cudnnCreateConvolutionDescriptor(&covdes);
    cudnnSetConvolution2dDescriptor(covdes, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(covdes, 2);
    cudnnConvolutionFwdAlgoPerf_t algo;
    int retCount = 1;

    size_t size;
    void *workspacesize;
    cudnnGetConvolutionForwardWorkspaceSize(handle, dataTensor, filterTensor, covdes, outTensor, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, &size);
    cudaMalloc(&workspacesize, size);

    cudnnActivationDescriptor_t ActivationDesc;
    cudnnCreateActivationDescriptor(&ActivationDesc);
    cudnnSetActivationDescriptor(ActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0f);

    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionBackwardBias(
        handle,
        &alpha,
        outTensor,
        out,
        &beta,
        biasTensor,
        bias
    );

    return 0;
}