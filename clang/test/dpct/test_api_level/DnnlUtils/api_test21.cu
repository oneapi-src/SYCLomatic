// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test21_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test21_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test21_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test21_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test21_out

// CHECK: 51
// TEST_FEATURE: DnnlUtils_convolution_forward
// TEST_FEATURE: DnnlUtils_convolution_desc

#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    int nDevices;
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    cudnnFilterDescriptor_t filterTensor;
    cudaStream_t stream1;

    int in = 1, ic = 2, ih = 5, iw = 5;
    int on = 1, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 2, fh = 2, fw = 2;

    int filterdim[4] = {fk, fc, fh, fw};
    cudnnSetFilterNdDescriptor(filterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterdim);
    
    float *data, *out, *filter, *bias;

    cudnnConvolutionDescriptor_t covdes;
    cudnnCreateConvolutionDescriptor(&covdes);
    cudnnSetConvolution2dDescriptor(covdes, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(covdes, 2);

    int retCount;
    size_t size;
    void *workspacesize;
    cudnnGetConvolutionForwardWorkspaceSize(handle, dataTensor, filterTensor, covdes, outTensor, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, &size);
    cudaMalloc(&workspacesize, size);

    int dimo[4];
    cudnnGetConvolutionNdForwardOutputDim(covdes, dataTensor, filterTensor, 4, dimo);

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle, &alpha, dataTensor, data, filterTensor, filter, covdes, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, workspacesize, size, &beta, outTensor, out);

    cudaDeviceSynchronize();
    std::vector<float> host_bias;
    std::vector<float> host_out;
    cudaMemcpy(host_bias.data(), bias, sizeof(float) * on * oc * oh * ow, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out.data(), out, sizeof(float) * on * oc * oh * ow, cudaMemcpyDeviceToHost);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    return 0;
}