// UNSUPPORTED: cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1
// UNSUPPORTED: v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1
// RUN: dpct -in-root %S -out-root %T/convolution_v7 %S/convolution_v7.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/convolution_v7/convolution_v7.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor;
    cudnnFilterDescriptor_t filterTensor;
    cudnnConvolutionDescriptor_t covdes;
    // CHECK: dnnl::algorithm bwd_filter_a;
    // CHECK: dnnl::algorithm bwd_data_a;
    // CHECK: dnnl::algorithm fwd_a;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_a;
    cudnnConvolutionBwdDataAlgo_t bwd_data_a;
    cudnnConvolutionFwdAlgo_t fwd_a;
    // CHECK: bwd_filter_a = dnnl::algorithm::convolution_auto;
    cudnnGetConvolutionBackwardFilterAlgorithm(handle, dataTensor, dataTensor, covdes, filterTensor, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 10, &bwd_filter_a);
    // CHECK: bwd_data_a = dnnl::algorithm::convolution_auto;
    cudnnGetConvolutionBackwardDataAlgorithm(handle, filterTensor, dataTensor, covdes, dataTensor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 10, &bwd_data_a);    
    // CHECK: fwd_a = dnnl::algorithm::convolution_auto;
    cudnnGetConvolutionForwardAlgorithm(handle, dataTensor, filterTensor, covdes, dataTensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 10, &fwd_a);

    return 0;
}
