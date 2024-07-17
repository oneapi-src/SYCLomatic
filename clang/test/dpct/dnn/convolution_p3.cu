// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/convolution_p3 %S/convolution_p3.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/convolution_p3/convolution_p3.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    cudnnHandle_t handle;
    cudnnConvolutionDescriptor_t covdes;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    cudnnFilterDescriptor_t filterTensor;
    cudnnCreate(&handle);

    int returned_count = 1;

    // CHECK: dpct::dnnl::convolution_algorithm_info fwd_perf;
    // CHECK: dpct::dnnl::convolution_algorithm_info bwd_filter_perf;
    // CHECK: dpct::dnnl::convolution_algorithm_info bwd_data_perf;
    cudnnConvolutionFwdAlgoPerf_t fwd_perf;
    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf;
    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf;

    // CHECK: fwd_perf.algo = dnnl::algorithm::convolution_auto;
    // CHECK: returned_count = 1;
    // CHECK: bwd_data_perf.algo = dnnl::algorithm::convolution_auto;
    // CHECK: returned_count = 1;
    // CHECK: bwd_filter_perf.algo = dnnl::algorithm::convolution_auto;
    // CHECK: returned_count = 1;
    cudnnFindConvolutionForwardAlgorithm(handle, dataTensor, filterTensor, covdes, outTensor, 1, &returned_count, &fwd_perf);
    cudnnFindConvolutionBackwardDataAlgorithm(handle, filterTensor, outTensor, covdes, dataTensor, 1, &returned_count, &bwd_data_perf);
    cudnnFindConvolutionBackwardFilterAlgorithm(handle, dataTensor, outTensor, covdes, filterTensor, 1, &returned_count, &bwd_filter_perf);

    // CHECK: fwd_perf.algo = dnnl::algorithm::convolution_auto;
    // CHECK: returned_count = 1;
    // CHECK: bwd_filter_perf.algo = dnnl::algorithm::convolution_auto;
    // CHECK: returned_count = 1;
    // CHECK: bwd_data_perf.algo = dnnl::algorithm::convolution_auto;
    // CHECK: returned_count = 1;
    cudnnGetConvolutionForwardAlgorithm_v7(handle, dataTensor, filterTensor, covdes, outTensor, 1, &returned_count, &fwd_perf);
    cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, dataTensor, outTensor, covdes, filterTensor, 1, &returned_count, &bwd_filter_perf);
    cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterTensor, outTensor, covdes, dataTensor, 1, &returned_count, &bwd_data_perf);

    int max_count = 0;
    // CHECK: max_count = 1;
    // CHECK: max_count = 1;
    // CHECK: max_count = 1;
    cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, &max_count);
    cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, &max_count);
    cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &max_count);

    return 0;
}
