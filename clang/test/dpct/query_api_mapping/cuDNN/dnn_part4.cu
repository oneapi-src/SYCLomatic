// UNSUPPORTED: v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6
// UNSUPPORTED: cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionBackwardDataAlgorithm | FileCheck %s -check-prefix=cudnnGetConvolutionBackwardDataAlgorithm
// cudnnGetConvolutionBackwardDataAlgorithm: CUDA API:
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:   cudnnHandle_t h;
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:   cudnnGetConvolutionBackwardDataAlgorithm(
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:       h /*cudnnHandle_t*/, filter_d /*cudnnFilterDescriptor_t*/,
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/,
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:       preference /*cudnnConvolutionBwdDataPreference_t*/, limit /*size_t*/,
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:       alg /*cudnnConvolutionBwdDataAlgo_t **/);
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT: Is migrated to:
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:   h.create_engine();
// cudnnGetConvolutionBackwardDataAlgorithm-NEXT:   *alg = dnnl::algorithm::convolution_auto;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionBackwardFilterAlgorithm | FileCheck %s -check-prefix=cudnnGetConvolutionBackwardFilterAlgorithm
// cudnnGetConvolutionBackwardFilterAlgorithm: CUDA API:
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:   cudnnHandle_t h;
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:   cudnnGetConvolutionBackwardFilterAlgorithm(
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:       h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/,
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:       diff_filter_d /*cudnnFilterDescriptor_t*/,
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:       preference /*cudnnConvolutionBwdFilterPreference_t*/, limit /*size_t*/,
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:       alg /*cudnnConvolutionBwdFilterAlgo_t **/);
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT: Is migrated to:
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:   h.create_engine();
// cudnnGetConvolutionBackwardFilterAlgorithm-NEXT:   *alg = dnnl::algorithm::convolution_auto;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionForwardAlgorithm | FileCheck %s -check-prefix=cudnnGetConvolutionForwardAlgorithm
// cudnnGetConvolutionForwardAlgorithm: CUDA API:
// cudnnGetConvolutionForwardAlgorithm-NEXT:   cudnnHandle_t h;
// cudnnGetConvolutionForwardAlgorithm-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetConvolutionForwardAlgorithm-NEXT:   cudnnGetConvolutionForwardAlgorithm(
// cudnnGetConvolutionForwardAlgorithm-NEXT:       h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionForwardAlgorithm-NEXT:       filter_d /*cudnnFilterDescriptor_t*/,
// cudnnGetConvolutionForwardAlgorithm-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionForwardAlgorithm-NEXT:       preference /*cudnnConvolutionFwdPreference_t*/, limit /*size_t*/,
// cudnnGetConvolutionForwardAlgorithm-NEXT:       alg /*cudnnConvolutionFwdAlgo_t **/);
// cudnnGetConvolutionForwardAlgorithm-NEXT: Is migrated to:
// cudnnGetConvolutionForwardAlgorithm-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetConvolutionForwardAlgorithm-NEXT:   h.create_engine();
// cudnnGetConvolutionForwardAlgorithm-NEXT:   *alg = dnnl::algorithm::convolution_auto;
