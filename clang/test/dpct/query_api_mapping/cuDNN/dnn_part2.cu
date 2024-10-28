// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.2
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDropoutBackward | FileCheck %s -check-prefix=cudnnDropoutBackward
// cudnnDropoutBackward: CUDA API:
// cudnnDropoutBackward-NEXT:   cudnnHandle_t h;
// cudnnDropoutBackward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnDropoutBackward-NEXT:   cudnnDropoutBackward(
// cudnnDropoutBackward-NEXT:       h /*cudnnHandle_t*/, d /*cudnnDropoutDescriptor_t*/,
// cudnnDropoutBackward-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
// cudnnDropoutBackward-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/,
// cudnnDropoutBackward-NEXT:       reservespace /*void **/, reservespace_size /*size_t*/);
// cudnnDropoutBackward-NEXT: Is migrated to:
// cudnnDropoutBackward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnDropoutBackward-NEXT:   h.create_engine();
// cudnnDropoutBackward-NEXT:   h.async_dropout_backward(d, diff_dst_d, diff_dst, diff_src_d, diff_src, reservespace, reservespace_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDropoutForward | FileCheck %s -check-prefix=cudnnDropoutForward
// cudnnDropoutForward: CUDA API:
// cudnnDropoutForward-NEXT:   cudnnHandle_t h;
// cudnnDropoutForward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnDropoutForward-NEXT:   cudnnDropoutForward(h /*cudnnHandle_t*/, d /*cudnnDropoutDescriptor_t*/,
// cudnnDropoutForward-NEXT:                       src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnDropoutForward-NEXT:                       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
// cudnnDropoutForward-NEXT:                       reservespace /*void **/, reservespace_size /*size_t*/);
// cudnnDropoutForward-NEXT: Is migrated to:
// cudnnDropoutForward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnDropoutForward-NEXT:   h.create_engine();
// cudnnDropoutForward-NEXT:   h.async_dropout_forward(d, src_d, src, dst_d, dst, reservespace, reservespace_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDropoutGetReserveSpaceSize | FileCheck %s -check-prefix=cudnnDropoutGetReserveSpaceSize
// cudnnDropoutGetReserveSpaceSize: CUDA API:
// cudnnDropoutGetReserveSpaceSize-NEXT:   cudnnDropoutGetReserveSpaceSize(src_d /*cudnnTensorDescriptor_t*/,
// cudnnDropoutGetReserveSpaceSize-NEXT:                                   size /*size_t **/);
// cudnnDropoutGetReserveSpaceSize-NEXT: Is migrated to:
// cudnnDropoutGetReserveSpaceSize-NEXT:   *size = dpct::dnnl::engine_ext::get_dropout_workspace_size(src_d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDropoutGetStatesSize | FileCheck %s -check-prefix=cudnnDropoutGetStatesSize
// cudnnDropoutGetStatesSize: CUDA API:
// cudnnDropoutGetStatesSize-NEXT:   cudnnHandle_t h;
// cudnnDropoutGetStatesSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnDropoutGetStatesSize-NEXT:   cudnnDropoutGetStatesSize(h /*cudnnHandle_t*/, size /*size_t **/);
// cudnnDropoutGetStatesSize-NEXT: Is migrated to:
// cudnnDropoutGetStatesSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnDropoutGetStatesSize-NEXT:   h.create_engine();
// cudnnDropoutGetStatesSize-NEXT:   *size = h.get_dropout_state_size();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnFindConvolutionForwardAlgorithm | FileCheck %s -check-prefix=cudnnFindConvolutionForwardAlgorithm
// cudnnFindConvolutionForwardAlgorithm: CUDA API:
// cudnnFindConvolutionForwardAlgorithm-NEXT:   cudnnHandle_t h;
// cudnnFindConvolutionForwardAlgorithm-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnFindConvolutionForwardAlgorithm-NEXT:   cudnnConvolutionFwdAlgoPerf_t r;
// cudnnFindConvolutionForwardAlgorithm-NEXT:   cudnnFindConvolutionForwardAlgorithm(
// cudnnFindConvolutionForwardAlgorithm-NEXT:       h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnFindConvolutionForwardAlgorithm-NEXT:       filter_d /*cudnnFilterDescriptor_t*/,
// cudnnFindConvolutionForwardAlgorithm-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
// cudnnFindConvolutionForwardAlgorithm-NEXT:       reqc /*int*/, realc /*int **/, &r /*cudnnConvolutionFwdAlgoPerf_t*/);
// cudnnFindConvolutionForwardAlgorithm-NEXT: Is migrated to:
// cudnnFindConvolutionForwardAlgorithm-NEXT:   dpct::dnnl::engine_ext h;
// cudnnFindConvolutionForwardAlgorithm-NEXT:   h.create_engine();
// cudnnFindConvolutionForwardAlgorithm-NEXT:   dpct::dnnl::convolution_algorithm_info r;
// cudnnFindConvolutionForwardAlgorithm-NEXT:   r.algo = dnnl::algorithm::convolution_auto;
// cudnnFindConvolutionForwardAlgorithm-NEXT:   *realc = 1;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetActivationDescriptor | FileCheck %s -check-prefix=cudnnGetActivationDescriptor
// cudnnGetActivationDescriptor: CUDA API:
// cudnnGetActivationDescriptor-NEXT:   cudnnActivationDescriptor_t d;
// cudnnGetActivationDescriptor-NEXT:   cudnnGetActivationDescriptor(d /*cudnnActivationDescriptor_t*/,
// cudnnGetActivationDescriptor-NEXT:                                m /*cudnnActivationMode_t**/,
// cudnnGetActivationDescriptor-NEXT:                                p /*cudnnNanPropagation_t**/, c /*double**/);
// cudnnGetActivationDescriptor-NEXT: Is migrated to:
// cudnnGetActivationDescriptor-NEXT:   dpct::dnnl::activation_desc d;
// cudnnGetActivationDescriptor-NEXT:   d.get(m, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetActivationDescriptorSwishBeta | FileCheck %s -check-prefix=cudnnGetActivationDescriptorSwishBeta
// cudnnGetActivationDescriptorSwishBeta: CUDA API:
// cudnnGetActivationDescriptorSwishBeta-NEXT:   cudnnActivationDescriptor_t d;
// cudnnGetActivationDescriptorSwishBeta-NEXT:   cudnnGetActivationDescriptorSwishBeta(d /*cudnnActivationDescriptor_t*/,
// cudnnGetActivationDescriptorSwishBeta-NEXT:                                         s /*double **/);
// cudnnGetActivationDescriptorSwishBeta-NEXT: Is migrated to:
// cudnnGetActivationDescriptorSwishBeta-NEXT:   dpct::dnnl::activation_desc d;
// cudnnGetActivationDescriptorSwishBeta-NEXT:   *s = d.get_beta();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetBatchNormalizationBackwardExWorkspaceSize | FileCheck %s -check-prefix=cudnnGetBatchNormalizationBackwardExWorkspaceSize
// cudnnGetBatchNormalizationBackwardExWorkspaceSize: CUDA API:
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:   cudnnHandle_t h;
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:   cudnnGetBatchNormalizationBackwardExWorkspaceSize(
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:       h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/,
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:       op /*cudnnBatchNormOps_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:       dst_d /*cudnnTensorDescriptor_t*/, diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:       diff_summand_d /*cudnnTensorDescriptor_t*/,
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/, p_d /*cudnnTensorDescriptor_t*/,
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:       adesc /*cudnnActivationDescriptor_t*/, size /*size_t **/);
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT: Is migrated to:
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:   h.create_engine();
// cudnnGetBatchNormalizationBackwardExWorkspaceSize-NEXT:   *size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize | FileCheck %s -check-prefix=cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize: CUDA API:
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT:   cudnnHandle_t h;
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT:   cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT:       h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/,
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT:       op /*cudnnBatchNormOps_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT:       summand_d /*cudnnTensorDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT:       p_d /*cudnnTensorDescriptor_t*/, adesc /*cudnnActivationDescriptor_t*/,
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT:       size /*size_t **/);
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT: Is migrated to:
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT:   h.create_engine();
// cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize-NEXT:   *size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetBatchNormalizationTrainingExReserveSpaceSize | FileCheck %s -check-prefix=cudnnGetBatchNormalizationTrainingExReserveSpaceSize
// cudnnGetBatchNormalizationTrainingExReserveSpaceSize: CUDA API:
// cudnnGetBatchNormalizationTrainingExReserveSpaceSize-NEXT:   cudnnHandle_t h;
// cudnnGetBatchNormalizationTrainingExReserveSpaceSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetBatchNormalizationTrainingExReserveSpaceSize-NEXT:   cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
// cudnnGetBatchNormalizationTrainingExReserveSpaceSize-NEXT:       h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/,
// cudnnGetBatchNormalizationTrainingExReserveSpaceSize-NEXT:       op /*cudnnBatchNormOps_t*/, adesc /*cudnnActivationDescriptor_t*/,
// cudnnGetBatchNormalizationTrainingExReserveSpaceSize-NEXT:       src_d /*cudnnTensorDescriptor_t*/, size /*size_t **/);
// cudnnGetBatchNormalizationTrainingExReserveSpaceSize-NEXT: Is migrated to:
// cudnnGetBatchNormalizationTrainingExReserveSpaceSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetBatchNormalizationTrainingExReserveSpaceSize-NEXT:   h.create_engine();
// cudnnGetBatchNormalizationTrainingExReserveSpaceSize-NEXT:   *size = h.get_batch_normalization_workspace_size(op, src_d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolution2dDescriptor | FileCheck %s -check-prefix=cudnnGetConvolution2dDescriptor
// cudnnGetConvolution2dDescriptor: CUDA API:
// cudnnGetConvolution2dDescriptor-NEXT:   cudnnConvolutionDescriptor_t d;
// cudnnGetConvolution2dDescriptor-NEXT:   cudnnGetConvolution2dDescriptor(
// cudnnGetConvolution2dDescriptor-NEXT:       d /*cudnnConvolutionDescriptor_t*/, padding_h /*int**/,
// cudnnGetConvolution2dDescriptor-NEXT:       padding_h /*int**/, stride_h /*int**/, stride_w /*int**/,
// cudnnGetConvolution2dDescriptor-NEXT:       dilation_h /*int**/, dilation_w /*int**/, m /*cudnnConvolutionMode_t*/,
// cudnnGetConvolution2dDescriptor-NEXT:       t /*cudnnDataType_t*/);
// cudnnGetConvolution2dDescriptor-NEXT: Is migrated to:
// cudnnGetConvolution2dDescriptor-NEXT:   dpct::dnnl::convolution_desc d;
// cudnnGetConvolution2dDescriptor-NEXT:   d.get(padding_h, padding_h, stride_h, stride_w, dilation_h, dilation_w);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolution2dForwardOutputDim | FileCheck %s -check-prefix=cudnnGetConvolution2dForwardOutputDim
// cudnnGetConvolution2dForwardOutputDim: CUDA API:
// cudnnGetConvolution2dForwardOutputDim-NEXT:   cudnnConvolutionDescriptor_t d;
// cudnnGetConvolution2dForwardOutputDim-NEXT:   cudnnGetConvolution2dForwardOutputDim(
// cudnnGetConvolution2dForwardOutputDim-NEXT:       d /*cudnnConvolutionDescriptor_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolution2dForwardOutputDim-NEXT:       f_d /*cudnnFilterDescriptor_t*/, n /*int**/, c /*int**/, h /*int**/,
// cudnnGetConvolution2dForwardOutputDim-NEXT:       w /*int**/);
// cudnnGetConvolution2dForwardOutputDim-NEXT: Is migrated to:
// cudnnGetConvolution2dForwardOutputDim-NEXT:   dpct::dnnl::convolution_desc d;
// cudnnGetConvolution2dForwardOutputDim-NEXT:   d.get_forward_output_dim(src_d, f_d, n, c, h, w);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionBackwardDataWorkspaceSize | FileCheck %s -check-prefix=cudnnGetConvolutionBackwardDataWorkspaceSize
// cudnnGetConvolutionBackwardDataWorkspaceSize: CUDA API:
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT:   cudnnHandle_t h;
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT:   cudnnGetConvolutionBackwardDataWorkspaceSize(
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT:       h /*cudnnHandle_t*/, filter_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/,
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/, alg /*cudnnConvolutionFwdAlgo_t*/,
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT:       size /*size_t **/);
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT: Is migrated to:
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT:   h.create_engine();
// cudnnGetConvolutionBackwardDataWorkspaceSize-NEXT:   *size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionBackwardFilterWorkspaceSize | FileCheck %s -check-prefix=cudnnGetConvolutionBackwardFilterWorkspaceSize
// cudnnGetConvolutionBackwardFilterWorkspaceSize: CUDA API:
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT:   cudnnHandle_t h;
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT:   cudnnGetConvolutionBackwardFilterWorkspaceSize(
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT:       h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/,
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT:       diff_filter_d /*cudnnFilterDescriptor_t*/,
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT:       alg /*cudnnConvolutionFwdAlgo_t*/, size /*size_t **/);
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT: Is migrated to:
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT:   h.create_engine();
// cudnnGetConvolutionBackwardFilterWorkspaceSize-NEXT:   *size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionForwardWorkspaceSize | FileCheck %s -check-prefix=cudnnGetConvolutionForwardWorkspaceSize
// cudnnGetConvolutionForwardWorkspaceSize: CUDA API:
// cudnnGetConvolutionForwardWorkspaceSize-NEXT:   cudnnHandle_t h;
// cudnnGetConvolutionForwardWorkspaceSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetConvolutionForwardWorkspaceSize-NEXT:   cudnnGetConvolutionForwardWorkspaceSize(
// cudnnGetConvolutionForwardWorkspaceSize-NEXT:       h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionForwardWorkspaceSize-NEXT:       filter_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionForwardWorkspaceSize-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionForwardWorkspaceSize-NEXT:       alg /*cudnnConvolutionFwdAlgo_t*/, size /*size_t **/);
// cudnnGetConvolutionForwardWorkspaceSize-NEXT: Is migrated to:
// cudnnGetConvolutionForwardWorkspaceSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetConvolutionForwardWorkspaceSize-NEXT:   h.create_engine();
// cudnnGetConvolutionForwardWorkspaceSize-NEXT:   *size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionGroupCount | FileCheck %s -check-prefix=cudnnGetConvolutionGroupCount
// cudnnGetConvolutionGroupCount: CUDA API:
// cudnnGetConvolutionGroupCount-NEXT:   cudnnConvolutionDescriptor_t d;
// cudnnGetConvolutionGroupCount-NEXT:   cudnnGetConvolutionGroupCount(d /*cudnnActivationDescriptor_t*/,
// cudnnGetConvolutionGroupCount-NEXT:                                 group_count /*int**/);
// cudnnGetConvolutionGroupCount-NEXT: Is migrated to:
// cudnnGetConvolutionGroupCount-NEXT:   dpct::dnnl::convolution_desc d;
// cudnnGetConvolutionGroupCount-NEXT:   *group_count = d.get_group_count();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionNdDescriptor | FileCheck %s -check-prefix=cudnnGetConvolutionNdDescriptor
// cudnnGetConvolutionNdDescriptor: CUDA API:
// cudnnGetConvolutionNdDescriptor-NEXT:   cudnnConvolutionDescriptor_t d;
// cudnnGetConvolutionNdDescriptor-NEXT:   cudnnGetConvolutionNdDescriptor(
// cudnnGetConvolutionNdDescriptor-NEXT:       d /*cudnnConvolutionDescriptor_t*/, rn /*int*/, n /*int**/,
// cudnnGetConvolutionNdDescriptor-NEXT:       pada /*int[]*/, stridea /*int[]*/, dilationa /*int[]*/,
// cudnnGetConvolutionNdDescriptor-NEXT:       m /*cudnnConvolutionMode_t**/, t /*cudnnDataType_t**/);
// cudnnGetConvolutionNdDescriptor-NEXT: Is migrated to:
// cudnnGetConvolutionNdDescriptor-NEXT:   dpct::dnnl::convolution_desc d;
// cudnnGetConvolutionNdDescriptor-NEXT:   d.get(rn, n, pada, stridea, dilationa);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionNdForwardOutputDim | FileCheck %s -check-prefix=cudnnGetConvolutionNdForwardOutputDim
// cudnnGetConvolutionNdForwardOutputDim: CUDA API:
// cudnnGetConvolutionNdForwardOutputDim-NEXT:   cudnnConvolutionDescriptor_t d;
// cudnnGetConvolutionNdForwardOutputDim-NEXT:   cudnnGetConvolutionNdForwardOutputDim(
// cudnnGetConvolutionNdForwardOutputDim-NEXT:       d /*cudnnPoolingDescriptor_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionNdForwardOutputDim-NEXT:       f_d /*cudnnTensorDescriptor_t*/, n /*int*/, da /*int[]*/);
// cudnnGetConvolutionNdForwardOutputDim-NEXT: Is migrated to:
// cudnnGetConvolutionNdForwardOutputDim-NEXT:   dpct::dnnl::convolution_desc d;
// cudnnGetConvolutionNdForwardOutputDim-NEXT:   d.get_forward_output_dim(src_d, f_d, n, da);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetDropoutDescriptor | FileCheck %s -check-prefix=cudnnGetDropoutDescriptor
// cudnnGetDropoutDescriptor: CUDA API:
// cudnnGetDropoutDescriptor-NEXT:   cudnnDropoutDescriptor_t d;
// cudnnGetDropoutDescriptor-NEXT:   cudnnGetDropoutDescriptor(d /*cudnnDropoutDescriptor_t*/, h /*cudnnHandle_t*/,
// cudnnGetDropoutDescriptor-NEXT:                             dropout /*float **/, states /*void ***/,
// cudnnGetDropoutDescriptor-NEXT:                             seed /*unsigned long long **/);
// cudnnGetDropoutDescriptor-NEXT: Is migrated to:
// cudnnGetDropoutDescriptor-NEXT:   dpct::dnnl::dropout_desc d;
// cudnnGetDropoutDescriptor-NEXT:   d.get(dropout, states, seed);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetErrorString | FileCheck %s -check-prefix=cudnnGetErrorString
// cudnnGetErrorString: CUDA API:
// cudnnGetErrorString-NEXT:   r = cudnnGetErrorString(s /*cudnnStatus_t*/);
// cudnnGetErrorString-NEXT: Is migrated to:
// cudnnGetErrorString-NEXT:   /*
// cudnnGetErrorString-NEXT:   DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
// cudnnGetErrorString-NEXT:   */
// cudnnGetErrorString-NEXT:   r = dpct::get_error_string_dummy(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetFilter4dDescriptor | FileCheck %s -check-prefix=cudnnGetFilter4dDescriptor
// cudnnGetFilter4dDescriptor: CUDA API:
// cudnnGetFilter4dDescriptor-NEXT:   cudnnFilterDescriptor_t d;
// cudnnGetFilter4dDescriptor-NEXT:   cudnnGetFilter4dDescriptor(d /*cudnnFilterDescriptor_t*/,
// cudnnGetFilter4dDescriptor-NEXT:                              t /*cudnnDataType_t **/,
// cudnnGetFilter4dDescriptor-NEXT:                              f /*cudnnTensorFormat_t **/, k /*int **/,
// cudnnGetFilter4dDescriptor-NEXT:                              c /*int **/, h /*int **/, w /*int **/);
// cudnnGetFilter4dDescriptor-NEXT: Is migrated to:
// cudnnGetFilter4dDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnGetFilter4dDescriptor-NEXT:   d.get(t, f, k, c, h, w);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetFilterNdDescriptor | FileCheck %s -check-prefix=cudnnGetFilterNdDescriptor
// cudnnGetFilterNdDescriptor: CUDA API:
// cudnnGetFilterNdDescriptor-NEXT:   cudnnFilterDescriptor_t d;
// cudnnGetFilterNdDescriptor-NEXT:   cudnnGetFilterNdDescriptor(d /*cudnnFilterDescriptor_t*/, rn /*int*/,
// cudnnGetFilterNdDescriptor-NEXT:                              t /*cudnnDataType_t*/, f /*cudnnTensorFormat_t*/,
// cudnnGetFilterNdDescriptor-NEXT:                              n /*int **/, da /*int[]*/);
// cudnnGetFilterNdDescriptor-NEXT: Is migrated to:
// cudnnGetFilterNdDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnGetFilterNdDescriptor-NEXT:   d.get(rn, t, f, n, da);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetFilterSizeInBytes | FileCheck %s -check-prefix=cudnnGetFilterSizeInBytes
// cudnnGetFilterSizeInBytes: CUDA API:
// cudnnGetFilterSizeInBytes-NEXT:   cudnnFilterDescriptor_t d;
// cudnnGetFilterSizeInBytes-NEXT:   cudnnGetFilterSizeInBytes(d /*cudnnFilterDescriptor_t*/, size /*size_t **/);
// cudnnGetFilterSizeInBytes-NEXT: Is migrated to:
// cudnnGetFilterSizeInBytes-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnGetFilterSizeInBytes-NEXT:   *size = d.get_size();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetLRNDescriptor | FileCheck %s -check-prefix=cudnnGetLRNDescriptor
// cudnnGetLRNDescriptor: CUDA API:
// cudnnGetLRNDescriptor-NEXT:   cudnnLRNDescriptor_t d;
// cudnnGetLRNDescriptor-NEXT:   cudnnGetLRNDescriptor(d /*cudnnLRNDescriptor_t*/, n /*unsigned**/,
// cudnnGetLRNDescriptor-NEXT:                         alpha /*double**/, beta /*double**/, k /*double**/);
// cudnnGetLRNDescriptor-NEXT: Is migrated to:
// cudnnGetLRNDescriptor-NEXT:   dpct::dnnl::lrn_desc d;
// cudnnGetLRNDescriptor-NEXT:   d.get(n, alpha, beta, k);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetNormalizationBackwardWorkspaceSize | FileCheck %s -check-prefix=cudnnGetNormalizationBackwardWorkspaceSize
// cudnnGetNormalizationBackwardWorkspaceSize: CUDA API:
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:   cudnnHandle_t h;
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:   cudnnGetNormalizationBackwardWorkspaceSize(
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:       h /*cudnnHandle_t*/, m /*cudnnNormMode_t*/, op /*cudnnNormOps_t*/,
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:       alg /*cudnnNormAlgo_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:       dst_d /*cudnnTensorDescriptor_t*/, diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:       diff_summand_d /*cudnnTensorDescriptor_t*/,
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:       diff_p1_d /*cudnnTensorDescriptor_t*/,
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:       adesc /*cudnnActivationDescriptor_t*/, p2_d /*cudnnTensorDescriptor_t*/,
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:       size /*size_t **/, group_count /*int*/);
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT: Is migrated to:
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:   h.create_engine();
// cudnnGetNormalizationBackwardWorkspaceSize-NEXT:   *size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetNormalizationForwardTrainingWorkspaceSize | FileCheck %s -check-prefix=cudnnGetNormalizationForwardTrainingWorkspaceSize
// cudnnGetNormalizationForwardTrainingWorkspaceSize: CUDA API:
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT:   cudnnHandle_t h;
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT:   cudnnGetNormalizationForwardTrainingWorkspaceSize(
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT:       h /*cudnnHandle_t*/, m /*cudnnNormMode_t*/, op /*cudnnNormOps_t*/,
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT:       alg /*cudnnNormAlgo_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT:       summand_d /*cudnnTensorDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT:       p1_d /*cudnnTensorDescriptor_t*/, adesc /*cudnnActivationDescriptor_t*/,
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT:       p2_d /*cudnnTensorDescriptor_t*/, size /*size_t **/, group_count /*int*/);
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT: Is migrated to:
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT:   h.create_engine();
// cudnnGetNormalizationForwardTrainingWorkspaceSize-NEXT:   *size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetNormalizationTrainingReserveSpaceSize | FileCheck %s -check-prefix=cudnnGetNormalizationTrainingReserveSpaceSize
// cudnnGetNormalizationTrainingReserveSpaceSize: CUDA API:
// cudnnGetNormalizationTrainingReserveSpaceSize-NEXT:   cudnnHandle_t h;
// cudnnGetNormalizationTrainingReserveSpaceSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetNormalizationTrainingReserveSpaceSize-NEXT:   cudnnGetNormalizationTrainingReserveSpaceSize(
// cudnnGetNormalizationTrainingReserveSpaceSize-NEXT:       h /*cudnnHandle_t*/, m /*cudnnNormMode_t*/, op /*cudnnNormOps_t*/,
// cudnnGetNormalizationTrainingReserveSpaceSize-NEXT:       alg /*cudnnNormAlgo_t*/, adesc /*cudnnActivationDescriptor_t*/,
// cudnnGetNormalizationTrainingReserveSpaceSize-NEXT:       src_d /*cudnnTensorDescriptor_t*/, size /*size_t **/,
// cudnnGetNormalizationTrainingReserveSpaceSize-NEXT:       group_count /*int*/);
// cudnnGetNormalizationTrainingReserveSpaceSize-NEXT: Is migrated to:
// cudnnGetNormalizationTrainingReserveSpaceSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetNormalizationTrainingReserveSpaceSize-NEXT:   h.create_engine();
// cudnnGetNormalizationTrainingReserveSpaceSize-NEXT:   *size = h.get_batch_normalization_workspace_size(op, src_d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetOpTensorDescriptor | FileCheck %s -check-prefix=cudnnGetOpTensorDescriptor
// cudnnGetOpTensorDescriptor: CUDA API:
// cudnnGetOpTensorDescriptor-NEXT:   cudnnOpTensorDescriptor_t d;
// cudnnGetOpTensorDescriptor-NEXT:   cudnnGetOpTensorDescriptor(d /*cudnnOpTensorDescriptor_t*/,
// cudnnGetOpTensorDescriptor-NEXT:                              op /*cudnnOpTensorOp_t**/, dt /*cudnnDataType_t**/,
// cudnnGetOpTensorDescriptor-NEXT:                              p /*cudnnNanPropagation_t**/);
// cudnnGetOpTensorDescriptor-NEXT: Is migrated to:
// cudnnGetOpTensorDescriptor-NEXT:   dpct::dnnl::binary_op d;
// cudnnGetOpTensorDescriptor-NEXT:   *op = d;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetPooling2dDescriptor | FileCheck %s -check-prefix=cudnnGetPooling2dDescriptor
// cudnnGetPooling2dDescriptor: CUDA API:
// cudnnGetPooling2dDescriptor-NEXT:   cudnnPoolingDescriptor_t d;
// cudnnGetPooling2dDescriptor-NEXT:   cudnnGetPooling2dDescriptor(
// cudnnGetPooling2dDescriptor-NEXT:       d /*cudnnPoolingDescriptor_t*/, m /*cudnnPoolingMode_t**/,
// cudnnGetPooling2dDescriptor-NEXT:       p /*cudnnNanPropagation_t**/, h /*int**/, w /*int**/, vp /*int**/,
// cudnnGetPooling2dDescriptor-NEXT:       hp /*int**/, vs /*int**/, hs /*int**/);
// cudnnGetPooling2dDescriptor-NEXT: Is migrated to:
// cudnnGetPooling2dDescriptor-NEXT:   dpct::dnnl::pooling_desc d;
// cudnnGetPooling2dDescriptor-NEXT:   d.get(m, h, w, vp, hp, vs, hs);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetPooling2dForwardOutputDim | FileCheck %s -check-prefix=cudnnGetPooling2dForwardOutputDim
// cudnnGetPooling2dForwardOutputDim: CUDA API:
// cudnnGetPooling2dForwardOutputDim-NEXT:   cudnnPoolingDescriptor_t d;
// cudnnGetPooling2dForwardOutputDim-NEXT:   cudnnGetPooling2dForwardOutputDim(
// cudnnGetPooling2dForwardOutputDim-NEXT:       d /*cudnnPoolingDescriptor_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetPooling2dForwardOutputDim-NEXT:       n /*int**/, c /*int**/, h /*int**/, w /*int**/);
// cudnnGetPooling2dForwardOutputDim-NEXT: Is migrated to:
// cudnnGetPooling2dForwardOutputDim-NEXT:   dpct::dnnl::pooling_desc d;
// cudnnGetPooling2dForwardOutputDim-NEXT:   d.get_forward_output_dim(src_d, n, c, h, w);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetPoolingNdDescriptor | FileCheck %s -check-prefix=cudnnGetPoolingNdDescriptor
// cudnnGetPoolingNdDescriptor: CUDA API:
// cudnnGetPoolingNdDescriptor-NEXT:   cudnnPoolingDescriptor_t d;
// cudnnGetPoolingNdDescriptor-NEXT:   cudnnGetPoolingNdDescriptor(d /*cudnnPoolingDescriptor_t*/, rn /*int*/,
// cudnnGetPoolingNdDescriptor-NEXT:                               m /*cudnnPoolingMode_t**/,
// cudnnGetPoolingNdDescriptor-NEXT:                               p /*cudnnNanPropagation_t**/, nd /*int**/,
// cudnnGetPoolingNdDescriptor-NEXT:                               da /*int[]*/, pa /*int[]*/, sa /*int[]*/);
// cudnnGetPoolingNdDescriptor-NEXT: Is migrated to:
// cudnnGetPoolingNdDescriptor-NEXT:   dpct::dnnl::pooling_desc d;
// cudnnGetPoolingNdDescriptor-NEXT:   d.get(rn, m, nd, da, pa, sa);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetPoolingNdForwardOutputDim | FileCheck %s -check-prefix=cudnnGetPoolingNdForwardOutputDim
// cudnnGetPoolingNdForwardOutputDim: CUDA API:
// cudnnGetPoolingNdForwardOutputDim-NEXT:   cudnnPoolingDescriptor_t d;
// cudnnGetPoolingNdForwardOutputDim-NEXT:   cudnnGetPoolingNdForwardOutputDim(d /*cudnnPoolingDescriptor_t*/,
// cudnnGetPoolingNdForwardOutputDim-NEXT:                                     src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetPoolingNdForwardOutputDim-NEXT:                                     n /*int*/, da /*int[]*/);
// cudnnGetPoolingNdForwardOutputDim-NEXT: Is migrated to:
// cudnnGetPoolingNdForwardOutputDim-NEXT:   dpct::dnnl::pooling_desc d;
// cudnnGetPoolingNdForwardOutputDim-NEXT:   d.get_forward_output_dim(src_d, n, da);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetReduceTensorDescriptor | FileCheck %s -check-prefix=cudnnGetReduceTensorDescriptor
// cudnnGetReduceTensorDescriptor: CUDA API:
// cudnnGetReduceTensorDescriptor-NEXT:   cudnnReduceTensorDescriptor_t d;
// cudnnGetReduceTensorDescriptor-NEXT:   cudnnGetReduceTensorDescriptor(
// cudnnGetReduceTensorDescriptor-NEXT:       d /*cudnnReduceTensorDescriptor_t*/, o /*cudnnPoolingMode_t**/,
// cudnnGetReduceTensorDescriptor-NEXT:       dt /*cudnnDataType_t**/, p /*cudnnNanPropagation_t**/,
// cudnnGetReduceTensorDescriptor-NEXT:       i /*cudnnReduceTensorIndices_t**/, it /*cudnnIndicesType_t**/);
// cudnnGetReduceTensorDescriptor-NEXT: Is migrated to:
// cudnnGetReduceTensorDescriptor-NEXT:   dpct::dnnl::reduction_op d;
// cudnnGetReduceTensorDescriptor-NEXT:   *o = d;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetReductionWorkspaceSize | FileCheck %s -check-prefix=cudnnGetReductionWorkspaceSize
// cudnnGetReductionWorkspaceSize: CUDA API:
// cudnnGetReductionWorkspaceSize-NEXT:   cudnnHandle_t h;
// cudnnGetReductionWorkspaceSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetReductionWorkspaceSize-NEXT:   cudnnGetReductionWorkspaceSize(
// cudnnGetReductionWorkspaceSize-NEXT:       h /*cudnnHandle_t*/, d /*cudnnReduceTensorDescriptor_t*/,
// cudnnGetReductionWorkspaceSize-NEXT:       src_d /*cudnnTensorDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetReductionWorkspaceSize-NEXT:       size /*size_t **/);
// cudnnGetReductionWorkspaceSize-NEXT: Is migrated to:
// cudnnGetReductionWorkspaceSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetReductionWorkspaceSize-NEXT:   h.create_engine();
// cudnnGetReductionWorkspaceSize-NEXT:   *size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetRNNDataDescriptor | FileCheck %s -check-prefix=cudnnGetRNNDataDescriptor
// cudnnGetRNNDataDescriptor: CUDA API:
// cudnnGetRNNDataDescriptor-NEXT:   cudnnRNNDataDescriptor_t d;
// cudnnGetRNNDataDescriptor-NEXT:   cudnnGetRNNDataDescriptor(
// cudnnGetRNNDataDescriptor-NEXT:       d /*cudnnRNNDataDescriptor_t*/, t /*cudnnDataType_t **/,
// cudnnGetRNNDataDescriptor-NEXT:       l /*cudnnRNNDataLayout_t **/, len /*int **/, b /*int **/, v /*int **/,
// cudnnGetRNNDataDescriptor-NEXT:       rlen /*int*/, sa /*int[]*/, p /*void **/);
// cudnnGetRNNDataDescriptor-NEXT: Is migrated to:
// cudnnGetRNNDataDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnGetRNNDataDescriptor-NEXT:   d.get(t, l, len, b, v);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetRNNDescriptor_v8 | FileCheck %s -check-prefix=cudnnGetRNNDescriptor_v8
// cudnnGetRNNDescriptor_v8: CUDA API:
// cudnnGetRNNDescriptor_v8-NEXT:   cudnnRNNDescriptor_t d;
// cudnnGetRNNDescriptor_v8-NEXT:   cudnnGetRNNDescriptor_v8(
// cudnnGetRNNDescriptor_v8-NEXT:       d /*cudnnRNNDescriptor_t*/, alg /*cudnnRNNAlgo_t **/,
// cudnnGetRNNDescriptor_v8-NEXT:       m /*cudnnRNNMode_t **/, bm /*cudnnRNNBiasMode_t **/,
// cudnnGetRNNDescriptor_v8-NEXT:       dm /*cudnnDirectionMode_t **/, im /*cudnnRNNInputMode_t **/,
// cudnnGetRNNDescriptor_v8-NEXT:       t /*cudnnDataType_t **/, mp /*cudnnDataType_t **/,
// cudnnGetRNNDescriptor_v8-NEXT:       mt /*cudnnMathType_t **/, is /*int32_t **/, hs /*int32_t **/,
// cudnnGetRNNDescriptor_v8-NEXT:       ps /*int32_t **/, l /*int32_t **/, dropout /*cudnnDropoutDescriptor_t **/,
// cudnnGetRNNDescriptor_v8-NEXT:       f /*uint32_t **/);
// cudnnGetRNNDescriptor_v8-NEXT: Is migrated to:
// cudnnGetRNNDescriptor_v8-NEXT:   dpct::dnnl::rnn_desc d;
// cudnnGetRNNDescriptor_v8-NEXT:   d.get(m, bm, dm, t, is, hs, ps, l);
