// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.2
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetRNNTempSpaceSizes | FileCheck %s -check-prefix=cudnnGetRNNTempSpaceSizes
// cudnnGetRNNTempSpaceSizes: CUDA API:
// cudnnGetRNNTempSpaceSizes-NEXT:   cudnnHandle_t h;
// cudnnGetRNNTempSpaceSizes-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetRNNTempSpaceSizes-NEXT:   cudnnGetRNNTempSpaceSizes(
// cudnnGetRNNTempSpaceSizes-NEXT:       h /*cudnnHandle_t*/, d /*cudnnReduceTensorDescriptor_t*/,
// cudnnGetRNNTempSpaceSizes-NEXT:       m /*cudnnReduceTensorDescriptor_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnGetRNNTempSpaceSizes-NEXT:       workspace_size /*size_t **/, reservespace_size /*size_t **/);
// cudnnGetRNNTempSpaceSizes-NEXT: Is migrated to:
// cudnnGetRNNTempSpaceSizes-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetRNNTempSpaceSizes-NEXT:   h.create_engine();
// cudnnGetRNNTempSpaceSizes-NEXT:   h.rnn_get_scratchpad_workspace_size(d, m, src_d, workspace_size, reservespace_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetRNNWeightSpaceSize | FileCheck %s -check-prefix=cudnnGetRNNWeightSpaceSize
// cudnnGetRNNWeightSpaceSize: CUDA API:
// cudnnGetRNNWeightSpaceSize-NEXT:   cudnnHandle_t h;
// cudnnGetRNNWeightSpaceSize-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetRNNWeightSpaceSize-NEXT:   cudnnGetRNNWeightSpaceSize(h /*cudnnHandle_t*/, d /*cudnnRNNDescriptor_t*/,
// cudnnGetRNNWeightSpaceSize-NEXT:                              size /*size_t **/);
// cudnnGetRNNWeightSpaceSize-NEXT: Is migrated to:
// cudnnGetRNNWeightSpaceSize-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetRNNWeightSpaceSize-NEXT:   h.create_engine();
// cudnnGetRNNWeightSpaceSize-NEXT:   h.rnn_get_weight_space_size(d, size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetStream | FileCheck %s -check-prefix=cudnnGetStream
// cudnnGetStream: CUDA API:
// cudnnGetStream-NEXT:   cudnnHandle_t h;
// cudnnGetStream-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetStream-NEXT:   cudnnGetStream(h /*cudnnHandle_t*/, s /*cudaStream_t **/);
// cudnnGetStream-NEXT: Is migrated to:
// cudnnGetStream-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetStream-NEXT:   h.create_engine();
// cudnnGetStream-NEXT:   *s = h.get_queue();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetTensorNdDescriptor | FileCheck %s -check-prefix=cudnnGetTensorNdDescriptor
// cudnnGetTensorNdDescriptor: CUDA API:
// cudnnGetTensorNdDescriptor-NEXT:   cudnnTensorDescriptor_t d;
// cudnnGetTensorNdDescriptor-NEXT:   cudnnGetTensorNdDescriptor(d /*cudnnTensorDescriptor_t*/, rn /*int*/,
// cudnnGetTensorNdDescriptor-NEXT:                              t /*cudnnDataType_t **/, n /*int **/, da /*int[]*/,
// cudnnGetTensorNdDescriptor-NEXT:                              sa /*int[]*/);
// cudnnGetTensorNdDescriptor-NEXT: Is migrated to:
// cudnnGetTensorNdDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnGetTensorNdDescriptor-NEXT:   d.get(rn, t, n, da, sa);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetTensorSizeInBytes | FileCheck %s -check-prefix=cudnnGetTensorSizeInBytes
// cudnnGetTensorSizeInBytes: CUDA API:
// cudnnGetTensorSizeInBytes-NEXT:   cudnnTensorDescriptor_t d;
// cudnnGetTensorSizeInBytes-NEXT:   cudnnGetTensorSizeInBytes(d /*cudnnTensorDescriptor_t*/, size /*size_t **/);
// cudnnGetTensorSizeInBytes-NEXT: Is migrated to:
// cudnnGetTensorSizeInBytes-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnGetTensorSizeInBytes-NEXT:   *size = d.get_size();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetVersion | FileCheck %s -check-prefix=cudnnGetVersion
// cudnnGetVersion: CUDA API:
// cudnnGetVersion-NEXT:   version = cudnnGetVersion();
// cudnnGetVersion-NEXT: Is migrated to:
// cudnnGetVersion-NEXT:   version = dpct::dnnl::get_version();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnLRNCrossChannelBackward | FileCheck %s -check-prefix=cudnnLRNCrossChannelBackward
// cudnnLRNCrossChannelBackward: CUDA API:
// cudnnLRNCrossChannelBackward-NEXT:   cudnnHandle_t h;
// cudnnLRNCrossChannelBackward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnLRNCrossChannelBackward-NEXT:   cudnnLRNCrossChannelBackward(
// cudnnLRNCrossChannelBackward-NEXT:       h /*cudnnHandle_t*/, desc /*cudnnLRNDescriptor_t*/, m /*cudnnLRNMode_t*/,
// cudnnLRNCrossChannelBackward-NEXT:       alpha /*void **/, dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
// cudnnLRNCrossChannelBackward-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
// cudnnLRNCrossChannelBackward-NEXT:       src_d /*cudnnTensorDescriptor_t*/, src /*void **/, beta /*void **/,
// cudnnLRNCrossChannelBackward-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/);
// cudnnLRNCrossChannelBackward-NEXT: Is migrated to:
// cudnnLRNCrossChannelBackward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnLRNCrossChannelBackward-NEXT:   h.create_engine();
// cudnnLRNCrossChannelBackward-NEXT:   h.async_lrn_backward(desc, *alpha, dst_d, dst, diff_dst_d, diff_dst, src_d, src, *beta, diff_src_d, diff_src);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnLRNCrossChannelForward | FileCheck %s -check-prefix=cudnnLRNCrossChannelForward
// cudnnLRNCrossChannelForward: CUDA API:
// cudnnLRNCrossChannelForward-NEXT:   cudnnHandle_t h;
// cudnnLRNCrossChannelForward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnLRNCrossChannelForward-NEXT:   cudnnLRNCrossChannelForward(
// cudnnLRNCrossChannelForward-NEXT:       h /*cudnnHandle_t*/, desc /*cudnnLRNDescriptor_t*/, m /*cudnnLRNMode_t*/,
// cudnnLRNCrossChannelForward-NEXT:       alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnLRNCrossChannelForward-NEXT:       beta /*void **/, dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
// cudnnLRNCrossChannelForward-NEXT: Is migrated to:
// cudnnLRNCrossChannelForward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnLRNCrossChannelForward-NEXT:   h.create_engine();
// cudnnLRNCrossChannelForward-NEXT:   h.async_lrn_forward(desc, *alpha, src_d, src, *beta, dst_d, dst);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnNormalizationBackward | FileCheck %s -check-prefix=cudnnNormalizationBackward
// cudnnNormalizationBackward: CUDA API:
// cudnnNormalizationBackward-NEXT:   cudnnHandle_t h;
// cudnnNormalizationBackward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnNormalizationBackward-NEXT:   cudnnNormalizationBackward(
// cudnnNormalizationBackward-NEXT:       h /*cudnnHandle_t*/, m /*cudnnNormMode_t*/, op /*cudnnNormOps_t*/,
// cudnnNormalizationBackward-NEXT:       alg /*cudnnNormAlgo_t*/, diff_alphad /*void **/, diff_betad /*void **/,
// cudnnNormalizationBackward-NEXT:       diff_alphap /*void **/, diff_betap /*void **/,
// cudnnNormalizationBackward-NEXT:       src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnNormalizationBackward-NEXT:       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
// cudnnNormalizationBackward-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
// cudnnNormalizationBackward-NEXT:       diff_summand_d /*cudnnTensorDescriptor_t*/, diff_summand /*void **/,
// cudnnNormalizationBackward-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/,
// cudnnNormalizationBackward-NEXT:       p1_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
// cudnnNormalizationBackward-NEXT:       diff_scale /*void **/, diff_bias /*void **/, eps /*double*/,
// cudnnNormalizationBackward-NEXT:       p2_d /*cudnnTensorDescriptor_t*/, smean /*void **/, svar /*void **/,
// cudnnNormalizationBackward-NEXT:       adesc /*cudnnActivationDescriptor_t*/, workspace /*void **/,
// cudnnNormalizationBackward-NEXT:       workspace_size /*size_t*/, reservespace /*void **/,
// cudnnNormalizationBackward-NEXT:       reservespace_size /*size_t*/, group_count /*int*/);
// cudnnNormalizationBackward-NEXT: Is migrated to:
// cudnnNormalizationBackward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnNormalizationBackward-NEXT:   h.create_engine();
// cudnnNormalizationBackward-NEXT:   h.async_batch_normalization_backward(m, op, adesc, eps, *diff_alphad, src_d, src, dst_d, dst, diff_dst_d, diff_dst, *diff_betad, diff_src_d, diff_src, diff_summand_d, diff_summand, *diff_alphap, p1_d, scale, bias, *diff_betap, diff_scale, diff_bias, p2_d, smean, svar, reservespace_size, reservespace);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnNormalizationForwardInference | FileCheck %s -check-prefix=cudnnNormalizationForwardInference
// cudnnNormalizationForwardInference: CUDA API:
// cudnnNormalizationForwardInference-NEXT:   cudnnHandle_t h;
// cudnnNormalizationForwardInference-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnNormalizationForwardInference-NEXT:   cudnnNormalizationForwardInference(
// cudnnNormalizationForwardInference-NEXT:       h /*cudnnHandle_t*/, m /*cudnnNormMode_t*/, op /*cudnnNormOps_t*/,
// cudnnNormalizationForwardInference-NEXT:       alg /*cudnnNormAlgo_t*/, alpha /*void **/, beta /*void **/,
// cudnnNormalizationForwardInference-NEXT:       src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnNormalizationForwardInference-NEXT:       p1_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
// cudnnNormalizationForwardInference-NEXT:       p2_d /*cudnnTensorDescriptor_t*/, emean /*void **/, evar /*void **/,
// cudnnNormalizationForwardInference-NEXT:       summand_d /*cudnnTensorDescriptor_t*/, summand /*void **/,
// cudnnNormalizationForwardInference-NEXT:       adesc /*cudnnActivationDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
// cudnnNormalizationForwardInference-NEXT:       dst /*void **/, eps /*double*/, group_count /*int*/);
// cudnnNormalizationForwardInference-NEXT: Is migrated to:
// cudnnNormalizationForwardInference-NEXT:   dpct::dnnl::engine_ext h;
// cudnnNormalizationForwardInference-NEXT:   h.create_engine();
// cudnnNormalizationForwardInference-NEXT:   h.async_batch_normalization_forward_inference(m, op, adesc, eps, *alpha, src_d, src, *beta, dst_d, dst, summand_d, summand, p1_d, scale, bias, p2_d, emean, evar);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnNormalizationForwardTraining | FileCheck %s -check-prefix=cudnnNormalizationForwardTraining
// cudnnNormalizationForwardTraining: CUDA API:
// cudnnNormalizationForwardTraining-NEXT:   cudnnHandle_t h;
// cudnnNormalizationForwardTraining-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnNormalizationForwardTraining-NEXT:   cudnnNormalizationForwardTraining(
// cudnnNormalizationForwardTraining-NEXT:       h /*cudnnHandle_t*/, m /*cudnnNormMode_t*/, op /*cudnnNormOps_t*/,
// cudnnNormalizationForwardTraining-NEXT:       alg /*cudnnNormAlgo_t*/, alpha /*void **/, beta /*void **/,
// cudnnNormalizationForwardTraining-NEXT:       src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnNormalizationForwardTraining-NEXT:       p1_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
// cudnnNormalizationForwardTraining-NEXT:       factor /*double*/, p2_d /*cudnnTensorDescriptor_t*/, rmean /*void **/,
// cudnnNormalizationForwardTraining-NEXT:       rvar /*void **/, eps /*double*/, smean /*void **/, svar /*void **/,
// cudnnNormalizationForwardTraining-NEXT:       adesc /*cudnnActivationDescriptor_t*/,
// cudnnNormalizationForwardTraining-NEXT:       summand_d /*cudnnTensorDescriptor_t*/, summand /*void **/,
// cudnnNormalizationForwardTraining-NEXT:       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/, workspace /*void **/,
// cudnnNormalizationForwardTraining-NEXT:       workspace_size /*size_t*/, reservespace /*void **/,
// cudnnNormalizationForwardTraining-NEXT:       reservespace_size /*size_t*/, group_count /*int*/);
// cudnnNormalizationForwardTraining-NEXT: Is migrated to:
// cudnnNormalizationForwardTraining-NEXT:   dpct::dnnl::engine_ext h;
// cudnnNormalizationForwardTraining-NEXT:   h.create_engine();
// cudnnNormalizationForwardTraining-NEXT:   h.async_batch_normalization_forward_training(m, op, adesc, eps, factor, *alpha, src_d, src, *beta, dst_d, dst, summand_d, summand, p1_d, scale, bias, p2_d, rmean, rvar, smean, svar, reservespace_size, reservespace);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnOpTensor | FileCheck %s -check-prefix=cudnnOpTensor
// cudnnOpTensor: CUDA API:
// cudnnOpTensor-NEXT:   cudnnHandle_t h;
// cudnnOpTensor-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnOpTensor-NEXT:   cudnnOpTensor(
// cudnnOpTensor-NEXT:       h /*cudnnHandle_t*/, d /*cudnnOpTensorDescriptor_t*/, alpha1 /*void **/,
// cudnnOpTensor-NEXT:       src1_d /*cudnnTensorDescriptor_t*/, src1 /*void **/, alpha2 /*void **/,
// cudnnOpTensor-NEXT:       src2_d /*cudnnTensorDescriptor_t*/, src2 /*void **/, beta /*void **/,
// cudnnOpTensor-NEXT:       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
// cudnnOpTensor-NEXT: Is migrated to:
// cudnnOpTensor-NEXT:   dpct::dnnl::engine_ext h;
// cudnnOpTensor-NEXT:   h.create_engine();
// cudnnOpTensor-NEXT:   h.async_binary(d, *alpha1, src1_d, src1, *alpha2, src2_d, src2, *beta, dst_d, dst);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnPoolingBackward | FileCheck %s -check-prefix=cudnnPoolingBackward
// cudnnPoolingBackward: CUDA API:
// cudnnPoolingBackward-NEXT:   cudnnHandle_t h;
// cudnnPoolingBackward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnPoolingBackward-NEXT:   cudnnPoolingBackward(
// cudnnPoolingBackward-NEXT:       h /*cudnnHandle_t*/, desc /*cudnnPoolingDescriptor_t*/, alpha /*void **/,
// cudnnPoolingBackward-NEXT:       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
// cudnnPoolingBackward-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
// cudnnPoolingBackward-NEXT:       src_d /*cudnnTensorDescriptor_t*/, src /*void **/, beta /*void **/,
// cudnnPoolingBackward-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/);
// cudnnPoolingBackward-NEXT: Is migrated to:
// cudnnPoolingBackward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnPoolingBackward-NEXT:   h.create_engine();
// cudnnPoolingBackward-NEXT:   h.async_pooling_backward(desc, *alpha, dst_d, dst, diff_dst_d, diff_dst, src_d, src, *beta, diff_src_d, diff_src);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnPoolingForward | FileCheck %s -check-prefix=cudnnPoolingForward
// cudnnPoolingForward: CUDA API:
// cudnnPoolingForward-NEXT:   cudnnHandle_t h;
// cudnnPoolingForward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnPoolingForward-NEXT:   cudnnPoolingForward(h /*cudnnHandle_t*/, desc /*cudnnLRNDescriptor_t*/,
// cudnnPoolingForward-NEXT:                       alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnPoolingForward-NEXT:                       src /*void **/, beta /*void **/,
// cudnnPoolingForward-NEXT:                       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
// cudnnPoolingForward-NEXT: Is migrated to:
// cudnnPoolingForward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnPoolingForward-NEXT:   h.create_engine();
// cudnnPoolingForward-NEXT:   h.async_pooling_forward(desc, *alpha, src_d, src, *beta, dst_d, dst);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnReduceTensor | FileCheck %s -check-prefix=cudnnReduceTensor
// cudnnReduceTensor: CUDA API:
// cudnnReduceTensor-NEXT:   cudnnHandle_t h;
// cudnnReduceTensor-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnReduceTensor-NEXT:   cudnnReduceTensor(h /*cudnnHandle_t*/, d /*cudnnReduceTensorDescriptor_t*/,
// cudnnReduceTensor-NEXT:                     i /*void**/, is /*size_t*/, w /*void**/, ws /*size_t*/,
// cudnnReduceTensor-NEXT:                     alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnReduceTensor-NEXT:                     src /*void **/, beta /*void **/,
// cudnnReduceTensor-NEXT:                     dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
// cudnnReduceTensor-NEXT: Is migrated to:
// cudnnReduceTensor-NEXT:   dpct::dnnl::engine_ext h;
// cudnnReduceTensor-NEXT:   h.create_engine();
// cudnnReduceTensor-NEXT:   h.async_reduction(d, *alpha, src_d, src, *beta, dst_d, dst);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnRestoreDropoutDescriptor | FileCheck %s -check-prefix=cudnnRestoreDropoutDescriptor
// cudnnRestoreDropoutDescriptor: CUDA API:
// cudnnRestoreDropoutDescriptor-NEXT:   cudnnDropoutDescriptor_t d;
// cudnnRestoreDropoutDescriptor-NEXT:   cudnnRestoreDropoutDescriptor(
// cudnnRestoreDropoutDescriptor-NEXT:       d /*cudnnDropoutDescriptor_t*/, h /*cudnnHandle_t*/, dropout /*float*/,
// cudnnRestoreDropoutDescriptor-NEXT:       states /*void **/, statesize /*size_t*/, seed /*unsigned long long*/);
// cudnnRestoreDropoutDescriptor-NEXT: Is migrated to:
// cudnnRestoreDropoutDescriptor-NEXT:   dpct::dnnl::dropout_desc d;
// cudnnRestoreDropoutDescriptor-NEXT:   d.restore(h, dropout, states, statesize, seed);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnRNNBackwardData_v8 | FileCheck %s -check-prefix=cudnnRNNBackwardData_v8
// cudnnRNNBackwardData_v8: CUDA API:
// cudnnRNNBackwardData_v8-NEXT:   cudnnHandle_t h;
// cudnnRNNBackwardData_v8-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnRNNBackwardData_v8-NEXT:   cudnnRNNBackwardData_v8(
// cudnnRNNBackwardData_v8-NEXT:       h /*cudnnHandle_t*/, d /*cudnnRNNDescriptor_t*/, sa /*int32_t []*/,
// cudnnRNNBackwardData_v8-NEXT:       dst_d /*cudnnRNNDataDescriptor_t*/, dst /*void **/, diff_dst /*void **/,
// cudnnRNNBackwardData_v8-NEXT:       src_d /*cudnnRNNDataDescriptor_t*/, diff_src /*void **/,
// cudnnRNNBackwardData_v8-NEXT:       h_d /*cudnnTensorDescriptor_t*/, hx /*void **/, diff_hy /*void **/,
// cudnnRNNBackwardData_v8-NEXT:       diff_hx /*void **/, c_d /*cudnnTensorDescriptor_t*/, cx /*void **/,
// cudnnRNNBackwardData_v8-NEXT:       diff_cy /*void **/, diff_cx /*void **/, weightspace_size /*size_t*/,
// cudnnRNNBackwardData_v8-NEXT:       weightspace /*void **/, workspace_size /*size_t*/, workspace /*void **/,
// cudnnRNNBackwardData_v8-NEXT:       reservespace_size /*size_t*/, reservespace /*void **/);
// cudnnRNNBackwardData_v8-NEXT:   cudnnRNNBackwardWeights_v8(
// cudnnRNNBackwardData_v8-NEXT:       h /*cudnnHandle_t*/, d /*cudnnRNNDescriptor_t*/, wm /*cudnnWgradMode_t*/,
// cudnnRNNBackwardData_v8-NEXT:       sa /*int32_t []*/, src_d /*cudnnRNNDataDescriptor_t*/, src /*void **/,
// cudnnRNNBackwardData_v8-NEXT:       h_d /*cudnnTensorDescriptor_t*/, hx /*void **/,
// cudnnRNNBackwardData_v8-NEXT:       dst_d /*cudnnRNNDataDescriptor_t*/, dst /*void **/,
// cudnnRNNBackwardData_v8-NEXT:       weightspace_size /*size_t*/, diff_weightspace /*void **/,
// cudnnRNNBackwardData_v8-NEXT:       workspace_size /*size_t*/, workspace /*void **/,
// cudnnRNNBackwardData_v8-NEXT:       reservespace_size /*size_t*/, reservespace /*void **/);
// cudnnRNNBackwardData_v8-NEXT: Is migrated to:
// cudnnRNNBackwardData_v8-NEXT:   dpct::dnnl::engine_ext h;
// cudnnRNNBackwardData_v8-NEXT:   h.create_engine();
// cudnnRNNBackwardData_v8-NEXT:   h.async_rnn_backward(d, dst_d, dst, diff_dst, src_d, src, diff_src, h_d, hx, diff_hy, diff_hx, c_d, cx, diff_cy, diff_cx, weightspace_size, weightspace, diff_weightspace, workspace_size, workspace, reservespace_size, reservespace);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnRNNBackwardWeights_v8 | FileCheck %s -check-prefix=cudnnRNNBackwardWeights_v8
// cudnnRNNBackwardWeights_v8: CUDA API:
// cudnnRNNBackwardWeights_v8-NEXT:   cudnnHandle_t h;
// cudnnRNNBackwardWeights_v8-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnRNNBackwardWeights_v8-NEXT:   cudnnRNNBackwardData_v8(
// cudnnRNNBackwardWeights_v8-NEXT:       h /*cudnnHandle_t*/, d /*cudnnRNNDescriptor_t*/, sa /*int32_t []*/,
// cudnnRNNBackwardWeights_v8-NEXT:       dst_d /*cudnnRNNDataDescriptor_t*/, dst /*void **/, diff_dst /*void **/,
// cudnnRNNBackwardWeights_v8-NEXT:       src_d /*cudnnRNNDataDescriptor_t*/, diff_src /*void **/,
// cudnnRNNBackwardWeights_v8-NEXT:       h_d /*cudnnTensorDescriptor_t*/, hx /*void **/, diff_hy /*void **/,
// cudnnRNNBackwardWeights_v8-NEXT:       diff_hx /*void **/, c_d /*cudnnTensorDescriptor_t*/, cx /*void **/,
// cudnnRNNBackwardWeights_v8-NEXT:       diff_cy /*void **/, diff_cx /*void **/, weightspace_size /*size_t*/,
// cudnnRNNBackwardWeights_v8-NEXT:       weightspace /*void **/, workspace_size /*size_t*/, workspace /*void **/,
// cudnnRNNBackwardWeights_v8-NEXT:       reservespace_size /*size_t*/, reservespace /*void **/);
// cudnnRNNBackwardWeights_v8-NEXT:   cudnnRNNBackwardWeights_v8(
// cudnnRNNBackwardWeights_v8-NEXT:       h /*cudnnHandle_t*/, d /*cudnnRNNDescriptor_t*/, wm /*cudnnWgradMode_t*/,
// cudnnRNNBackwardWeights_v8-NEXT:       sa /*int32_t []*/, src_d /*cudnnRNNDataDescriptor_t*/, src /*void **/,
// cudnnRNNBackwardWeights_v8-NEXT:       h_d /*cudnnTensorDescriptor_t*/, hx /*void **/,
// cudnnRNNBackwardWeights_v8-NEXT:       dst_d /*cudnnRNNDataDescriptor_t*/, dst /*void **/,
// cudnnRNNBackwardWeights_v8-NEXT:       weightspace_size /*size_t*/, diff_weightspace /*void **/,
// cudnnRNNBackwardWeights_v8-NEXT:       workspace_size /*size_t*/, workspace /*void **/,
// cudnnRNNBackwardWeights_v8-NEXT:       reservespace_size /*size_t*/, reservespace /*void **/);
// cudnnRNNBackwardWeights_v8-NEXT: Is migrated to:
// cudnnRNNBackwardWeights_v8-NEXT:   dpct::dnnl::engine_ext h;
// cudnnRNNBackwardWeights_v8-NEXT:   h.create_engine();
// cudnnRNNBackwardWeights_v8-NEXT:   h.async_rnn_backward(d, dst_d, dst, diff_dst, src_d, src, diff_src, h_d, hx, diff_hy, diff_hx, c_d, cx, diff_cy, diff_cx, weightspace_size, weightspace, diff_weightspace, workspace_size, workspace, reservespace_size, reservespace);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnRNNForward | FileCheck %s -check-prefix=cudnnRNNForward
// cudnnRNNForward: CUDA API:
// cudnnRNNForward-NEXT:   cudnnHandle_t h;
// cudnnRNNForward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnRNNForward-NEXT:   cudnnRNNForward(h /*cudnnHandle_t*/, d /*cudnnRNNDescriptor_t*/,
// cudnnRNNForward-NEXT:                   m /*cudnnForwardMode_t*/, sa /*int32_t []*/,
// cudnnRNNForward-NEXT:                   src_d /*cudnnRNNDataDescriptor_t*/, src /*void **/,
// cudnnRNNForward-NEXT:                   dst_d /*cudnnRNNDataDescriptor_t*/, dst /*void **/,
// cudnnRNNForward-NEXT:                   h_d /*cudnnTensorDescriptor_t*/, hx /*void **/, hy /*void **/,
// cudnnRNNForward-NEXT:                   c_d /*cudnnTensorDescriptor_t*/, cx /*void **/, cy /*void **/,
// cudnnRNNForward-NEXT:                   weightspace_size /*size_t*/, weightspace /*void **/,
// cudnnRNNForward-NEXT:                   workspace_size /*size_t*/, workspace /*void **/,
// cudnnRNNForward-NEXT:                   reservespace_size /*size_t*/, reservespace /*void **/);
// cudnnRNNForward-NEXT: Is migrated to:
// cudnnRNNForward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnRNNForward-NEXT:   h.create_engine();
// cudnnRNNForward-NEXT:   h.async_rnn_forward(d, m, src_d, src, dst_d, dst, h_d, hx, hy, c_d, cx, cy, weightspace_size, weightspace, workspace_size, workspace, reservespace_size, reservespace);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnScaleTensor | FileCheck %s -check-prefix=cudnnScaleTensor
// cudnnScaleTensor: CUDA API:
// cudnnScaleTensor-NEXT:   cudnnHandle_t h;
// cudnnScaleTensor-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnScaleTensor-NEXT:   cudnnScaleTensor(h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnScaleTensor-NEXT:                    src /*void **/, factor /*void **/);
// cudnnScaleTensor-NEXT: Is migrated to:
// cudnnScaleTensor-NEXT:   dpct::dnnl::engine_ext h;
// cudnnScaleTensor-NEXT:   h.create_engine();
// cudnnScaleTensor-NEXT:   h.async_scale(*factor, src_d, src);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetActivationDescriptorSwishBeta | FileCheck %s -check-prefix=cudnnSetActivationDescriptorSwishBeta
// cudnnSetActivationDescriptorSwishBeta: CUDA API:
// cudnnSetActivationDescriptorSwishBeta-NEXT:   cudnnActivationDescriptor_t d;
// cudnnSetActivationDescriptorSwishBeta-NEXT:   cudnnSetActivationDescriptorSwishBeta(d /*cudnnActivationDescriptor_t*/,
// cudnnSetActivationDescriptorSwishBeta-NEXT:                                         s /*double*/);
// cudnnSetActivationDescriptorSwishBeta-NEXT: Is migrated to:
// cudnnSetActivationDescriptorSwishBeta-NEXT:   dpct::dnnl::activation_desc d;
// cudnnSetActivationDescriptorSwishBeta-NEXT:   d.set_beta(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetConvolution2dDescriptor | FileCheck %s -check-prefix=cudnnSetConvolution2dDescriptor
// cudnnSetConvolution2dDescriptor: CUDA API:
// cudnnSetConvolution2dDescriptor-NEXT:   cudnnConvolutionDescriptor_t d;
// cudnnSetConvolution2dDescriptor-NEXT:   cudnnSetConvolution2dDescriptor(
// cudnnSetConvolution2dDescriptor-NEXT:       d /*cudnnConvolutionDescriptor_t*/, padding_h /*int*/, padding_w /*int*/,
// cudnnSetConvolution2dDescriptor-NEXT:       stride_h /*int*/, stride_w /*int*/, dilation_h /*int*/,
// cudnnSetConvolution2dDescriptor-NEXT:       dilation_w /*int*/, m /*cudnnConvolutionMode_t*/, t /*cudnnDataType_t*/);
// cudnnSetConvolution2dDescriptor-NEXT: Is migrated to:
// cudnnSetConvolution2dDescriptor-NEXT:   dpct::dnnl::convolution_desc d;
// cudnnSetConvolution2dDescriptor-NEXT:   d.set(padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetConvolutionGroupCount | FileCheck %s -check-prefix=cudnnSetConvolutionGroupCount
// cudnnSetConvolutionGroupCount: CUDA API:
// cudnnSetConvolutionGroupCount-NEXT:   cudnnConvolutionDescriptor_t d;
// cudnnSetConvolutionGroupCount-NEXT:   cudnnSetConvolutionGroupCount(d /*cudnnActivationDescriptor_t*/,
// cudnnSetConvolutionGroupCount-NEXT:                                 group_count /*int*/);
// cudnnSetConvolutionGroupCount-NEXT: Is migrated to:
// cudnnSetConvolutionGroupCount-NEXT:   dpct::dnnl::convolution_desc d;
// cudnnSetConvolutionGroupCount-NEXT:   d.set_group_count(group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetConvolutionMathType | FileCheck %s -check-prefix=cudnnSetConvolutionMathType
// cudnnSetConvolutionMathType: CUDA API:
// cudnnSetConvolutionMathType-NEXT:   cudnnConvolutionDescriptor_t d;
// cudnnSetConvolutionMathType-NEXT:   cudnnSetConvolutionMathType(d /*cudnnActivationDescriptor_t*/,
// cudnnSetConvolutionMathType-NEXT:                               mt /*cudnnMathType_t*/);
// cudnnSetConvolutionMathType-NEXT: Is migrated to:
// cudnnSetConvolutionMathType-NEXT:   dpct::dnnl::convolution_desc d;
// cudnnSetConvolutionMathType-NEXT:   d.set_math_mode(mt);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetConvolutionNdDescriptor | FileCheck %s -check-prefix=cudnnSetConvolutionNdDescriptor
// cudnnSetConvolutionNdDescriptor: CUDA API:
// cudnnSetConvolutionNdDescriptor-NEXT:   cudnnConvolutionDescriptor_t d;
// cudnnSetConvolutionNdDescriptor-NEXT:   cudnnSetConvolutionNdDescriptor(
// cudnnSetConvolutionNdDescriptor-NEXT:       d /*cudnnConvolutionDescriptor_t*/, n /*int*/, paddinga /*int[]*/,
// cudnnSetConvolutionNdDescriptor-NEXT:       stridea /*int[]*/, dilationa /*int[]*/, m /*cudnnConvolutionMode_t*/,
// cudnnSetConvolutionNdDescriptor-NEXT:       t /*cudnnDataType_t*/);
// cudnnSetConvolutionNdDescriptor-NEXT: Is migrated to:
// cudnnSetConvolutionNdDescriptor-NEXT:   dpct::dnnl::convolution_desc d;
// cudnnSetConvolutionNdDescriptor-NEXT:   d.set(n, paddinga, stridea, dilationa);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetDropoutDescriptor | FileCheck %s -check-prefix=cudnnSetDropoutDescriptor
// cudnnSetDropoutDescriptor: CUDA API:
// cudnnSetDropoutDescriptor-NEXT:   cudnnDropoutDescriptor_t d;
// cudnnSetDropoutDescriptor-NEXT:   cudnnSetDropoutDescriptor(d /*cudnnDropoutDescriptor_t*/, h /*cudnnHandle_t*/,
// cudnnSetDropoutDescriptor-NEXT:                             dropout /*float*/, states /*void **/,
// cudnnSetDropoutDescriptor-NEXT:                             statesize /*size_t*/, seed /*unsigned long long*/);
// cudnnSetDropoutDescriptor-NEXT: Is migrated to:
// cudnnSetDropoutDescriptor-NEXT:   dpct::dnnl::dropout_desc d;
// cudnnSetDropoutDescriptor-NEXT:   d.set(h, dropout, states, statesize, seed);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetFilter4dDescriptor | FileCheck %s -check-prefix=cudnnSetFilter4dDescriptor
// cudnnSetFilter4dDescriptor: CUDA API:
// cudnnSetFilter4dDescriptor-NEXT:   cudnnFilterDescriptor_t d;
// cudnnSetFilter4dDescriptor-NEXT:   cudnnSetFilter4dDescriptor(d /*cudnnFilterDescriptor_t*/,
// cudnnSetFilter4dDescriptor-NEXT:                              t /*cudnnDataType_t*/, f /*cudnnTensorFormat_t*/,
// cudnnSetFilter4dDescriptor-NEXT:                              k /*int*/, c /*int*/, h /*int*/, w /*int*/);
// cudnnSetFilter4dDescriptor-NEXT: Is migrated to:
// cudnnSetFilter4dDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnSetFilter4dDescriptor-NEXT:   d.set(f, t, k, c, h, w);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetFilterNdDescriptor | FileCheck %s -check-prefix=cudnnSetFilterNdDescriptor
// cudnnSetFilterNdDescriptor: CUDA API:
// cudnnSetFilterNdDescriptor-NEXT:   cudnnFilterDescriptor_t d;
// cudnnSetFilterNdDescriptor-NEXT:   cudnnSetFilterNdDescriptor(d /*cudnnFilterDescriptor_t*/,
// cudnnSetFilterNdDescriptor-NEXT:                              t /*cudnnDataType_t*/, f /*cudnnTensorFormat_t*/,
// cudnnSetFilterNdDescriptor-NEXT:                              n /*int*/, da /*int[]*/);
// cudnnSetFilterNdDescriptor-NEXT: Is migrated to:
// cudnnSetFilterNdDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnSetFilterNdDescriptor-NEXT:   d.set(f, t, n, da);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetLRNDescriptor | FileCheck %s -check-prefix=cudnnSetLRNDescriptor
// cudnnSetLRNDescriptor: CUDA API:
// cudnnSetLRNDescriptor-NEXT:   cudnnLRNDescriptor_t d;
// cudnnSetLRNDescriptor-NEXT:   cudnnSetLRNDescriptor(d /*cudnnLRNDescriptor_t*/, n /*unsigned*/,
// cudnnSetLRNDescriptor-NEXT:                         alpha /*double*/, beta /*double*/, k /*double*/);
// cudnnSetLRNDescriptor-NEXT: Is migrated to:
// cudnnSetLRNDescriptor-NEXT:   dpct::dnnl::lrn_desc d;
// cudnnSetLRNDescriptor-NEXT:   d.set(n, alpha, beta, k);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetOpTensorDescriptor | FileCheck %s -check-prefix=cudnnSetOpTensorDescriptor
// cudnnSetOpTensorDescriptor: CUDA API:
// cudnnSetOpTensorDescriptor-NEXT:   cudnnOpTensorDescriptor_t d;
// cudnnSetOpTensorDescriptor-NEXT:   cudnnSetOpTensorDescriptor(d /*cudnnOpTensorDescriptor_t*/,
// cudnnSetOpTensorDescriptor-NEXT:                              op /*cudnnOpTensorOp_t*/, dt /*cudnnDataType_t*/,
// cudnnSetOpTensorDescriptor-NEXT:                              p /*cudnnNanPropagation_t*/);
// cudnnSetOpTensorDescriptor-NEXT: Is migrated to:
// cudnnSetOpTensorDescriptor-NEXT:   dpct::dnnl::binary_op d;
// cudnnSetOpTensorDescriptor-NEXT:   d = op;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetPooling2dDescriptor | FileCheck %s -check-prefix=cudnnSetPooling2dDescriptor
// cudnnSetPooling2dDescriptor: CUDA API:
// cudnnSetPooling2dDescriptor-NEXT:   cudnnPoolingDescriptor_t d;
// cudnnSetPooling2dDescriptor-NEXT:   cudnnSetPooling2dDescriptor(d /*cudnnPoolingDescriptor_t*/,
// cudnnSetPooling2dDescriptor-NEXT:                               m /*cudnnPoolingMode_t*/,
// cudnnSetPooling2dDescriptor-NEXT:                               p /*cudnnNanPropagation_t*/, h /*int*/, w /*int*/,
// cudnnSetPooling2dDescriptor-NEXT:                               vp /*int*/, hp /*int*/, vs /*int*/, hs /*int*/);
// cudnnSetPooling2dDescriptor-NEXT: Is migrated to:
// cudnnSetPooling2dDescriptor-NEXT:   dpct::dnnl::pooling_desc d;
// cudnnSetPooling2dDescriptor-NEXT:   d.set(m, h, w, vp, hp, vs, hs);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetPoolingNdDescriptor | FileCheck %s -check-prefix=cudnnSetPoolingNdDescriptor
// cudnnSetPoolingNdDescriptor: CUDA API:
// cudnnSetPoolingNdDescriptor-NEXT:   cudnnPoolingDescriptor_t d;
// cudnnSetPoolingNdDescriptor-NEXT:   cudnnSetPoolingNdDescriptor(d /*cudnnPoolingDescriptor_t*/,
// cudnnSetPoolingNdDescriptor-NEXT:                               m /*cudnnPoolingMode_t*/,
// cudnnSetPoolingNdDescriptor-NEXT:                               p /*cudnnNanPropagation_t*/, nd /*int*/,
// cudnnSetPoolingNdDescriptor-NEXT:                               da /*int[]*/, pa /*int[]*/, sa /*int[]*/);
// cudnnSetPoolingNdDescriptor-NEXT: Is migrated to:
// cudnnSetPoolingNdDescriptor-NEXT:   dpct::dnnl::pooling_desc d;
// cudnnSetPoolingNdDescriptor-NEXT:   d.set(m, nd, da, pa, sa);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetReduceTensorDescriptor | FileCheck %s -check-prefix=cudnnSetReduceTensorDescriptor
// cudnnSetReduceTensorDescriptor: CUDA API:
// cudnnSetReduceTensorDescriptor-NEXT:   cudnnReduceTensorDescriptor_t d;
// cudnnSetReduceTensorDescriptor-NEXT:   cudnnSetReduceTensorDescriptor(
// cudnnSetReduceTensorDescriptor-NEXT:       d /*cudnnReduceTensorDescriptor_t */, o /*cudnnPoolingMode_t*/,
// cudnnSetReduceTensorDescriptor-NEXT:       dt /*cudnnDataType_t*/, p /*cudnnNanPropagation_t*/,
// cudnnSetReduceTensorDescriptor-NEXT:       i /*cudnnReduceTensorIndices_t*/, it /*cudnnIndicesType_t*/);
// cudnnSetReduceTensorDescriptor-NEXT: Is migrated to:
// cudnnSetReduceTensorDescriptor-NEXT:   dpct::dnnl::reduction_op d;
// cudnnSetReduceTensorDescriptor-NEXT:   d = o;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetRNNDataDescriptor | FileCheck %s -check-prefix=cudnnSetRNNDataDescriptor
// cudnnSetRNNDataDescriptor: CUDA API:
// cudnnSetRNNDataDescriptor-NEXT:   cudnnRNNDataDescriptor_t d;
// cudnnSetRNNDataDescriptor-NEXT:   cudnnSetRNNDataDescriptor(d /*cudnnTensorDescriptor_t*/,
// cudnnSetRNNDataDescriptor-NEXT:                             t /*cudnnDataType_t*/, l /*cudnnRNNDataLayout_t*/,
// cudnnSetRNNDataDescriptor-NEXT:                             len /*int*/, b /*int*/, v /*int*/, sa /*int[]*/,
// cudnnSetRNNDataDescriptor-NEXT:                             p /*void **/);
// cudnnSetRNNDataDescriptor-NEXT: Is migrated to:
// cudnnSetRNNDataDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnSetRNNDataDescriptor-NEXT:   d.set(l, t, len, b, v);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetRNNDescriptor_v8 | FileCheck %s -check-prefix=cudnnSetRNNDescriptor_v8
// cudnnSetRNNDescriptor_v8: CUDA API:
// cudnnSetRNNDescriptor_v8-NEXT:   cudnnRNNDescriptor_t d;
// cudnnSetRNNDescriptor_v8-NEXT:   cudnnSetRNNDescriptor_v8(
// cudnnSetRNNDescriptor_v8-NEXT:       d /*cudnnRNNDescriptor_t*/, alg /*cudnnRNNAlgo_t*/, m /*cudnnRNNMode_t*/,
// cudnnSetRNNDescriptor_v8-NEXT:       bm /*cudnnRNNBiasMode_t*/, dm /*cudnnDirectionMode_t*/,
// cudnnSetRNNDescriptor_v8-NEXT:       im /*cudnnRNNInputMode_t*/, t /*cudnnDataType_t*/, mp /*cudnnDataType_t*/,
// cudnnSetRNNDescriptor_v8-NEXT:       mt /*cudnnMathType_t*/, is /*int32_t*/, hs /*int32_t*/, ps /*int32_t*/,
// cudnnSetRNNDescriptor_v8-NEXT:       l /*int32_t[]*/, dropout /*cudnnDropoutDescriptor_t*/, f /*uint32_t*/);
// cudnnSetRNNDescriptor_v8-NEXT: Is migrated to:
// cudnnSetRNNDescriptor_v8-NEXT:   dpct::dnnl::rnn_desc d;
// cudnnSetRNNDescriptor_v8-NEXT:   d.set(m, bm, dm, t, is, hs, ps, l);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetTensor | FileCheck %s -check-prefix=cudnnSetTensor
// cudnnSetTensor: CUDA API:
// cudnnSetTensor-NEXT:   cudnnHandle_t h;
// cudnnSetTensor-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnSetTensor-NEXT:   cudnnSetTensor(h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnSetTensor-NEXT:                  src /*void **/, value /*void **/);
// cudnnSetTensor-NEXT: Is migrated to:
// cudnnSetTensor-NEXT:   dpct::dnnl::engine_ext h;
// cudnnSetTensor-NEXT:   h.create_engine();
// cudnnSetTensor-NEXT:   h.async_fill(src_d, src, value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetTensor4dDescriptorEx | FileCheck %s -check-prefix=cudnnSetTensor4dDescriptorEx
// cudnnSetTensor4dDescriptorEx: CUDA API:
// cudnnSetTensor4dDescriptorEx-NEXT:   cudnnTensorDescriptor_t d;
// cudnnSetTensor4dDescriptorEx-NEXT:   cudnnSetTensor4dDescriptorEx(d /*cudnnTensorDescriptor_t*/,
// cudnnSetTensor4dDescriptorEx-NEXT:                                t /*cudnnDataType_t*/, n /*int*/, c /*int*/,
// cudnnSetTensor4dDescriptorEx-NEXT:                                h /*int*/, w /*int*/, ns /*int*/, cs /*int*/,
// cudnnSetTensor4dDescriptorEx-NEXT:                                hs /*int*/, ws /*int*/);
// cudnnSetTensor4dDescriptorEx-NEXT: Is migrated to:
// cudnnSetTensor4dDescriptorEx-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnSetTensor4dDescriptorEx-NEXT:   d.set(t, n, c, h, w, ns, cs, hs, ws);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetTensorNdDescriptor | FileCheck %s -check-prefix=cudnnSetTensorNdDescriptor
// cudnnSetTensorNdDescriptor: CUDA API:
// cudnnSetTensorNdDescriptor-NEXT:   cudnnTensorDescriptor_t d;
// cudnnSetTensorNdDescriptor-NEXT:   cudnnSetTensorNdDescriptor(d /*cudnnTensorDescriptor_t*/,
// cudnnSetTensorNdDescriptor-NEXT:                              t /*cudnnDataType_t*/, nd /*int*/, da /*int[]*/,
// cudnnSetTensorNdDescriptor-NEXT:                              sa /*int[]*/);
// cudnnSetTensorNdDescriptor-NEXT: Is migrated to:
// cudnnSetTensorNdDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnSetTensorNdDescriptor-NEXT:   d.set(t, nd, da, sa);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetTensorNdDescriptorEx | FileCheck %s -check-prefix=cudnnSetTensorNdDescriptorEx
// cudnnSetTensorNdDescriptorEx: CUDA API:
// cudnnSetTensorNdDescriptorEx-NEXT:   cudnnTensorDescriptor_t d;
// cudnnSetTensorNdDescriptorEx-NEXT:   cudnnSetTensorNdDescriptorEx(d /*cudnnTensorDescriptor_t*/,
// cudnnSetTensorNdDescriptorEx-NEXT:                                f /*cudnnTensorFormat_t*/, t /*cudnnDataType_t*/,
// cudnnSetTensorNdDescriptorEx-NEXT:                                nd /*int*/, da /*int[]*/);
// cudnnSetTensorNdDescriptorEx-NEXT: Is migrated to:
// cudnnSetTensorNdDescriptorEx-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnSetTensorNdDescriptorEx-NEXT:   d.set(f, t, nd, da);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSoftmaxBackward | FileCheck %s -check-prefix=cudnnSoftmaxBackward
// cudnnSoftmaxBackward: CUDA API:
// cudnnSoftmaxBackward-NEXT:   cudnnHandle_t h;
// cudnnSoftmaxBackward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnSoftmaxBackward-NEXT:   cudnnSoftmaxBackward(h /*cudnnHandle_t*/, a /*cudnnSoftmaxAlgorithm_t*/,
// cudnnSoftmaxBackward-NEXT:                        m /*cudnnSoftmaxMode_t*/, alpha /*void **/,
// cudnnSoftmaxBackward-NEXT:                        dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
// cudnnSoftmaxBackward-NEXT:                        diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnSoftmaxBackward-NEXT:                        diff_dst /*void **/, beta /*void **/,
// cudnnSoftmaxBackward-NEXT:                        diff_src_d /*cudnnTensorDescriptor_t*/,
// cudnnSoftmaxBackward-NEXT:                        diff_src /*void **/);
// cudnnSoftmaxBackward-NEXT: Is migrated to:
// cudnnSoftmaxBackward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnSoftmaxBackward-NEXT:   h.create_engine();
// cudnnSoftmaxBackward-NEXT:   h.async_softmax_backward(a, m, *alpha, dst_d, dst, diff_dst_d, diff_dst, *beta, diff_src_d, diff_src);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSoftmaxForward | FileCheck %s -check-prefix=cudnnSoftmaxForward
// cudnnSoftmaxForward: CUDA API:
// cudnnSoftmaxForward-NEXT:   cudnnHandle_t h;
// cudnnSoftmaxForward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnSoftmaxForward-NEXT:   cudnnSoftmaxForward(h /*cudnnHandle_t*/, a /*cudnnSoftmaxAlgorithm_t*/,
// cudnnSoftmaxForward-NEXT:                       m /*cudnnSoftmaxMode_t*/, alpha /*void **/,
// cudnnSoftmaxForward-NEXT:                       src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnSoftmaxForward-NEXT:                       beta /*void **/, dst_d /*cudnnTensorDescriptor_t*/,
// cudnnSoftmaxForward-NEXT:                       dst /*void **/);
// cudnnSoftmaxForward-NEXT: Is migrated to:
// cudnnSoftmaxForward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnSoftmaxForward-NEXT:   h.create_engine();
// cudnnSoftmaxForward-NEXT:   h.async_softmax_forward(a, m, *alpha, src_d, src, *beta, dst_d, dst);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnTransformTensor | FileCheck %s -check-prefix=cudnnTransformTensor
// cudnnTransformTensor: CUDA API:
// cudnnTransformTensor-NEXT:   cudnnHandle_t h;
// cudnnTransformTensor-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnTransformTensor-NEXT:   cudnnTransformTensor(h /*cudnnHandle_t*/, alpha /*void **/,
// cudnnTransformTensor-NEXT:                        src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnTransformTensor-NEXT:                        beta /*void **/, dst_d /*cudnnTensorDescriptor_t*/,
// cudnnTransformTensor-NEXT:                        dst /*void **/);
// cudnnTransformTensor-NEXT: Is migrated to:
// cudnnTransformTensor-NEXT:   dpct::dnnl::engine_ext h;
// cudnnTransformTensor-NEXT:   h.create_engine();
// cudnnTransformTensor-NEXT:   h.async_reorder(*alpha, src_d, src, *beta, dst_d, dst);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnFindConvolutionBackwardDataAlgorithm | FileCheck %s -check-prefix=cudnnFindConvolutionBackwardDataAlgorithm
// cudnnFindConvolutionBackwardDataAlgorithm: CUDA API:
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:   cudnnHandle_t h;
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:   cudnnConvolutionBwdDataAlgoPerf_t r;
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:   cudnnFindConvolutionBackwardDataAlgorithm(
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:       h /*cudnnHandle_t*/, weight_d /*cudnnFilterDescriptor_t*/,
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/,
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/, reqc /*int*/, realc /*int **/,
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:       &r /*cudnnConvolutionBwdDataAlgoPerf_t*/);
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT: Is migrated to:
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:   dpct::dnnl::engine_ext h;
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:   h.create_engine();
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:   dpct::dnnl::convolution_algorithm_info r;
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:   r.algo = dnnl::algorithm::convolution_auto;
// cudnnFindConvolutionBackwardDataAlgorithm-NEXT:   *realc = 1;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnFindConvolutionBackwardFilterAlgorithm | FileCheck %s -check-prefix=cudnnFindConvolutionBackwardFilterAlgorithm
// cudnnFindConvolutionBackwardFilterAlgorithm: CUDA API:
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:   cudnnHandle_t h;
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:   cudnnConvolutionBwdFilterAlgoPerf_t r;
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:   cudnnFindConvolutionBackwardFilterAlgorithm(
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:       h /*cudnnHandle_t*/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/,
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:       diff_weight_d /*cudnnFilterDescriptor_t*/, reqc /*int*/, realc /*int **/,
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:       &r /*cudnnConvolutionBwdFilterAlgoPerf_t*/);
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT: Is migrated to:
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:   dpct::dnnl::engine_ext h;
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:   h.create_engine();
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:   dpct::dnnl::convolution_algorithm_info r;
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:   r.algo = dnnl::algorithm::convolution_auto;
// cudnnFindConvolutionBackwardFilterAlgorithm-NEXT:   *realc = 1;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionBackwardDataAlgorithm_v7 | FileCheck %s -check-prefix=cudnnGetConvolutionBackwardDataAlgorithm_v7
// cudnnGetConvolutionBackwardDataAlgorithm_v7: CUDA API:
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:   cudnnHandle_t h;
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:   cudnnConvolutionBwdDataAlgoPerf_t r;
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:   cudnnGetConvolutionBackwardDataAlgorithm_v7(
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:       h /*cudnnHandle_t*/, weight_d /*cudnnFilterDescriptor_t*/,
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/,
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/, reqc /*int*/, realc /*int **/,
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:       &r /*cudnnConvolutionBwdDataAlgoPerf_t*/);
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT: Is migrated to:
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:   h.create_engine();
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:   dpct::dnnl::convolution_algorithm_info r;
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:   r.algo = dnnl::algorithm::convolution_auto;
// cudnnGetConvolutionBackwardDataAlgorithm_v7-NEXT:   *realc = 1;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionBackwardFilterAlgorithm_v7 | FileCheck %s -check-prefix=cudnnGetConvolutionBackwardFilterAlgorithm_v7
// cudnnGetConvolutionBackwardFilterAlgorithm_v7: CUDA API:
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:   cudnnHandle_t h;
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:   cudnnConvolutionBwdFilterAlgoPerf_t r;
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:   cudnnGetConvolutionBackwardFilterAlgorithm_v7(
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:       h /*cudnnHandle_t*/, src_d /*cudnnFilterDescriptor_t*/,
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/,
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:       diff_weight_d /*cudnnFilterDescriptor_t*/, reqc /*int*/, realc /*int **/,
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:       &r /*cudnnConvolutionBwdFilterAlgoPerf_t*/);
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT: Is migrated to:
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:   h.create_engine();
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:   dpct::dnnl::convolution_algorithm_info r;
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:   r.algo = dnnl::algorithm::convolution_auto;
// cudnnGetConvolutionBackwardFilterAlgorithm_v7-NEXT:   *realc = 1;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetConvolutionForwardAlgorithm_v7 | FileCheck %s -check-prefix=cudnnGetConvolutionForwardAlgorithm_v7
// cudnnGetConvolutionForwardAlgorithm_v7: CUDA API:
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:   cudnnHandle_t h;
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:   cudnnConvolutionFwdAlgoPerf_t r;
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:   cudnnGetConvolutionForwardAlgorithm_v7(
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:       h /*cudnnHandle_t*/, src_d /*cudnnFilterDescriptor_t*/,
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:       weight_d /*cudnnTensorDescriptor_t*/,
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/, dst_d /*cudnnFilterDescriptor_t*/,
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:       reqc /*int*/, realc /*int **/, &r /*cudnnConvolutionFwdAlgoPerf_t*/);
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT: Is migrated to:
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:   dpct::dnnl::engine_ext h;
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:   h.create_engine();
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:   dpct::dnnl::convolution_algorithm_info r;
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:   r.algo = dnnl::algorithm::convolution_auto;
// cudnnGetConvolutionForwardAlgorithm_v7-NEXT:   *realc = 1;
