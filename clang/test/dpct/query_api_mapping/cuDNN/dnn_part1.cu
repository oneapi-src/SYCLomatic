// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreate | FileCheck %s -check-prefix=cudnnCreate
// cudnnCreate: CUDA API:
// cudnnCreate-NEXT: cudnnHandle_t h;
// cudnnCreate-NEXT: cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnCreate-NEXT: Is migrated to:
// cudnnCreate-NEXT:   dpct::dnnl::engine_ext h;
// cudnnCreate-NEXT:   h.create_engine();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnActivationBackward | FileCheck %s -check-prefix=cudnnActivationBackward
// cudnnActivationBackward: CUDA API:
// cudnnActivationBackward-NEXT: cudnnHandle_t h;
// cudnnActivationBackward-NEXT: cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnActivationBackward-NEXT: cudnnActivationBackward(
// cudnnActivationBackward-NEXT:     h /*cudnnHandle_t*/, desc /*cudnnActivationDescriptor_t*/,
// cudnnActivationBackward-NEXT:     alpha /*void **/, dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
// cudnnActivationBackward-NEXT:     diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
// cudnnActivationBackward-NEXT:     src_d /*cudnnTensorDescriptor_t*/, src /*void **/, beta /*void **/,
// cudnnActivationBackward-NEXT:     diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/);
// cudnnActivationBackward-NEXT: Is migrated to:
// cudnnActivationBackward-NEXT: dpct::dnnl::engine_ext h;
// cudnnActivationBackward-NEXT: h.create_engine();
// cudnnActivationBackward-NEXT: h.async_activation_backward(desc, *alpha, dst_d, dst, diff_dst_d, diff_dst, src_d, src, *beta, diff_src_d, diff_src);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnActivationForward | FileCheck %s -check-prefix=cudnnActivationForward
// cudnnActivationForward: CUDA API:
// cudnnActivationForward-NEXT: cudnnHandle_t h;
// cudnnActivationForward-NEXT: cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnActivationForward-NEXT: cudnnActivationForward(
// cudnnActivationForward-NEXT:     h /*cudnnHandle_t*/, desc /*cudnnActivationDescriptor_t*/,
// cudnnActivationForward-NEXT:     alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnActivationForward-NEXT:     beta /*void **/, dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
// cudnnActivationForward-NEXT: Is migrated to:
// cudnnActivationForward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnActivationForward-NEXT:   h.create_engine();
// cudnnActivationForward-NEXT:   h.async_activation_forward(desc, *alpha, src_d, src, *beta, dst_d, dst);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateActivationDescriptor | FileCheck %s -check-prefix=cudnnCreateActivationDescriptor
// cudnnCreateActivationDescriptor: CUDA API:
// cudnnCreateActivationDescriptor-NEXT:   cudnnCreateActivationDescriptor(d /*cudnnActivationDescriptor_t **/);
// cudnnCreateActivationDescriptor-NEXT: The API is Removed.
// cudnnCreateActivationDescriptor-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateTensorDescriptor | FileCheck %s -check-prefix=cudnnCreateTensorDescriptor
// cudnnCreateTensorDescriptor: CUDA API:
// cudnnCreateTensorDescriptor-NEXT:   cudnnCreateTensorDescriptor(d /*cudnnTensorDescriptor_t **/);
// cudnnCreateTensorDescriptor-NEXT: The API is Removed.
// cudnnCreateTensorDescriptor-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroy | FileCheck %s -check-prefix=cudnnDestroy
// cudnnDestroy: CUDA API:
// cudnnDestroy-NEXT: cudnnDestroy(h /*cudnnHandle_t*/);
// cudnnDestroy-NEXT: The API is Removed.
// cudnnDestroy-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetActivationDescriptor | FileCheck %s -check-prefix=cudnnSetActivationDescriptor
// cudnnSetActivationDescriptor: CUDA API:
// cudnnSetActivationDescriptor-NEXT: cudnnActivationDescriptor_t d;
// cudnnSetActivationDescriptor-NEXT: cudnnSetActivationDescriptor(d /*cudnnActivationDescriptor_t*/,
// cudnnSetActivationDescriptor-NEXT:                              m /*cudnnActivationMode_t*/,
// cudnnSetActivationDescriptor-NEXT:                              p /*cudnnNanPropagation_t*/, c /*double*/);
// cudnnSetActivationDescriptor-NEXT: Is migrated to:
// cudnnSetActivationDescriptor-NEXT:   dpct::dnnl::activation_desc d;
// cudnnSetActivationDescriptor-NEXT:   d.set(m, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetStream | FileCheck %s -check-prefix=cudnnSetStream
// cudnnSetStream: CUDA API:
// cudnnSetStream-NEXT: cudnnHandle_t h;
// cudnnSetStream-NEXT: cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnSetStream-NEXT: cudnnSetStream(h /*cudnnHandle_t*/, s /*cudaStream_t*/);
// cudnnSetStream-NEXT: Is migrated to:
// cudnnSetStream-NEXT:   dpct::dnnl::engine_ext h;
// cudnnSetStream-NEXT:   h.create_engine();
// cudnnSetStream-NEXT:   h.set_queue(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnSetTensor4dDescriptor | FileCheck %s -check-prefix=cudnnSetTensor4dDescriptor
// cudnnSetTensor4dDescriptor: CUDA API:
// cudnnSetTensor4dDescriptor-NEXT: cudnnTensorDescriptor_t d;
// cudnnSetTensor4dDescriptor-NEXT: cudnnSetTensor4dDescriptor(d /*cudnnTensorDescriptor_t*/,
// cudnnSetTensor4dDescriptor-NEXT:                            f /*cudnnTensorFormat_t*/, t /*cudnnDataType_t*/,
// cudnnSetTensor4dDescriptor-NEXT:                            n /*int*/, c /*int*/, h /*int*/, w /*int*/);
// cudnnSetTensor4dDescriptor-NEXT: Is migrated to:
// cudnnSetTensor4dDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnSetTensor4dDescriptor-NEXT:   d.set(f, t, n, c, h, w);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnGetTensor4dDescriptor | FileCheck %s -check-prefix=cudnnGetTensor4dDescriptor
// cudnnGetTensor4dDescriptor: CUDA API:
// cudnnGetTensor4dDescriptor-NEXT: cudnnTensorDescriptor_t d;
// cudnnGetTensor4dDescriptor-NEXT: cudnnGetTensor4dDescriptor(d /*cudnnTensorDescriptor_t*/,
// cudnnGetTensor4dDescriptor-NEXT:                            t /*cudnnDataType_t **/, n /*int **/, c /*int **/,
// cudnnGetTensor4dDescriptor-NEXT:                            h /*int **/, w /*int **/, ns /*int **/,
// cudnnGetTensor4dDescriptor-NEXT:                            cs /*int **/, hs /*int **/, ws /*int **/);
// cudnnGetTensor4dDescriptor-NEXT: Is migrated to:
// cudnnGetTensor4dDescriptor-NEXT:   dpct::dnnl::memory_desc_ext d;
// cudnnGetTensor4dDescriptor-NEXT:   d.get(t, n, c, h, w, ns, cs, hs, ws);

///dpct --cuda-include-path="/rdrive/ref/cuda/lin/cuda-12.2/include" --query-api-mapping=cudnnBatchNormalizationBackward

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnAddTensor | FileCheck %s -check-prefix=cudnnAddTensor
// cudnnAddTensor: CUDA API:
// cudnnAddTensor-NEXT:   cudnnHandle_t h;
// cudnnAddTensor-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnAddTensor-NEXT:   cudnnAddTensor(h /*cudnnHandle_t*/, alpha /*void **/,
// cudnnAddTensor-NEXT:                  src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnAddTensor-NEXT:                  beta /*void **/, dst_d /*cudnnTensorDescriptor_t*/,
// cudnnAddTensor-NEXT:                  dst /*void **/);
// cudnnAddTensor-NEXT: Is migrated to:
// cudnnAddTensor-NEXT:   dpct::dnnl::engine_ext h;
// cudnnAddTensor-NEXT:   h.create_engine();
// cudnnAddTensor-NEXT:   h.async_sum(*alpha, src_d, src, *beta, dst_d, dst);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnBatchNormalizationBackward | FileCheck %s -check-prefix=cudnnBatchNormalizationBackward
// cudnnBatchNormalizationBackward: CUDA API:
// cudnnBatchNormalizationBackward-NEXT: cudnnHandle_t h;
// cudnnBatchNormalizationBackward-NEXT: cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnBatchNormalizationBackward-NEXT: cudnnBatchNormalizationBackward(
// cudnnBatchNormalizationBackward-NEXT:     h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/, alphad /*void **/,
// cudnnBatchNormalizationBackward-NEXT:     betad /*void **/, alphap /*void **/, betap /*void **/,
// cudnnBatchNormalizationBackward-NEXT:     src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnBatchNormalizationBackward-NEXT:     diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
// cudnnBatchNormalizationBackward-NEXT:     diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/,
// cudnnBatchNormalizationBackward-NEXT:     p_d /*cudnnTensorDescriptor_t*/, scale /*void **/, diff_scale /*void **/,
// cudnnBatchNormalizationBackward-NEXT:     diff_bias /*void **/, eps /*double*/, smean /*void **/, svar /*void **/);
// cudnnBatchNormalizationBackward-NEXT: Is migrated to:
// cudnnBatchNormalizationBackward-NEXT: dpct::dnnl::engine_ext h;
// cudnnBatchNormalizationBackward-NEXT: h.create_engine();
// cudnnBatchNormalizationBackward-NEXT: h.async_batch_normalization_backward(m, eps, *alphad, src_d, src, diff_dst_d, diff_dst, *betad, diff_src_d, diff_src, *alphap, p_d, scale, *betap, diff_scale, diff_bias, smean, svar);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnBatchNormalizationBackwardEx | FileCheck %s -check-prefix=cudnnBatchNormalizationBackwardEx
// cudnnBatchNormalizationBackwardEx: CUDA API:
// cudnnBatchNormalizationBackwardEx-NEXT: cudnnHandle_t h;
// cudnnBatchNormalizationBackwardEx-NEXT: cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnBatchNormalizationBackwardEx-NEXT: cudnnBatchNormalizationBackwardEx(
// cudnnBatchNormalizationBackwardEx-NEXT:       h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/,
// cudnnBatchNormalizationBackwardEx-NEXT:       op /*cudnnBatchNormOps_t*/, diff_alphad /*void **/, diff_betad /*void **/,
// cudnnBatchNormalizationBackwardEx-NEXT:       diff_alphap /*void **/, diff_betap /*void **/,
// cudnnBatchNormalizationBackwardEx-NEXT:       src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnBatchNormalizationBackwardEx-NEXT:       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
// cudnnBatchNormalizationBackwardEx-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
// cudnnBatchNormalizationBackwardEx-NEXT:       diff_summand_d /*cudnnTensorDescriptor_t*/, diff_summand /*void **/,
// cudnnBatchNormalizationBackwardEx-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/,
// cudnnBatchNormalizationBackwardEx-NEXT:       p_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
// cudnnBatchNormalizationBackwardEx-NEXT:       diff_scale /*void **/, diff_bias /*void **/, eps /*double*/,
// cudnnBatchNormalizationBackwardEx-NEXT:       smean /*void **/, svar /*void **/, adesc /*cudnnActivationDescriptor_t*/,
// cudnnBatchNormalizationBackwardEx-NEXT:       workspace /*void **/, workspace_size /*size_t*/, reservespace /*void **/,
// cudnnBatchNormalizationBackwardEx-NEXT:       reservespace_size /*size_t*/);
// cudnnBatchNormalizationBackwardEx-NEXT: Is migrated to:
// cudnnBatchNormalizationBackwardEx-NEXT: dpct::dnnl::engine_ext h;
// cudnnBatchNormalizationBackwardEx-NEXT: h.create_engine();
// cudnnBatchNormalizationBackwardEx-NEXT: h.async_batch_normalization_backward(m, op, adesc, eps, *diff_alphad, src_d, src, dst_d, dst, diff_dst_d, diff_dst, *diff_betad, diff_src_d, diff_src, diff_summand_d, diff_summand, *diff_alphap, p_d, scale, bias, *diff_betap, diff_scale, diff_bias, smean, svar, reservespace_size, reservespace);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnBatchNormalizationForwardInference | FileCheck %s -check-prefix=cudnnBatchNormalizationForwardInference
// cudnnBatchNormalizationForwardInference: CUDA API:
// cudnnBatchNormalizationForwardInference-NEXT:   cudnnHandle_t h;
// cudnnBatchNormalizationForwardInference-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnBatchNormalizationForwardInference-NEXT:   cudnnBatchNormalizationForwardInference(
// cudnnBatchNormalizationForwardInference-NEXT:       h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/, alpha /*void **/,
// cudnnBatchNormalizationForwardInference-NEXT:       beta /*void **/, src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnBatchNormalizationForwardInference-NEXT:       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
// cudnnBatchNormalizationForwardInference-NEXT:       p_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
// cudnnBatchNormalizationForwardInference-NEXT:       mean /*void **/, var /*void **/, eps /*double*/);
// cudnnBatchNormalizationForwardInference-NEXT: Is migrated to:
// cudnnBatchNormalizationForwardInference-NEXT:   dpct::dnnl::engine_ext h;
// cudnnBatchNormalizationForwardInference-NEXT:   h.create_engine();
// cudnnBatchNormalizationForwardInference-NEXT:   h.async_batch_normalization_forward_inference(m, eps, *alpha, src_d, src, *beta, dst_d, dst, p_d, scale, bias, mean, var);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnBatchNormalizationForwardTraining | FileCheck %s -check-prefix=cudnnBatchNormalizationForwardTraining
// cudnnBatchNormalizationForwardTraining: CUDA API:
// cudnnBatchNormalizationForwardTraining-NEXT:   cudnnHandle_t h;
// cudnnBatchNormalizationForwardTraining-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnBatchNormalizationForwardTraining-NEXT:   cudnnBatchNormalizationForwardTraining(
// cudnnBatchNormalizationForwardTraining-NEXT:       h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/, alpha /*void **/,
// cudnnBatchNormalizationForwardTraining-NEXT:       beta /*void **/, src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnBatchNormalizationForwardTraining-NEXT:       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
// cudnnBatchNormalizationForwardTraining-NEXT:       p_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
// cudnnBatchNormalizationForwardTraining-NEXT:       factor /*double*/, rmean /*void **/, rvar /*void **/, eps /*double*/,
// cudnnBatchNormalizationForwardTraining-NEXT:       mean /*void **/, var /*void **/);
// cudnnBatchNormalizationForwardTraining-NEXT: Is migrated to:
// cudnnBatchNormalizationForwardTraining-NEXT:   dpct::dnnl::engine_ext h;
// cudnnBatchNormalizationForwardTraining-NEXT:   h.create_engine();
// cudnnBatchNormalizationForwardTraining-NEXT:   h.async_batch_normalization_forward_training(m, eps, factor, *alpha, src_d, src, *beta, dst_d, dst, p_d, scale, bias, rmean, rvar, mean, var);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnBatchNormalizationForwardTrainingEx | FileCheck %s -check-prefix=cudnnBatchNormalizationForwardTrainingEx
// cudnnBatchNormalizationForwardTrainingEx: CUDA API:
// cudnnBatchNormalizationForwardTrainingEx-NEXT:   cudnnHandle_t h;
// cudnnBatchNormalizationForwardTrainingEx-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnBatchNormalizationForwardTrainingEx-NEXT:   cudnnBatchNormalizationForwardTrainingEx(
// cudnnBatchNormalizationForwardTrainingEx-NEXT:       h /*cudnnHandle_t*/, m /*cudnnBatchNormMode_t*/,
// cudnnBatchNormalizationForwardTrainingEx-NEXT:       op /*cudnnBatchNormOps_t*/, alpha /*void **/, beta /*void **/,
// cudnnBatchNormalizationForwardTrainingEx-NEXT:       src_d /*cudnnTensorDescriptor_t*/, src /*void **/,
// cudnnBatchNormalizationForwardTrainingEx-NEXT:       summand_d /*cudnnTensorDescriptor_t*/, summand /*void **/,
// cudnnBatchNormalizationForwardTrainingEx-NEXT:       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/,
// cudnnBatchNormalizationForwardTrainingEx-NEXT:       p_d /*cudnnTensorDescriptor_t*/, scale /*void **/, bias /*void **/,
// cudnnBatchNormalizationForwardTrainingEx-NEXT:       factor /*double*/, rmean /*void **/, rvar /*void **/, eps /*double*/,
// cudnnBatchNormalizationForwardTrainingEx-NEXT:       smean /*void **/, svar /*void **/, adesc /*cudnnActivationDescriptor_t*/,
// cudnnBatchNormalizationForwardTrainingEx-NEXT:       workspace /*void **/, workspace_size /*size_t*/, reservespace /*void **/,
// cudnnBatchNormalizationForwardTrainingEx-NEXT:       reservespace_size /*size_t*/);
// cudnnBatchNormalizationForwardTrainingEx-NEXT: Is migrated to:
// cudnnBatchNormalizationForwardTrainingEx-NEXT:   dpct::dnnl::engine_ext h;
// cudnnBatchNormalizationForwardTrainingEx-NEXT:   h.create_engine();
// cudnnBatchNormalizationForwardTrainingEx-NEXT:   h.async_batch_normalization_forward_training(m, op, adesc, eps, factor, *alpha, src_d, src, *beta, dst_d, dst, summand_d, summand, p_d, scale, bias, rmean, rvar, smean, svar, reservespace_size, reservespace);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnConvolutionBackwardBias | FileCheck %s -check-prefix=cudnnConvolutionBackwardBias
// cudnnConvolutionBackwardBias: CUDA API:
// cudnnConvolutionBackwardBias-NEXT:   cudnnHandle_t h;
// cudnnConvolutionBackwardBias-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnConvolutionBackwardBias-NEXT:   cudnnConvolutionBackwardBias(h /*cudnnHandle_t*/, alpha /*void **/,
// cudnnConvolutionBackwardBias-NEXT:                                diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnConvolutionBackwardBias-NEXT:                                diff_dst /*void **/, beta /*void **/,
// cudnnConvolutionBackwardBias-NEXT:                                diff_bias_d /*cudnnTensorDescriptor_t*/,
// cudnnConvolutionBackwardBias-NEXT:                                diff_bias /*void **/);
// cudnnConvolutionBackwardBias-NEXT: Is migrated to:
// cudnnConvolutionBackwardBias-NEXT:   dpct::dnnl::engine_ext h;
// cudnnConvolutionBackwardBias-NEXT:   h.create_engine();
// cudnnConvolutionBackwardBias-NEXT:   h.async_convolution_backward_bias(*alpha, diff_dst_d, diff_dst, *beta, diff_bias_d, diff_bias);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnConvolutionBackwardData | FileCheck %s -check-prefix=cudnnConvolutionBackwardData
// cudnnConvolutionBackwardData: CUDA API:
// cudnnConvolutionBackwardData-NEXT:   cudnnHandle_t h;
// cudnnConvolutionBackwardData-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnConvolutionBackwardData-NEXT:   cudnnConvolutionBackwardData(
// cudnnConvolutionBackwardData-NEXT:       h /*cudnnHandle_t*/, alpha /*void **/,
// cudnnConvolutionBackwardData-NEXT:       filter_d /*cudnnTensorDescriptor_t*/, filter /*void **/,
// cudnnConvolutionBackwardData-NEXT:       diff_dst_d /*cudnnTensorDescriptor_t*/, diff_dst /*void **/,
// cudnnConvolutionBackwardData-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/, alg /*cudnnConvolutionFwdAlgo_t*/,
// cudnnConvolutionBackwardData-NEXT:       workspace /*void **/, workspace_size /*size_t*/, beta /*void **/,
// cudnnConvolutionBackwardData-NEXT:       diff_src_d /*cudnnTensorDescriptor_t*/, diff_src /*void **/);
// cudnnConvolutionBackwardData-NEXT: Is migrated to:
// cudnnConvolutionBackwardData-NEXT:   dpct::dnnl::engine_ext h;
// cudnnConvolutionBackwardData-NEXT:   h.create_engine();
// cudnnConvolutionBackwardData-NEXT:   h.async_convolution_backward_data(cdesc, alg, *alpha, filter_d, filter, diff_dst_d, diff_dst, *beta, diff_src_d, diff_src);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnConvolutionBackwardFilter | FileCheck %s -check-prefix=cudnnConvolutionBackwardFilter
// cudnnConvolutionBackwardFilter: CUDA API:
// cudnnConvolutionBackwardFilter-NEXT:   cudnnHandle_t h;
// cudnnConvolutionBackwardFilter-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnConvolutionBackwardFilter-NEXT:   cudnnConvolutionBackwardFilter(
// cudnnConvolutionBackwardFilter-NEXT:       h /*cudnnHandle_t*/, alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnConvolutionBackwardFilter-NEXT:       src /*void **/, diff_dst_d /*cudnnTensorDescriptor_t*/,
// cudnnConvolutionBackwardFilter-NEXT:       diff_dst /*void **/, cdesc /*cudnnConvolutionDescriptor_t*/,
// cudnnConvolutionBackwardFilter-NEXT:       alg /*cudnnConvolutionFwdAlgo_t*/, workspace /*void **/,
// cudnnConvolutionBackwardFilter-NEXT:       workspace_size /*size_t*/, beta /*void **/,
// cudnnConvolutionBackwardFilter-NEXT:       diff_filter_d /*cudnnTensorDescriptor_t*/, diff_filter /*void **/);
// cudnnConvolutionBackwardFilter-NEXT: Is migrated to:
// cudnnConvolutionBackwardFilter-NEXT:   dpct::dnnl::engine_ext h;
// cudnnConvolutionBackwardFilter-NEXT:   h.create_engine();
// cudnnConvolutionBackwardFilter-NEXT:   h.async_convolution_backward_weight(cdesc, alg, *alpha, src_d, src, diff_dst_d, diff_dst, *beta, diff_filter_d, diff_filter);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnConvolutionBiasActivationForward | FileCheck %s -check-prefix=cudnnConvolutionBiasActivationForward
// cudnnConvolutionBiasActivationForward: CUDA API:
// cudnnConvolutionBiasActivationForward-NEXT:   cudnnHandle_t h;
// cudnnConvolutionBiasActivationForward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnConvolutionBiasActivationForward-NEXT:   cudnnConvolutionBiasActivationForward(
// cudnnConvolutionBiasActivationForward-NEXT:       h /*cudnnHandle_t*/, alpha1 /*void **/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnConvolutionBiasActivationForward-NEXT:       src /*void **/, filter_d /*cudnnTensorDescriptor_t*/, filter /*void **/,
// cudnnConvolutionBiasActivationForward-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/, alg /*cudnnConvolutionFwdAlgo_t*/,
// cudnnConvolutionBiasActivationForward-NEXT:       workspace /*void **/, workspace_size /*size_t*/, alpha2 /*void **/,
// cudnnConvolutionBiasActivationForward-NEXT:       summand_d /*cudnnTensorDescriptor_t*/, summand /*void **/,
// cudnnConvolutionBiasActivationForward-NEXT:       bias_d /*cudnnTensorDescriptor_t*/, bias /*void **/,
// cudnnConvolutionBiasActivationForward-NEXT:       adesc /*cudnnActivationDescriptor_t*/, dst_d /*cudnnTensorDescriptor_t*/,
// cudnnConvolutionBiasActivationForward-NEXT:       dst /*void **/);
// cudnnConvolutionBiasActivationForward-NEXT: Is migrated to:
// cudnnConvolutionBiasActivationForward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnConvolutionBiasActivationForward-NEXT:   h.create_engine();
// cudnnConvolutionBiasActivationForward-NEXT:   h.async_convolution_forward(cdesc, alg, adesc, *alpha1, src_d, src, filter_d, filter, *alpha2, summand_d, summand, bias_d, bias, dst_d, dst);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnConvolutionForward | FileCheck %s -check-prefix=cudnnConvolutionForward
// cudnnConvolutionForward: CUDA API:
// cudnnConvolutionForward-NEXT:   cudnnHandle_t h;
// cudnnConvolutionForward-NEXT:   cudnnCreate(&h /*cudnnHandle_t **/);
// cudnnConvolutionForward-NEXT:   cudnnConvolutionForward(
// cudnnConvolutionForward-NEXT:       h /*cudnnHandle_t*/, alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/,
// cudnnConvolutionForward-NEXT:       src /*void **/, filter_d /*cudnnTensorDescriptor_t*/, filter /*void **/,
// cudnnConvolutionForward-NEXT:       cdesc /*cudnnConvolutionDescriptor_t*/, alg /*cudnnConvolutionFwdAlgo_t*/,
// cudnnConvolutionForward-NEXT:       workspace /*void **/, workspace_size /*size_t*/, beta /*void **/,
// cudnnConvolutionForward-NEXT:       dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
// cudnnConvolutionForward-NEXT: Is migrated to:
// cudnnConvolutionForward-NEXT:   dpct::dnnl::engine_ext h;
// cudnnConvolutionForward-NEXT:   h.create_engine();
// cudnnConvolutionForward-NEXT:   h.async_convolution_forward(cdesc, alg, *(float *)alpha, src_d, src, filter_d, filter, *(float *)beta, dst_d, dst);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateConvolutionDescriptor | FileCheck %s -check-prefix=cudnnCreateConvolutionDescriptor
// cudnnCreateConvolutionDescriptor: CUDA API:
// cudnnCreateConvolutionDescriptor-NEXT:   cudnnCreateConvolutionDescriptor(d /*cudnnConvolutionDescriptor_t **/);
// cudnnCreateConvolutionDescriptor-NEXT: The API is Removed.
// cudnnCreateConvolutionDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateDropoutDescriptor | FileCheck %s -check-prefix=cudnnCreateDropoutDescriptor
// cudnnCreateDropoutDescriptor: CUDA API:
// cudnnCreateDropoutDescriptor-NEXT:   cudnnDropoutDescriptor_t d;
// cudnnCreateDropoutDescriptor-NEXT:   cudnnCreateDropoutDescriptor(&d /*cudnnDropoutDescriptor_t **/);
// cudnnCreateDropoutDescriptor-NEXT: Is migrated to:
// cudnnCreateDropoutDescriptor-NEXT:   dpct::dnnl::dropout_desc d;
// cudnnCreateDropoutDescriptor-NEXT:   d.init();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateFilterDescriptor | FileCheck %s -check-prefix=cudnnCreateFilterDescriptor
// cudnnCreateFilterDescriptor: CUDA API:
// cudnnCreateFilterDescriptor-NEXT:   cudnnCreateFilterDescriptor(d /*cudnnFilterDescriptor_t **/);
// cudnnCreateFilterDescriptor-NEXT: The API is Removed.
// cudnnCreateFilterDescriptor-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateLRNDescriptor | FileCheck %s -check-prefix=cudnnCreateLRNDescriptor
// cudnnCreateLRNDescriptor: CUDA API:
// cudnnCreateLRNDescriptor-NEXT:   cudnnCreateLRNDescriptor(d /*cudnnLRNDescriptor_t **/);
// cudnnCreateLRNDescriptor-NEXT: The API is Removed.
// cudnnCreateLRNDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateOpTensorDescriptor | FileCheck %s -check-prefix=cudnnCreateOpTensorDescriptor
// cudnnCreateOpTensorDescriptor: CUDA API:
// cudnnCreateOpTensorDescriptor-NEXT:   cudnnCreateOpTensorDescriptor(d /*cudnnOpTensorDescriptor_t **/);
// cudnnCreateOpTensorDescriptor-NEXT: The API is Removed.
// cudnnCreateOpTensorDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreatePoolingDescriptor | FileCheck %s -check-prefix=cudnnCreatePoolingDescriptor
// cudnnCreatePoolingDescriptor: CUDA API:
// cudnnCreatePoolingDescriptor-NEXT:   cudnnCreatePoolingDescriptor(d /*cudnnPoolingDescriptor_t  **/);
// cudnnCreatePoolingDescriptor-NEXT: The API is Removed.
// cudnnCreatePoolingDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateReduceTensorDescriptor | FileCheck %s -check-prefix=cudnnCreateReduceTensorDescriptor
// cudnnCreateReduceTensorDescriptor: CUDA API:
// cudnnCreateReduceTensorDescriptor-NEXT:   cudnnCreateReduceTensorDescriptor(d /*cudnnReduceTensorDescriptor_t  **/);
// cudnnCreateReduceTensorDescriptor-NEXT: The API is Removed.
// cudnnCreateReduceTensorDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateRNNDataDescriptor | FileCheck %s -check-prefix=cudnnCreateRNNDataDescriptor
// cudnnCreateRNNDataDescriptor: CUDA API:
// cudnnCreateRNNDataDescriptor-NEXT:   cudnnCreateRNNDataDescriptor(d /*cudnnRNNDataDescriptor_t **/);
// cudnnCreateRNNDataDescriptor-NEXT: The API is Removed.
// cudnnCreateRNNDataDescriptor-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnCreateRNNDescriptor | FileCheck %s -check-prefix=cudnnCreateRNNDescriptor
// cudnnCreateRNNDescriptor: CUDA API:
// cudnnCreateRNNDescriptor-NEXT:   cudnnCreateRNNDescriptor(d /*cudnnRNNDescriptor_t **/);
// cudnnCreateRNNDescriptor-NEXT: The API is Removed.
// cudnnCreateRNNDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDeriveBNTensorDescriptor | FileCheck %s -check-prefix=cudnnDeriveBNTensorDescriptor
// cudnnDeriveBNTensorDescriptor: CUDA API:
// cudnnDeriveBNTensorDescriptor-NEXT:   cudnnDeriveBNTensorDescriptor(derived_desc /*cudnnTensorDescriptor_t*/,
// cudnnDeriveBNTensorDescriptor-NEXT:                                 src_d /*cudnnTensorDescriptor_t*/,
// cudnnDeriveBNTensorDescriptor-NEXT:                                 m /*cudnnBatchNormMode_t*/);
// cudnnDeriveBNTensorDescriptor-NEXT: Is migrated to:
// cudnnDeriveBNTensorDescriptor-NEXT:   dpct::dnnl::derive_batch_normalization_memory_desc(derived_desc, src_d, m);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDeriveNormTensorDescriptor | FileCheck %s -check-prefix=cudnnDeriveNormTensorDescriptor
// cudnnDeriveNormTensorDescriptor: CUDA API:
// cudnnDeriveNormTensorDescriptor-NEXT:   cudnnDeriveNormTensorDescriptor(derived_p1_desc /*cudnnTensorDescriptor_t*/,
// cudnnDeriveNormTensorDescriptor-NEXT:                                   derived_p2_desc /*cudnnTensorDescriptor_t*/,
// cudnnDeriveNormTensorDescriptor-NEXT:                                   src_d /*cudnnTensorDescriptor_t*/,
// cudnnDeriveNormTensorDescriptor-NEXT:                                   m /*cudnnNormMode_t*/, group_count /*int*/);
// cudnnDeriveNormTensorDescriptor-NEXT: Is migrated to:
// cudnnDeriveNormTensorDescriptor-NEXT:   dpct::dnnl::derive_batch_normalization_memory_desc(derived_p1_desc, derived_p2_desc, src_d, m);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroyActivationDescriptor | FileCheck %s -check-prefix=cudnnDestroyActivationDescriptor
// cudnnDestroyActivationDescriptor: CUDA API:
// cudnnDestroyActivationDescriptor-NEXT:   cudnnDestroyActivationDescriptor(d /*cudnnActivationDescriptor_t*/);
// cudnnDestroyActivationDescriptor-NEXT: The API is Removed.
// cudnnDestroyActivationDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroyConvolutionDescriptor | FileCheck %s -check-prefix=cudnnDestroyConvolutionDescriptor
// cudnnDestroyConvolutionDescriptor: CUDA API:
// cudnnDestroyConvolutionDescriptor-NEXT:   cudnnDestroyConvolutionDescriptor(d /*cudnnConvolutionDescriptor_t*/);
// cudnnDestroyConvolutionDescriptor-NEXT: The API is Removed.
// cudnnDestroyConvolutionDescriptor-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroyDropoutDescriptor | FileCheck %s -check-prefix=cudnnDestroyDropoutDescriptor
// cudnnDestroyDropoutDescriptor: CUDA API:
// cudnnDestroyDropoutDescriptor-NEXT:   cudnnDestroyDropoutDescriptor(d /*cudnnDropoutDescriptor_t*/);
// cudnnDestroyDropoutDescriptor-NEXT: The API is Removed.
// cudnnDestroyDropoutDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroyFilterDescriptor | FileCheck %s -check-prefix=cudnnDestroyFilterDescriptor
// cudnnDestroyFilterDescriptor: CUDA API:
// cudnnDestroyFilterDescriptor-NEXT:   cudnnDestroyFilterDescriptor(d /*cudnnFilterDescriptor_t*/);
// cudnnDestroyFilterDescriptor-NEXT: The API is Removed.
// cudnnDestroyFilterDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroyLRNDescriptor | FileCheck %s -check-prefix=cudnnDestroyLRNDescriptor
// cudnnDestroyLRNDescriptor: CUDA API:
// cudnnDestroyLRNDescriptor-NEXT:   cudnnDestroyLRNDescriptor(d /*cudnnLRNDescriptor_t*/);
// cudnnDestroyLRNDescriptor-NEXT: The API is Removed.
// cudnnDestroyLRNDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroyOpTensorDescriptor | FileCheck %s -check-prefix=cudnnDestroyOpTensorDescriptor
// cudnnDestroyOpTensorDescriptor: CUDA API:
// cudnnDestroyOpTensorDescriptor-NEXT:   cudnnDestroyOpTensorDescriptor(d /*cudnnOpTensorDescriptor_t*/);
// cudnnDestroyOpTensorDescriptor-NEXT: The API is Removed.
// cudnnDestroyOpTensorDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroyPoolingDescriptor | FileCheck %s -check-prefix=cudnnDestroyPoolingDescriptor
// cudnnDestroyPoolingDescriptor: CUDA API:
// cudnnDestroyPoolingDescriptor-NEXT:   cudnnDestroyPoolingDescriptor(d /*cudnnPoolingDescriptor_t*/);
// cudnnDestroyPoolingDescriptor-NEXT: The API is Removed.
// cudnnDestroyPoolingDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroyReduceTensorDescriptor | FileCheck %s -check-prefix=cudnnDestroyReduceTensorDescriptor
// cudnnDestroyReduceTensorDescriptor: CUDA API:
// cudnnDestroyReduceTensorDescriptor-NEXT:   cudnnDestroyReduceTensorDescriptor(d /*cudnnReduceTensorDescriptor_t*/);
// cudnnDestroyReduceTensorDescriptor-NEXT: The API is Removed.
// cudnnDestroyReduceTensorDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroyRNNDataDescriptor | FileCheck %s -check-prefix=cudnnDestroyRNNDataDescriptor
// cudnnDestroyRNNDataDescriptor: CUDA API:
// cudnnDestroyRNNDataDescriptor-NEXT:   cudnnDestroyRNNDataDescriptor(d /*cudnnRNNDataDescriptor_t*/);
// cudnnDestroyRNNDataDescriptor-NEXT: The API is Removed.
// cudnnDestroyRNNDataDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroyRNNDescriptor | FileCheck %s -check-prefix=cudnnDestroyRNNDescriptor
// cudnnDestroyRNNDescriptor: CUDA API:
// cudnnDestroyRNNDescriptor-NEXT:   cudnnDestroyRNNDescriptor(d /*cudnnRNNDescriptor_t*/);
// cudnnDestroyRNNDescriptor-NEXT: The API is Removed.
// cudnnDestroyRNNDescriptor-EMPTY: 

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDestroyTensorDescriptor | FileCheck %s -check-prefix=cudnnDestroyTensorDescriptor
// cudnnDestroyTensorDescriptor: CUDA API:
// cudnnDestroyTensorDescriptor-NEXT:   cudnnDestroyTensorDescriptor(d /*cudnnTensorDescriptor_t*/);
// cudnnDestroyTensorDescriptor-NEXT: The API is Removed.
// cudnnDestroyTensorDescriptor-EMPTY: 

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
// cudnnGetErrorString-NEXT:   r = "cudnnGetErrorString is not supported"/*cudnnGetErrorString(s)*/;

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
