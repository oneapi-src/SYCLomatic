// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.2
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.2

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
// cudnnDeriveBNTensorDescriptor-NEXT:   dpct::dnnl::engine_ext::derive_batch_normalization_memory_desc(derived_desc, src_d, m);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudnnDeriveNormTensorDescriptor | FileCheck %s -check-prefix=cudnnDeriveNormTensorDescriptor
// cudnnDeriveNormTensorDescriptor: CUDA API:
// cudnnDeriveNormTensorDescriptor-NEXT:   cudnnDeriveNormTensorDescriptor(derived_p1_desc /*cudnnTensorDescriptor_t*/,
// cudnnDeriveNormTensorDescriptor-NEXT:                                   derived_p2_desc /*cudnnTensorDescriptor_t*/,
// cudnnDeriveNormTensorDescriptor-NEXT:                                   src_d /*cudnnTensorDescriptor_t*/,
// cudnnDeriveNormTensorDescriptor-NEXT:                                   m /*cudnnNormMode_t*/, group_count /*int*/);
// cudnnDeriveNormTensorDescriptor-NEXT: Is migrated to:
// cudnnDeriveNormTensorDescriptor-NEXT:   dpct::dnnl::engine_ext::derive_batch_normalization_memory_desc(derived_p1_desc, derived_p2_desc, src_d, m);

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
