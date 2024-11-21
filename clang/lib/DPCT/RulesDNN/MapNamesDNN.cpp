//===--------------- MapNamesDNN.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MapNamesDNN.h"
#include "ASTTraversal.h"
#include "FileGenerator/GenFiles.h"
#include "RuleInfra/CallExprRewriter.h"
#include "RuleInfra/MapNames.h"
#include "RulesDNN/DNNAPIMigration.h"
#include "RulesDNN/MapNamesDNN.h"
#include "RulesLang/RulesLang.h"
#include <map>

using namespace clang;
using namespace clang::dpct;

namespace clang {
namespace dpct {

using MapTy = std::map<std::string, std::string>;

std::unordered_map<std::string, std::shared_ptr<TypeNameRule>>
    MapNamesDNN::CuDNNTypeNamesMap;

// declared in DNNAPIMigration.h
MapTy CuDNNTypeRule::CuDNNEnumNamesMap;
std::map<std::string /*Original API*/, HelperFeatureEnum>
    CuDNNTypeRule::CuDNNEnumNamesHelperFeaturesMap;

void MapNamesDNN::setExplicitNamespaceMap(
    const std::set<ExplicitNamespace> &ExplicitNamespaces) {
  // CuDNN Type names mapping.
  CuDNNTypeNamesMap = {
      {"cudnnHandle_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::engine_ext",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnStatus_t",
       std::make_shared<TypeNameRule>(MapNames::getDpctNamespace() + "err1",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnTensorDescriptor_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::memory_desc_ext",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnFilterDescriptor_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::memory_desc_ext",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnTensorFormat_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::memory_format_tag",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnDataType_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "library_data_t",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnActivationDescriptor_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::activation_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnActivationMode_t",
       std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnLRNDescriptor_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::lrn_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnLRNMode_t", std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnPoolingDescriptor_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::pooling_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnPoolingMode_t", std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnSoftmaxAlgorithm_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::softmax_algorithm",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnSoftmaxMode_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::softmax_mode",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnReduceTensorDescriptor_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::reduction_op",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnReduceTensorOp_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::reduction_op",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnOpTensorDescriptor_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::binary_op",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnOpTensorOp_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::binary_op",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnBatchNormOps_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::batch_normalization_ops",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnBatchNormMode_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::batch_normalization_mode",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnNormOps_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::batch_normalization_ops",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnNormMode_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::batch_normalization_mode",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnConvolutionDescriptor_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::convolution_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnConvolutionFwdAlgo_t",
       std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnConvolutionBwdDataAlgo_t",
       std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnConvolutionBwdFilterAlgo_t",
       std::make_shared<TypeNameRule>("dnnl::algorithm")},
      {"cudnnConvolutionFwdAlgoPerf_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::convolution_algorithm_info",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnConvolutionBwdFilterAlgoPerf_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::convolution_algorithm_info",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnConvolutionBwdDataAlgoPerf_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::convolution_algorithm_info",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnRNNMode_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::rnn_mode",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnRNNBiasMode_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::rnn_bias_mode",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnDirectionMode_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::rnn_direction",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnRNNDescriptor_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::rnn_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnForwardMode_t", std::make_shared<TypeNameRule>("dnnl::prop_kind")},
      {"cudnnRNNDataDescriptor_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::memory_desc_ext",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnRNNDataLayout_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::rnn_memory_format_tag",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnDropoutDescriptor_t",
       std::make_shared<TypeNameRule>(MapNames::getLibraryHelperNamespace() +
                                          "dnnl::dropout_desc",
                                      HelperFeatureEnum::device_ext)},
      {"cudnnConvolutionMode_t", std::make_shared<TypeNameRule>("int")},
      {"cudnnNanPropagation_t", std::make_shared<TypeNameRule>("int")},
  };
  // CuDNN Enum constants name mapping.
  CuDNNTypeRule::CuDNNEnumNamesMap = {
      {"CUDNN_TENSOR_NCHW",
       MapNames::getLibraryHelperNamespace() + "dnnl::memory_format_tag::nchw"},
      {"CUDNN_TENSOR_NHWC",
       MapNames::getLibraryHelperNamespace() + "dnnl::memory_format_tag::nhwc"},
      {"CUDNN_TENSOR_NCHW_VECT_C", MapNames::getLibraryHelperNamespace() +
                                       "dnnl::memory_format_tag::nchw_blocked"},
      {"CUDNN_DATA_FLOAT",
       MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"},
      {"CUDNN_DATA_DOUBLE",
       MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"},
      {"CUDNN_DATA_HALF",
       MapNames::getLibraryHelperNamespace() + "library_data_t::real_half"},
      {"CUDNN_DATA_INT8",
       MapNames::getLibraryHelperNamespace() + "library_data_t::real_int8"},
      {"CUDNN_DATA_UINT8",
       MapNames::getLibraryHelperNamespace() + "library_data_t::real_uint8"},
      {"CUDNN_DATA_INT32",
       MapNames::getLibraryHelperNamespace() + "library_data_t::real_int32"},
      {"CUDNN_DATA_INT8x4",
       MapNames::getLibraryHelperNamespace() + "library_data_t::real_int8_4"},
      {"CUDNN_DATA_INT8x32",
       MapNames::getLibraryHelperNamespace() + "library_data_t::real_int8_32"},
      {"CUDNN_DATA_UINT8x4",
       MapNames::getLibraryHelperNamespace() + "library_data_t::real_uint8_4"},
      {"CUDNN_DATA_BFLOAT16",
       MapNames::getLibraryHelperNamespace() + "library_data_t::real_bfloat16"},
      {"CUDNN_ACTIVATION_SIGMOID",
       "dnnl::algorithm::eltwise_logistic_use_dst_for_bwd"},
      {"CUDNN_ACTIVATION_RELU",
       "dnnl::algorithm::eltwise_relu_use_dst_for_bwd"},
      {"CUDNN_ACTIVATION_TANH",
       "dnnl::algorithm::eltwise_tanh_use_dst_for_bwd"},
      {"CUDNN_ACTIVATION_CLIPPED_RELU", "dnnl::algorithm::eltwise_clip"},
      {"CUDNN_ACTIVATION_ELU", "dnnl::algorithm::eltwise_elu_use_dst_for_bwd"},
      {"CUDNN_ACTIVATION_IDENTITY", "dnnl::algorithm::eltwise_linear"},
      {"CUDNN_ACTIVATION_SWISH", "dnnl::algorithm::eltwise_swish"},
      {"CUDNN_LRN_CROSS_CHANNEL_DIM1", "dnnl::algorithm::lrn_across_channels"},
      {"CUDNN_POOLING_MAX", "dnnl::algorithm::pooling_max"},
      {"CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",
       "dnnl::algorithm::pooling_avg_include_padding"},
      {"CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",
       "dnnl::algorithm::pooling_avg_exclude_padding"},
      {"CUDNN_POOLING_MAX_DETERMINISTIC", "dnnl::algorithm::pooling_max"},
      {"CUDNN_SOFTMAX_FAST", MapNames::getLibraryHelperNamespace() +
                                 "dnnl::softmax_algorithm::normal"},
      {"CUDNN_SOFTMAX_ACCURATE", MapNames::getLibraryHelperNamespace() +
                                     "dnnl::softmax_algorithm::normal"},
      {"CUDNN_SOFTMAX_LOG",
       MapNames::getLibraryHelperNamespace() + "dnnl::softmax_algorithm::log"},
      {"CUDNN_SOFTMAX_MODE_INSTANCE",
       MapNames::getLibraryHelperNamespace() + "dnnl::softmax_mode::instance"},
      {"CUDNN_SOFTMAX_MODE_CHANNEL",
       MapNames::getLibraryHelperNamespace() + "dnnl::softmax_mode::channel"},
      {"CUDNN_REDUCE_TENSOR_ADD",
       MapNames::getLibraryHelperNamespace() + "dnnl::reduction_op::sum"},
      {"CUDNN_REDUCE_TENSOR_MUL",
       MapNames::getLibraryHelperNamespace() + "dnnl::reduction_op::mul"},
      {"CUDNN_REDUCE_TENSOR_MIN",
       MapNames::getLibraryHelperNamespace() + "dnnl::reduction_op::min"},
      {"CUDNN_REDUCE_TENSOR_MAX",
       MapNames::getLibraryHelperNamespace() + "dnnl::reduction_op::max"},
      {"CUDNN_REDUCE_TENSOR_AMAX",
       MapNames::getLibraryHelperNamespace() + "dnnl::reduction_op::amax"},
      {"CUDNN_REDUCE_TENSOR_AVG",
       MapNames::getLibraryHelperNamespace() + "dnnl::reduction_op::mean"},
      {"CUDNN_REDUCE_TENSOR_NORM1",
       MapNames::getLibraryHelperNamespace() + "dnnl::reduction_op::norm1"},
      {"CUDNN_REDUCE_TENSOR_NORM2",
       MapNames::getLibraryHelperNamespace() + "dnnl::reduction_op::norm2"},
      {"CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS",
       MapNames::getLibraryHelperNamespace() +
           "dnnl::reduction_op::mul_no_zeros"},
      {"CUDNN_OP_TENSOR_ADD",
       MapNames::getLibraryHelperNamespace() + "dnnl::binary_op::add"},
      {"CUDNN_OP_TENSOR_MUL",
       MapNames::getLibraryHelperNamespace() + "dnnl::binary_op::mul"},
      {"CUDNN_OP_TENSOR_MIN",
       MapNames::getLibraryHelperNamespace() + "dnnl::binary_op::min"},
      {"CUDNN_OP_TENSOR_MAX",
       MapNames::getLibraryHelperNamespace() + "dnnl::binary_op::max"},
      {"CUDNN_OP_TENSOR_SQRT",
       MapNames::getLibraryHelperNamespace() + "dnnl::binary_op::sqrt"},
      {"CUDNN_OP_TENSOR_NOT",
       MapNames::getLibraryHelperNamespace() + "dnnl::binary_op::neg"},
      {"CUDNN_BATCHNORM_OPS_BN", MapNames::getLibraryHelperNamespace() +
                                     "dnnl::batch_normalization_ops::none"},
      {"CUDNN_BATCHNORM_OPS_BN_ACTIVATION",
       MapNames::getLibraryHelperNamespace() +
           "dnnl::batch_normalization_ops::activation"},
      {"CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION",
       MapNames::getLibraryHelperNamespace() +
           "dnnl::batch_normalization_ops::add_activation"},
      {"CUDNN_BATCHNORM_PER_ACTIVATION",
       MapNames::getLibraryHelperNamespace() +
           "dnnl::batch_normalization_mode::per_activation"},
      {"CUDNN_BATCHNORM_SPATIAL",
       MapNames::getLibraryHelperNamespace() +
           "dnnl::batch_normalization_mode::spatial"},
      {"CUDNN_BATCHNORM_SPATIAL_PERSISTENT",
       MapNames::getLibraryHelperNamespace() +
           "dnnl::batch_normalization_mode::spatial"},
      {"CUDNN_NORM_OPS_NORM", MapNames::getLibraryHelperNamespace() +
                                  "dnnl::batch_normalization_ops::none"},
      {"CUDNN_NORM_OPS_NORM_ACTIVATION",
       MapNames::getLibraryHelperNamespace() +
           "dnnl::batch_normalization_ops::activation"},
      {"CUDNN_NORM_OPS_NORM_ADD_ACTIVATION",
       MapNames::getLibraryHelperNamespace() +
           "dnnl::batch_normalization_ops::add_activation"},
      {"CUDNN_NORM_PER_ACTIVATION",
       MapNames::getLibraryHelperNamespace() +
           "dnnl::batch_normalization_mode::per_activation"},
      {"CUDNN_NORM_PER_CHANNEL", MapNames::getLibraryHelperNamespace() +
                                     "dnnl::batch_normalization_mode::spatial"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_GEMM", "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
       "dnnl::algorithm::convolution_direct"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_FFT", "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
       "dnnl::algorithm::convolution_direct"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
       "dnnl::algorithm::convolution_direct"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
       "dnnl::algorithm::convolution_direct"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
       "dnnl::algorithm::convolution_direct"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",
       "dnnl::algorithm::convolution_auto"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",
       "dnnl::algorithm::convolution_winograd"},
      {"CUDNN_RNN_RELU",
       MapNames::getLibraryHelperNamespace() + "dnnl::rnn_mode::vanilla_relu"},
      {"CUDNN_RNN_TANH",
       MapNames::getLibraryHelperNamespace() + "dnnl::rnn_mode::vanilla_tanh"},
      {"CUDNN_LSTM",
       MapNames::getLibraryHelperNamespace() + "dnnl::rnn_mode::lstm"},
      {"CUDNN_GRU",
       MapNames::getLibraryHelperNamespace() + "dnnl::rnn_mode::gru"},
      {"CUDNN_RNN_NO_BIAS",
       MapNames::getLibraryHelperNamespace() + "dnnl::rnn_bias_mode::none"},
      {"CUDNN_RNN_SINGLE_INP_BIAS",
       MapNames::getLibraryHelperNamespace() + "dnnl::rnn_bias_mode::single"},
      {"CUDNN_UNIDIRECTIONAL", MapNames::getLibraryHelperNamespace() +
                                   "dnnl::rnn_direction::unidirectional"},
      {"CUDNN_BIDIRECTIONAL", MapNames::getLibraryHelperNamespace() +
                                  "dnnl::rnn_direction::bidirectional"},
      {"CUDNN_FWD_MODE_INFERENCE", "dnnl::prop_kind::forward_inference"},
      {"CUDNN_FWD_MODE_TRAINING", "dnnl::prop_kind::forward_training"},
      {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED",
       MapNames::getLibraryHelperNamespace() +
           "dnnl::rnn_memory_format_tag::tnc"},
      {"CUDNN_DEFAULT_MATH", "dnnl::fpmath_mode::strict"},
      {"CUDNN_TENSOR_OP_MATH", "dnnl::fpmath_mode::strict"},
      {"CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION", "dnnl::fpmath_mode::any"},
      {"CUDNN_FMA_MATH", "dnnl::fpmath_mode::strict"},
  };
  // CuDNN Enum constants name to helper feature mapping.
  CuDNNTypeRule::CuDNNEnumNamesHelperFeaturesMap = {
      {"CUDNN_TENSOR_NCHW", HelperFeatureEnum::device_ext},
      {"CUDNN_TENSOR_NHWC", HelperFeatureEnum::device_ext},
      {"CUDNN_TENSOR_NCHW_VECT_C", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_FLOAT", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_DOUBLE", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_HALF", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_INT8", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_UINT8", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_INT32", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_INT8x4", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_INT8x32", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_UINT8x4", HelperFeatureEnum::device_ext},
      {"CUDNN_DATA_BFLOAT16", HelperFeatureEnum::device_ext},
      {"CUDNN_SOFTMAX_FAST", HelperFeatureEnum::device_ext},
      {"CUDNN_SOFTMAX_ACCURATE", HelperFeatureEnum::device_ext},
      {"CUDNN_SOFTMAX_LOG", HelperFeatureEnum::device_ext},
      {"CUDNN_SOFTMAX_MODE_INSTANCE", HelperFeatureEnum::device_ext},
      {"CUDNN_SOFTMAX_MODE_CHANNEL", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_ADD", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_MUL", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_MIN", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_MAX", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_AMAX", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_AVG", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_NORM1", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_NORM2", HelperFeatureEnum::device_ext},
      {"CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_ADD", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_MUL", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_MIN", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_MAX", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_SQRT", HelperFeatureEnum::device_ext},
      {"CUDNN_OP_TENSOR_NOT", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_OPS_BN", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_OPS_BN_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_PER_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_SPATIAL", HelperFeatureEnum::device_ext},
      {"CUDNN_BATCHNORM_SPATIAL_PERSISTENT", HelperFeatureEnum::device_ext},
      {"CUDNN_NORM_OPS_NORM", HelperFeatureEnum::device_ext},
      {"CUDNN_NORM_OPS_NORM_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_NORM_OPS_NORM_ADD_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_NORM_PER_ACTIVATION", HelperFeatureEnum::device_ext},
      {"CUDNN_NORM_PER_CHANNEL", HelperFeatureEnum::device_ext},
      {"CUDNN_RNN_RELU", HelperFeatureEnum::device_ext},
      {"CUDNN_RNN_TANH", HelperFeatureEnum::device_ext},
      {"CUDNN_LSTM", HelperFeatureEnum::device_ext},
      {"CUDNN_GRU", HelperFeatureEnum::device_ext},
      {"CUDNN_RNN_NO_BIAS", HelperFeatureEnum::device_ext},
      {"CUDNN_RNN_SINGLE_INP_BIAS", HelperFeatureEnum::device_ext},
      {"CUDNN_UNIDIRECTIONAL", HelperFeatureEnum::device_ext},
      {"CUDNN_BIDIRECTIONAL", HelperFeatureEnum::device_ext},
      {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED", HelperFeatureEnum::device_ext},
  };
}

} // namespace dpct
} // namespace clang