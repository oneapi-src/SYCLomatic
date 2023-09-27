//===--------------- RewriterDeviceSegmentedReduce.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createDeviceSegmentedReduceRewriterMap() {
  return RewriterMap{
      // cub::DeviceSegmentedReduce::Reduce
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::Reduce"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(HEADER_INSERT_FACTORY(
              HeaderType::HT_DPCT_DPL_Utils,
              WARNING_FACTORY_ENTRY(
                  "cub::DeviceSegmentedReduce::Reduce",
                  CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(10, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                      CONDITIONAL_FACTORY_ENTRY(
                          checkEnableUserDefineReductions(),
                          FEATURE_REQUEST_FACTORY(
                              HelperFeatureEnum::device_ext,
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedReduce::Reduce",
                                  CALL(
                                      TEMPLATED_CALLEE_WITH_ARGS(
                                          MapNames::getDpctNamespace() +
                                              "device::experimental::segmented_"
                                              "reduce",
                                          LITERAL("128")),
                                      STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5),
                                      ARG(6), ARG(7), ARG(8)))),
                          FEATURE_REQUEST_FACTORY(
                              HelperFeatureEnum::device_ext,
                              CONDITIONAL_FACTORY_ENTRY(
                                  checkArgCanMappingToSyclNativeBinaryOp(7),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedReduce::Reduce",
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                               MapNames::getDpctNamespace() +
                                                   "device::segmented_reduce",
                                               LITERAL("128")),
                                           STREAM(9), ARG(2), ARG(3), ARG(4),
                                           ARG(5), ARG(6), ARG(7), ARG(8))),
                                  WARNING_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedReduce::Reduce",
                                      CALL_FACTORY_ENTRY(
                                          "cub::DeviceSegmentedReduce::Reduce",
                                          CALL(
                                              TEMPLATED_CALLEE_WITH_ARGS(
                                                  MapNames::getDpctNamespace() +
                                                      "device::segmented_"
                                                      "reduce",
                                                  LITERAL("128")),
                                              STREAM(9), ARG(2), ARG(3), ARG(4),
                                              ARG(5), ARG(6),
                                              LITERAL("dpct_placeholder"),
                                              ARG(8))),
                                      Diagnostics::
                                          UNSUPPORTED_BINARY_OPERATION)))),
                      CONDITIONAL_FACTORY_ENTRY(
                          checkEnableUserDefineReductions(),
                          FEATURE_REQUEST_FACTORY(
                              HelperFeatureEnum::device_ext,
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedReduce::Reduce",
                                  CALL(
                                      TEMPLATED_CALLEE_WITH_ARGS(
                                          MapNames::getDpctNamespace() +
                                              "device::experimental::segmented_"
                                              "reduce",
                                          LITERAL("128")),
                                      QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5),
                                      ARG(6), ARG(7), ARG(8)))),
                          FEATURE_REQUEST_FACTORY(
                              HelperFeatureEnum::device_ext,
                              CONDITIONAL_FACTORY_ENTRY(
                                  checkArgCanMappingToSyclNativeBinaryOp(7),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedReduce::Reduce",
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                               MapNames::getDpctNamespace() +
                                                   "device::segmented_reduce",
                                               LITERAL("128")),
                                           QUEUESTR, ARG(2), ARG(3), ARG(4),
                                           ARG(5), ARG(6), ARG(7), ARG(8))),
                                  WARNING_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedReduce::Reduce",
                                      CALL_FACTORY_ENTRY(
                                          "cub::DeviceSegmentedReduce::Reduce",
                                          CALL(
                                              TEMPLATED_CALLEE_WITH_ARGS(
                                                  MapNames::getDpctNamespace() +
                                                      "device::segmented_"
                                                      "reduce",
                                                  LITERAL("128")),
                                              QUEUESTR, ARG(2), ARG(3), ARG(4),
                                              ARG(5), ARG(6),
                                              LITERAL("dpct_placeholder"),
                                              ARG(8))),
                                      Diagnostics::
                                          UNSUPPORTED_BINARY_OPERATION))))),
                  Diagnostics::REDUCE_PERFORMANCE_TUNE))))

      // cub::DeviceSegmentedReduce::Sum
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::Sum"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  WARNING_FACTORY_ENTRY(
                      "cub::DeviceSegmentedReduce::Sum",
                      CONDITIONAL_FACTORY_ENTRY(
                          makeCheckAnd(
                              CheckArgCount(10, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedReduce::Sum",
                              CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                       MapNames::getDpctNamespace() +
                                           "device::segmented_reduce",
                                       LITERAL("128")),
                                   STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5),
                                   ARG(6),
                                   CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                       MapNames::getClNamespace() + "plus",
                                       LITERAL(""))),
                                   ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                       TEMPLATED_NAME("std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                       LITERAL("value_type")))))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedReduce::Sum",
                              CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                       MapNames::getDpctNamespace() +
                                           "device::segmented_reduce",
                                       LITERAL("128")),
                                   QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5),
                                   ARG(6),
                                   CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                       MapNames::getClNamespace() + "plus",
                                       LITERAL(""))),
                                   ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                       TEMPLATED_NAME("std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                       LITERAL("value_type"))))))),
                      Diagnostics::REDUCE_PERFORMANCE_TUNE)))))

      // cub::DeviceSegmentedReduce::Min
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::Min"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_Limits,
                      WARNING_FACTORY_ENTRY(
                          "cub::DeviceSegmentedReduce::Min",
                          CONDITIONAL_FACTORY_ENTRY(
                              makeCheckAnd(
                                  CheckArgCount(10, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedReduce::Min",
                                  CALL(
                                      TEMPLATED_CALLEE_WITH_ARGS(
                                          MapNames::getDpctNamespace() +
                                              "device::segmented_reduce",
                                          LITERAL("128")),
                                      STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5),
                                      ARG(6),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                          MapNames::getClNamespace() +
                                              "minimum",
                                          LITERAL(""))),
                                      CALL(STATIC_MEMBER_EXPR(
                                          TEMPLATED_NAME(
                                              "std::numeric_limits",
                                              TYPENAME(STATIC_MEMBER_EXPR(
                                                  TEMPLATED_NAME(
                                                      "std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                                  LITERAL("value_type")))),
                                          LITERAL("max"))))),
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedReduce::Min",
                                  CALL(
                                      TEMPLATED_CALLEE_WITH_ARGS(
                                          MapNames::getDpctNamespace() +
                                              "device::segmented_reduce",
                                          LITERAL("128")),
                                      QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5),
                                      ARG(6),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                          MapNames::getClNamespace() +
                                              "minimum",
                                          LITERAL(""))),
                                      CALL(STATIC_MEMBER_EXPR(
                                          TEMPLATED_NAME(
                                              "std::numeric_limits",
                                              TYPENAME(STATIC_MEMBER_EXPR(
                                                  TEMPLATED_NAME(
                                                      "std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                                  LITERAL("value_type")))),
                                          LITERAL("max")))))),
                          Diagnostics::REDUCE_PERFORMANCE_TUNE))))))

      // cub::DeviceSegmentedReduce::Max
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::Max"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_Limits,
                      WARNING_FACTORY_ENTRY(
                          "cub::DeviceSegmentedReduce::Max",
                          CONDITIONAL_FACTORY_ENTRY(
                              makeCheckAnd(
                                  CheckArgCount(10, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedReduce::Max",
                                  CALL(
                                      TEMPLATED_CALLEE_WITH_ARGS(
                                          MapNames::getDpctNamespace() +
                                              "device::segmented_reduce",
                                          LITERAL("128")),
                                      STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5),
                                      ARG(6),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                          MapNames::getClNamespace() +
                                              "maximum",
                                          LITERAL(""))),
                                      CALL(STATIC_MEMBER_EXPR(
                                          TEMPLATED_NAME(
                                              "std::numeric_limits",
                                              TYPENAME(STATIC_MEMBER_EXPR(
                                                  TEMPLATED_NAME(
                                                      "std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                                  LITERAL("value_type")))),
                                          LITERAL("lowest"))))),
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedReduce::Max",
                                  CALL(
                                      TEMPLATED_CALLEE_WITH_ARGS(
                                          MapNames::getDpctNamespace() +
                                              "device::segmented_reduce",
                                          LITERAL("128")),
                                      QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5),
                                      ARG(6),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                          MapNames::getClNamespace() +
                                              "maximum",
                                          LITERAL(""))),
                                      CALL(STATIC_MEMBER_EXPR(
                                          TEMPLATED_NAME(
                                              "std::numeric_limits",
                                              TYPENAME(STATIC_MEMBER_EXPR(
                                                  TEMPLATED_NAME(
                                                      "std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                                  LITERAL("value_type")))),
                                          LITERAL("lowest")))))),
                          Diagnostics::REDUCE_PERFORMANCE_TUNE))))))

      // cub::DeviceSegmentedReduce::ArgMin
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::ArgMin"),
          FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Execution,
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_DPCT_DPL_Utils,
                      REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                          makeCheckAnd(
                              CheckArgCount(8, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedReduce::ArgMin",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_reduce_argmin",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(7)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedReduce::ArgMin",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_reduce_argmin",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5),
                                   ARG(6)))))))))

      // cub::DeviceSegmentedReduce::ArgMax
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::ArgMax"),
          FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Execution,
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_DPCT_DPL_Utils,
                      REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                          makeCheckAnd(
                              CheckArgCount(8, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedReduce::ArgMax",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_reduce_argmax",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(7)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedReduce::ArgMax",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_reduce_argmax",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5),
                                   ARG(6)))))))))

  };
}
