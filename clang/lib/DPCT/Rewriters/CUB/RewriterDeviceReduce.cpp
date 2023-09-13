//===--------------- RewriterDeviceReduce.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createDeviceReduceRewriterMap() {
  return RewriterMap{
      // cub::DeviceReduce::Sum
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::Sum"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPL_Execution,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Algorithm,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(6, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceReduce::Sum",
                          MEMBER_CALL(
                              ARG(5), true, "fill", ARG(3),
                              CALL("oneapi::dpl::reduce",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(5)),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(4)),
                                   ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                       TEMPLATED_NAME("std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                       LITERAL("value_type"))))),
                              LITERAL("1")),
                          false, "wait"),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceReduce::Sum",
                          MEMBER_CALL(
                              QUEUESTR, false, "fill", ARG(3),
                              CALL("oneapi::dpl::reduce",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(4)),
                                   ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                       TEMPLATED_NAME("std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                       LITERAL("value_type"))))),
                              LITERAL("1")),
                          false, "wait"))))))

      // cub::DeviceReduce::Min
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::Min"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPL_Execution,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Algorithm,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(6, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceReduce::Min",
                          MEMBER_CALL(
                              ARG(5), true, "fill", ARG(3),
                              CALL("oneapi::dpl::reduce",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(5)),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(4)),
                                   ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                       TEMPLATED_NAME("std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                       LITERAL("value_type")))),
                                   CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                       MapNames::getClNamespace() + "minimum",
                                       LITERAL("")))),
                              LITERAL("1")),
                          false, "wait"),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceReduce::Min",
                          MEMBER_CALL(
                              QUEUESTR, false, "fill", ARG(3),
                              CALL("oneapi::dpl::reduce",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(4)),
                                   ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                       TEMPLATED_NAME("std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                       LITERAL("value_type")))),
                                   CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                       MapNames::getClNamespace() + "minimum",
                                       LITERAL("")))),
                              LITERAL("1")),
                          false, "wait"))))))

      // cub::DeviceReduce::Max
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::Max"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPL_Execution,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Algorithm,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(6, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceReduce::Max",
                          MEMBER_CALL(
                              ARG(5), true, "fill", ARG(3),
                              CALL("oneapi::dpl::reduce",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(5)),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(4)),
                                   ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                       TEMPLATED_NAME("std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                       LITERAL("value_type")))),
                                   CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                       MapNames::getClNamespace() + "maximum",
                                       LITERAL("")))),
                              LITERAL("1")),
                          false, "wait"),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceReduce::Max",
                          MEMBER_CALL(
                              QUEUESTR, false, "fill", ARG(3),
                              CALL("oneapi::dpl::reduce",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(4)),
                                   ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                       TEMPLATED_NAME("std::iterator_traits",
                                                      CALL("decltype", ARG(3))),
                                       LITERAL("value_type")))),
                                   CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                       MapNames::getClNamespace() + "maximum",
                                       LITERAL("")))),
                              LITERAL("1")),
                          false, "wait"))))))

      // cub::DeviceReduce::ArgMin
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::ArgMin"),
          FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Execution,
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_DPCT_DPL_Utils,
                      REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                          makeCheckAnd(
                              CheckArgCount(6, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceReduce::ArgMin",
                              CALL(MapNames::getDpctNamespace() +
                                       "reduce_argmin",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(5)),
                                   ARG(2), ARG(3), ARG(4))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceReduce::ArgMin",
                              CALL(MapNames::getDpctNamespace() +
                                       "reduce_argmin",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4)))))))))

      // cub::DeviceReduce::ArgMax
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::ArgMax"),
          FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Execution,
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_DPCT_DPL_Utils,
                      REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                          makeCheckAnd(
                              CheckArgCount(6, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceReduce::ArgMax",
                              CALL(MapNames::getDpctNamespace() +
                                       "reduce_argmax",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(5)),
                                   ARG(2), ARG(3), ARG(4))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceReduce::ArgMax",
                              CALL(MapNames::getDpctNamespace() +
                                       "reduce_argmax",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4)))))))))

      // cub::DeviceReduce::Reduce
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::Reduce"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPL_Execution,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Algorithm,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(8, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceReduce::Reduce",
                          MEMBER_CALL(
                              ARG(7), true, "fill", ARG(3),
                              CALL("oneapi::dpl::reduce",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(7)),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(4)),
                                   ARG(6), ARG(5)),
                              LITERAL("1")),
                          false, "wait"),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceReduce::Reduce",
                          MEMBER_CALL(
                              QUEUESTR, false, "fill", ARG(3),
                              CALL("oneapi::dpl::reduce",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(4)),
                                   ARG(6), ARG(5)),
                              LITERAL("1")),
                          false, "wait"))))))

      // cub::DeviceReduceByKey
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::ReduceByKey"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPL_Execution,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Algorithm,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(10, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceReduce::ReduceByKey",
                          MEMBER_CALL(
                              ARG(9),
                              true, "fill", ARG(6),
                              CALL("std::distance", ARG(3),
                                   MEMBER_EXPR(
                                       CALL("oneapi::dpl::reduce_by_key",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 STREAM(9)),
                                            ARG(2),
                                            BO(BinaryOperatorKind::BO_Add,
                                               ARG(2), ARG(8)),
                                            ARG(4), ARG(3), ARG(5),
                                            CALL(TEMPLATED_NAME(
                                                "std::equal_to",
                                                TYPENAME(STATIC_MEMBER_EXPR(
                                                    TEMPLATED_NAME(
                                                        "std::iterator_traits",
                                                        CALL("decltype",
                                                             ARG(2))),
                                                    LITERAL("value_type"))))),
                                            ARG(7)),
                                       false, LITERAL("first"))),
                              LITERAL("1")),
                          false, "wait"),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceReduce::ReduceByKey",
                          MEMBER_CALL(
                              QUEUESTR, false, "fill", ARG(6),
                              CALL("std::distance", ARG(3),
                                   MEMBER_EXPR(
                                       CALL("oneapi::dpl::reduce_by_key",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 QUEUESTR),
                                            ARG(2),
                                            BO(BinaryOperatorKind::BO_Add,
                                               ARG(2), ARG(8)),
                                            ARG(4), ARG(3), ARG(5),
                                            CALL(TEMPLATED_NAME(
                                                "std::equal_to",
                                                TYPENAME(STATIC_MEMBER_EXPR(
                                                    TEMPLATED_NAME(
                                                        "std::iterator_traits",
                                                        CALL("decltype",
                                                             ARG(2))),
                                                    LITERAL("value_type"))))),
                                            ARG(7)),
                                       false, LITERAL("first"))),
                              LITERAL("1")),
                          false, "wait"))))))

  };
}
