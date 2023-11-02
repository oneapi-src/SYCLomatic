//===--------------- RewriterDeviceRunLengthEncode.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createDeviceRunLengthEncodeRewriterMap() {
  return RewriterMap{
      // cub::DeviceRunLengthEncode::Encode
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceRunLengthEncode::Encode"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPCT_DPL_Utils,
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
                              "cub::DeviceRunLengthEncode::Encode",
                              MEMBER_CALL(
                                  ARG(7), true, "fill", ARG(5),
                                  CALL(
                                      "std::distance", ARG(3),
                                      MEMBER_EXPR(
                                          CALL(
                                              "oneapi::dpl::reduce_by_segment",
                                              CALL("oneapi::dpl::execution::"
                                                   "device_"
                                                   "policy",
                                                   STREAM(7)),
                                              ARG(2),
                                              BO(BinaryOperatorKind::BO_Add,
                                                 ARG(2), ARG(6)),
                                              CALL(
                                                  TEMPLATED_CALLEE_WITH_ARGS(
                                                      MapNames::
                                                              getDpctNamespace() +
                                                          "constant_iterator",
                                                      LITERAL("size_t")),
                                                  LITERAL("1")),
                                              ARG(3), ARG(4)),
                                          false, LITERAL("first"))),
                                  LITERAL("1")),
                              false, "wait"),
                          MEMBER_CALL_FACTORY_ENTRY(
                              "cub::DeviceRunLengthEncode::Encode",
                              MEMBER_CALL(
                                  QUEUESTR, false, "fill", ARG(5),
                                  CALL(
                                      "std::distance", ARG(3),
                                      MEMBER_EXPR(
                                          CALL(
                                              "oneapi::dpl::reduce_by_segment",
                                              CALL("oneapi::dpl::execution::"
                                                   "device_"
                                                   "policy",
                                                   QUEUESTR),
                                              ARG(2),
                                              BO(BinaryOperatorKind::BO_Add,
                                                 ARG(2), ARG(6)),
                                              CALL(
                                                  TEMPLATED_CALLEE_WITH_ARGS(
                                                      MapNames::
                                                              getDpctNamespace() +
                                                          "constant_iterator",
                                                      LITERAL("size_t")),
                                                  LITERAL("1")),
                                              ARG(3), ARG(4)),
                                          false, LITERAL("first"))),
                                  LITERAL("1")),
                              false, "wait")))))))

      // cub::DeviceRunLengthEncode::NonTrivialRuns
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY(
              "cub::DeviceRunLengthEncode::NonTrivialRuns"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPCT_DPL_Utils,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Execution,
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_DPL_Algorithm,
                      REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                          makeCheckAnd(
                              CheckArgCount(8, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceRunLengthEncode::"
                              "NonTrivialRuns",
                              CALL(MapNames::getDpctNamespace() +
                                       "nontrivial_run_length_encode",
                                   CALL("oneapi::dpl::execution::"
                                        "device_policy",
                                        STREAM(7)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceRunLengthEncode::"
                              "NonTrivialRuns",
                              CALL(MapNames::getDpctNamespace() +
                                       "nontrivial_run_length_encode",
                                   CALL("oneapi::dpl::execution::"
                                        "device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5),
                                   ARG(6)))))))))

  };
}
