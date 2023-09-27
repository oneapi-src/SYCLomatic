//===--------------- RewriterDeviceSelect.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createDeviceSelectRewriterMap() {
  return RewriterMap{
      // cub::DeviceSelect::Flagged
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSelect::Flagged"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPCT_DPL_Utils,
              REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                  makeCheckAnd(CheckArgCount(8, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                  MEMBER_CALL_FACTORY_ENTRY(
                      "cub::DeviceSelect::Flagged",
                      MEMBER_CALL(
                          ARG(7), true, "fill", ARG(5),
                          CALL(
                              "std::distance", ARG(4),
                              CALL(MapNames::getDpctNamespace() + "copy_if",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(7)),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(6)),
                                   ARG(3), ARG(4),
                                   LITERAL("[](const auto &t) -> bool { return "
                                           "t; }"))),
                          LITERAL("1")),
                      false, "wait"),
                  MEMBER_CALL_FACTORY_ENTRY(
                      "cub::DeviceSelect::Flagged",
                      MEMBER_CALL(
                          QUEUESTR, false, "fill", ARG(5),
                          CALL(
                              "std::distance", ARG(4),
                              CALL(MapNames::getDpctNamespace() + "copy_if",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(6)),
                                   ARG(3), ARG(4),
                                   LITERAL("[](const auto &t) -> bool { return "
                                           "t; }"))),
                          LITERAL("1")),
                      false, "wait")))))

      // cub::DeviceSelect::Unique
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSelect::Unique"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPL_Execution,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Algorithm,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(7, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(6))),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceSelect::Unique",
                          MEMBER_CALL(ARG(6), true, "fill", ARG(4),
                                      CALL("std::distance", ARG(3),
                                           CALL("oneapi::dpl::unique_copy",
                                                CALL("oneapi::dpl::execution::"
                                                     "device_policy",
                                                     STREAM(6)),
                                                ARG(2),
                                                BO(BinaryOperatorKind::BO_Add,
                                                   ARG(2), ARG(5)),
                                                ARG(3))),
                                      LITERAL("1")),
                          false, "wait"),
                      MEMBER_CALL_FACTORY_ENTRY(
                          "cub::DeviceSelect::Unique",
                          MEMBER_CALL(QUEUESTR, false, "fill", ARG(4),
                                      CALL("std::distance", ARG(3),
                                           CALL("oneapi::dpl::unique_copy",
                                                CALL("oneapi::dpl::execution::"
                                                     "device_policy",
                                                     QUEUESTR),
                                                ARG(2),
                                                BO(BinaryOperatorKind::BO_Add,
                                                   ARG(2), ARG(5)),
                                                ARG(3))),
                                      LITERAL("1")),
                          false, "wait"))))))

      // cub::DeviceSelect::UniqueByKey
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSelect::UniqueByKey"),
          FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Execution,
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_DPCT_DPL_Utils,
                      REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                          makeCheckAnd(
                              CheckArgCount(9, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              makeCheckNot(CheckArgIsDefaultCudaStream(8))),
                          MEMBER_CALL_FACTORY_ENTRY(
                              "cub::DeviceSelect::UniqueByKey",
                              MEMBER_CALL(
                                  ARG(8), true, "fill", ARG(6),
                                  CALL("std::distance", ARG(4),
                                       CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                                "std::get", LITERAL("0")),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "unique_copy",
                                                 CALL("oneapi::dpl::execution::"
                                                      "device_policy",
                                                      STREAM(8)),
                                                 ARG(2),
                                                 BO(BinaryOperatorKind::BO_Add,
                                                    ARG(2), ARG(7)),
                                                 ARG(3), ARG(4), ARG(5)))),
                                  LITERAL("1")),
                              false, "wait"),
                          MEMBER_CALL_FACTORY_ENTRY(
                              "cub::DeviceSelect::UniqueByKey",
                              MEMBER_CALL(
                                  QUEUESTR, false, "fill", ARG(6),
                                  CALL("std::distance", ARG(4),
                                       CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                                "std::get", LITERAL("0")),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "unique_copy",
                                                 CALL("oneapi::dpl::execution::"
                                                      "device_policy",
                                                      QUEUESTR),
                                                 ARG(2),
                                                 BO(BinaryOperatorKind::BO_Add,
                                                    ARG(2), ARG(7)),
                                                 ARG(3), ARG(4), ARG(5)))),
                                  LITERAL("1")),
                              false, "wait")))))))

      // cub::DeviceSelect::If
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSelect::If"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPCT_DPL_Utils,
              REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                  makeCheckAnd(CheckArgCount(8, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                  MEMBER_CALL_FACTORY_ENTRY(
                      "cub::DeviceSelect::If",
                      MEMBER_CALL(
                          ARG(7), true, "fill", ARG(4),
                          CALL(
                              "std::distance", ARG(3),
                              CALL("oneapi::dpl::copy_if",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(7)),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(5)),
                                   ARG(3), ARG(6))),
                          LITERAL("1")),
                      false, "wait"),
                  MEMBER_CALL_FACTORY_ENTRY(
                      "cub::DeviceSelect::If",
                      MEMBER_CALL(
                          QUEUESTR, false, "fill", ARG(4),
                          CALL(
                              "std::distance", ARG(3),
                              CALL("oneapi::dpl::copy_if",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(5)),
                                   ARG(3), ARG(6))),
                          LITERAL("1")),
                      false, "wait")))))

  };
}
