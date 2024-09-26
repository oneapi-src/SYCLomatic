//===--------------- RewriterDevicePartition.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"
#include "CallExprRewriterCommon.h"

using namespace clang::dpct;

RewriterMap dpct::createDevicePartitionRewriterMap() {
  return RewriterMap{
      // cub::DevicePartition::Flagged
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DevicePartition::Flagged"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(HEADER_INSERT_FACTORY(
              HeaderType::HT_DPCT_DPL_Utils,
              REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                  makeCheckAnd(CheckArgCount(8, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                  CALL_FACTORY_ENTRY(
                      "cub::DevicePartition::Flagged",
                      CALL(MapNames::getLibraryHelperNamespace() + "partition_flagged",
                           CALL("oneapi::dpl::execution::device_policy",
                                STREAM(7)),
                           ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
                  CALL_FACTORY_ENTRY(
                      "cub::DevicePartition::Flagged",
                      CALL(MapNames::getLibraryHelperNamespace() + "partition_flagged",
                           CALL("oneapi::dpl::execution::device_policy",
                                QUEUESTR),
                           ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))))))))

      // cub::DevicePartition::If
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DevicePartition::If"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(HEADER_INSERT_FACTORY(
              HeaderType::HT_DPCT_DPL_Utils,
              REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                  CASE(CheckArgCount(11, std::greater_equal<>(),
                                     /* IncludeDefaultArg */ false),
                       CALL_FACTORY_ENTRY(
                           "cub::DevicePartition::If",
                           CALL(MapNames::getLibraryHelperNamespace() + "partition_if",
                                CALL("oneapi::dpl::execution::device_policy",
                                     STREAM(10)),
                                ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                                ARG(8), ARG(9), LITERAL("false")))),
                  CASE(CheckArgCount(10, std::greater_equal<>(),
                                     /* IncludeDefaultArg */ false),
                       CALL_FACTORY_ENTRY(
                           "cub::DevicePartition::If",
                           CALL(MapNames::getLibraryHelperNamespace() + "partition_if",
                                CALL("oneapi::dpl::execution::device_policy",
                                     QUEUESTR),
                                ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                                ARG(8), ARG(9), LITERAL("false")))),
                  CASE(CheckArgCount(8, std::greater_equal<>(),
                                     /* IncludeDefaultArg */ false),
                       CALL_FACTORY_ENTRY(
                           "cub::DevicePartition::If",
                           CALL(MapNames::getLibraryHelperNamespace() + "partition_if",
                                CALL("oneapi::dpl::execution::device_policy",
                                     STREAM(7)),
                                ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))),

                  OTHERWISE(CALL_FACTORY_ENTRY(
                      "cub::DevicePartition::If",
                      CALL(MapNames::getLibraryHelperNamespace() + "partition_if",
                           CALL("oneapi::dpl::execution::device_policy",
                                QUEUESTR),
                           ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))))

                      )))))

  };
}
