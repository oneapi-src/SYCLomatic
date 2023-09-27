//===--------------- RewriterDeviceScan.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createDeviceScanRewriterMap() {
  return RewriterMap{
      // cub::DeviceScan::ExclusiveSum
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::ExclusiveSum"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPL_Execution,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Algorithm,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(6, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                      CALL_FACTORY_ENTRY(
                          "cub::DeviceScan::ExclusiveSum",
                          CALL("oneapi::dpl::exclusive_scan",
                               CALL("oneapi::dpl::execution::device_policy",
                                    STREAM(5)),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
                               ARG(3),
                               ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                   TEMPLATED_NAME("std::iterator_traits",
                                                  CALL("decltype", ARG(2))),
                                   LITERAL("value_type")))))),
                      CALL_FACTORY_ENTRY(
                          "cub::DeviceScan::ExclusiveSum",
                          CALL("oneapi::dpl::exclusive_scan",
                               CALL("oneapi::dpl::execution::device_policy",
                                    QUEUESTR),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
                               ARG(3),
                               ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                   TEMPLATED_NAME("std::iterator_traits",
                                                  CALL("decltype", ARG(2))),
                                   LITERAL("value_type")))))))))))

      // cub::DeviceScan::InclusiveSum
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::InclusiveSum"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPL_Execution,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Algorithm,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(6, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                      CALL_FACTORY_ENTRY(
                          "cub::DeviceScan::InclusiveSum",
                          CALL("oneapi::dpl::inclusive_scan",
                               CALL("oneapi::dpl::execution::device_policy",
                                    STREAM(5)),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
                               ARG(3))),
                      CALL_FACTORY_ENTRY(
                          "cub::DeviceScan::InclusiveSum",
                          CALL("oneapi::dpl::inclusive_scan",
                               CALL("oneapi::dpl::execution::device_policy",
                                    QUEUESTR),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
                               ARG(3))))))))

      // cub::DeviceScan::ExclusiveScan
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::ExclusiveScan"),
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
                          "cub::DeviceScan::ExclusiveScan",
                          CALL("oneapi::dpl::exclusive_scan",
                               CALL("oneapi::dpl::execution::device_policy",
                                    STREAM(7)),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
                               ARG(3), ARG(5), ARG(4))),
                      CALL_FACTORY_ENTRY(
                          "cub::DeviceScan::ExclusiveScan",
                          CALL("oneapi::dpl::exclusive_scan",
                               CALL("oneapi::dpl::execution::device_policy",
                                    QUEUESTR),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
                               ARG(3), ARG(5), ARG(4))))))))

      // cub::DeviceScan::InclusiveScan
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::InclusiveScan"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPL_Execution,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Algorithm,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(7, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(6))),
                      CALL_FACTORY_ENTRY(
                          "cub::DeviceScan::InclusiveScan",
                          CALL("oneapi::dpl::inclusive_scan",
                               CALL("oneapi::dpl::execution::device_policy",
                                    STREAM(6)),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
                               ARG(3), ARG(4))),
                      CALL_FACTORY_ENTRY(
                          "cub::DeviceScan::InclusiveScan",
                          CALL("oneapi::dpl::inclusive_scan",
                               CALL("oneapi::dpl::execution::device_policy",
                                    QUEUESTR),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
                               ARG(3), ARG(4))))))))
      // cub::DeviceScan::InclusiveScanByKey
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::InclusiveScanByKey"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPL_Execution,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Algorithm,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(9, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(8))),
                      CALL_FACTORY_ENTRY(
                          "cub::DeviceScan::InclusiveScanByKey",
                          CALL("oneapi::dpl::inclusive_scan_by_key",
                               CALL("oneapi::dpl::execution::device_policy",
                                    STREAM(8)),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
                               ARG(3), ARG(4), ARG(7), ARG(5))),
                      CONDITIONAL_FACTORY_ENTRY(
                          CheckArgCount(8, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceScan::InclusiveScanByKey",
                              CALL("oneapi::dpl::inclusive_scan_by_key",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(6)),
                                   ARG(3), ARG(4), ARG(7), ARG(5))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceScan::InclusiveScanByKey",
                              CALL("oneapi::dpl::inclusive_scan_by_key",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(6)),
                                   ARG(3), ARG(4), LITERAL("std::equal_to<>()"),
                                   ARG(5)))))))))

      // cub::DeviceScan::InclusiveSumByKey
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::InclusiveSumByKey"),
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
                          "cub::DeviceScan::InclusiveSumByKey",
                          CALL("oneapi::dpl::inclusive_scan_by_key",
                               CALL("oneapi::dpl::execution::device_policy",
                                    STREAM(7)),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
                               ARG(3), ARG(4), ARG(6))),
                      CONDITIONAL_FACTORY_ENTRY(
                          CheckArgCount(7, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceScan::InclusiveSumByKey",
                              CALL("oneapi::dpl::inclusive_scan_by_key",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(5)),
                                   ARG(3), ARG(4), ARG(6))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceScan::InclusiveSumByKey",
                              CALL("oneapi::dpl::inclusive_scan_by_key",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(5)),
                                   ARG(3), ARG(4)))))))))

      // cub::DeviceScan::ExclusiveScanByKey
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::ExclusiveScanByKey"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPL_Execution,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPL_Algorithm,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                      makeCheckAnd(
                          CheckArgCount(10, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                      CALL_FACTORY_ENTRY(
                          "cub::DeviceScan::ExclusiveScanByKey",
                          CALL("oneapi::dpl::exclusive_scan_by_key",
                               CALL("oneapi::dpl::execution::device_policy",
                                    STREAM(9)),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(7)),
                               ARG(3), ARG(4), ARG(6), ARG(8), ARG(5))),
                      CONDITIONAL_FACTORY_ENTRY(
                          CheckArgCount(9, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceScan::ExclusiveScanByKey",
                              CALL("oneapi::dpl::exclusive_scan_by_key",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(7)),
                                   ARG(3), ARG(4), ARG(6),
                                   LITERAL("std::equal_to<>()"), ARG(5))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceScan::ExclusiveScanByKey",
                              CALL("oneapi::dpl::exclusive_scan_by_key",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(7)),
                                   ARG(3), ARG(4), ARG(6),
                                   LITERAL("std::equal_to<>()"), ARG(5)))))))))

      // cub::DeviceScan::ExclusiveSumByKey
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::ExclusiveSumByKey"),
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
                          "cub::DeviceScan::ExclusiveSumByKey",
                          CALL("oneapi::dpl::exclusive_scan_by_key",
                               CALL("oneapi::dpl::execution::device_policy",
                                    STREAM(7)),
                               ARG(2),
                               BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
                               ARG(3), ARG(4),
                               ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                   TEMPLATED_NAME("std::iterator_traits",
                                                  CALL("decltype", ARG(2))),
                                   LITERAL("value_type")))),
                               ARG(6))),
                      CONDITIONAL_FACTORY_ENTRY(
                          CheckArgCount(7, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceScan::ExclusiveSumByKey",
                              CALL("oneapi::dpl::exclusive_scan_by_key",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(5)),
                                   ARG(3), ARG(4),
                                   ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
                                       TEMPLATED_NAME("std::iterator_traits",
                                                      CALL("decltype", ARG(2))),
                                       LITERAL("value_type")))),
                                   ARG(6))),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceScan::ExclusiveSumByKey",
                              CALL("oneapi::dpl::exclusive_scan_by_key",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2),
                                   BO(BinaryOperatorKind::BO_Add, ARG(2),
                                      ARG(5)),
                                   ARG(3), ARG(4)))))))))

  };
}
