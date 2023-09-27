//===--------------- RewriterDeviceRadixSort.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createDeviceRadixSortRewriterMap() {
  return RewriterMap{
      // cub::DeviceRadixSort::SortKeys
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY("cub::DeviceRadixSort::SortKeys")),
          OTHERWISE(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckParamType(2, "cub::DoubleBuffer"),
                          CASE_FACTORY_ENTRY(
                              CASE(makeCheckAnd(
                                       CheckArgCount(
                                           7, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                                       makeCheckNot(
                                           CheckArgIsDefaultCudaStream(6))),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::SortKeys",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_keys",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 STREAM(6)),
                                            ARG(2), ARG(3), LITERAL("false"),
                                            LITERAL("true"), ARG(4), ARG(5)))),
                              CASE(CheckArgCount(6, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::SortKeys",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_keys",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 QUEUESTR),
                                            ARG(2), ARG(3), LITERAL("false"),
                                            LITERAL("true"), ARG(4), ARG(5)))),
                              CASE(CheckArgCount(5, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::SortKeys",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_keys",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 QUEUESTR),
                                            ARG(2), ARG(3), LITERAL("false"),
                                            LITERAL("true"), ARG(4)))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceRadixSort::SortKeys",
                                  CALL(MapNames::getDpctNamespace() +
                                           "sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), LITERAL("false"),
                                       LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(
                              makeCheckAnd(
                                  CheckArgCount(8, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceRadixSort::SortKeys",
                                  CALL(MapNames::getDpctNamespace() +
                                           "sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            STREAM(7)),
                                       ARG(2), ARG(3), ARG(4), LITERAL("false"),
                                       ARG(5), ARG(6)))),
                          CASE(CheckArgCount(7, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceRadixSort::SortKeys",
                                   CALL(MapNames::getDpctNamespace() +
                                            "sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4),
                                        LITERAL("false"), ARG(5), ARG(6)))),
                          CASE(CheckArgCount(6, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceRadixSort::SortKeys",
                                   CALL(MapNames::getDpctNamespace() +
                                            "sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4),
                                        LITERAL("false"), ARG(5)))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceRadixSort::SortKeys",
                              CALL(MapNames::getDpctNamespace() + "sort_keys",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4),
                                   LITERAL("false"))))))))))))

      // cub::DeviceRadixSort::SortKeysDescending
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY(
                   "cub::DeviceRadixSort::SortKeysDescending")),

          OTHERWISE(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckParamType(2, "cub::DoubleBuffer"),
                          CASE_FACTORY_ENTRY(
                              CASE(makeCheckAnd(
                                       CheckArgCount(
                                           7, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                                       makeCheckNot(
                                           CheckArgIsDefaultCudaStream(6))),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::"
                                       "SortKeysDescending",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_keys",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 STREAM(6)),
                                            ARG(2), ARG(3), LITERAL("true"),
                                            LITERAL("true"), ARG(4), ARG(5)))),
                              CASE(CheckArgCount(6, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::"
                                       "SortKeysDescending",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_keys",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 QUEUESTR),
                                            ARG(2), ARG(3), LITERAL("true"),
                                            LITERAL("true"), ARG(4), ARG(5)))),
                              CASE(CheckArgCount(5, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::"
                                       "SortKeysDescending",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_keys",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 QUEUESTR),
                                            ARG(2), ARG(3), LITERAL("true"),
                                            LITERAL("true"), ARG(4)))),
                              OTHERWISE(MULTI_STMTS_FACTORY_ENTRY(
                                  "cub::DeviceRadixSort::SortKeysDescending",
                                  true, false, true, true,
                                  CALL(MapNames::getDpctNamespace() +
                                           "sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), LITERAL("true"),
                                       LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(makeCheckAnd(
                                   CheckArgCount(8, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(7))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceRadixSort::SortKeysDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(7)),
                                        ARG(2), ARG(3), ARG(4), LITERAL("true"),
                                        ARG(5), ARG(6)))),
                          CASE(CheckArgCount(7, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceRadixSort::SortKeysDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), LITERAL("true"),
                                        ARG(5), ARG(6)))),
                          CASE(CheckArgCount(6, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceRadixSort::SortKeysDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), LITERAL("true"),
                                        ARG(5)))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceRadixSort::SortKeysDescending",
                              CALL(MapNames::getDpctNamespace() + "sort_keys",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4),
                                   LITERAL("true"))))))))))))

      // cub::DeviceRadixSort::SortPairs
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY("cub::DeviceRadixSort::SortPairs")),
          OTHERWISE(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckParamType(2, "cub::DoubleBuffer"),
                          CASE_FACTORY_ENTRY(
                              CASE(makeCheckAnd(
                                       CheckArgCount(
                                           8, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                                       makeCheckNot(
                                           CheckArgIsDefaultCudaStream(7))),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::SortPairs",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_pairs",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 STREAM(7)),
                                            ARG(2), ARG(3), ARG(4),
                                            LITERAL("false"), LITERAL("true"),
                                            ARG(5), ARG(6)))),
                              CASE(CheckArgCount(7, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::SortPairs",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_pairs",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 QUEUESTR),
                                            ARG(2), ARG(3), ARG(4),
                                            LITERAL("false"), LITERAL("true"),
                                            ARG(5), ARG(6)))),
                              CASE(CheckArgCount(6, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::SortPairs",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_pairs",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 QUEUESTR),
                                            ARG(2), ARG(3), ARG(4),
                                            LITERAL("false"), LITERAL("true"),
                                            ARG(5)))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceRadixSort::SortPairs",
                                  CALL(MapNames::getDpctNamespace() +
                                           "sort_pairs",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), ARG(4), LITERAL("false"),
                                       LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(makeCheckAnd(
                                   CheckArgCount(10, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(9))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceRadixSort::SortPairs",
                                   CALL(MapNames::getDpctNamespace() +
                                            "sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(9)),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        LITERAL("false"), ARG(7), ARG(8)))),
                          CASE(CheckArgCount(9, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceRadixSort::SortPairs",
                                   CALL(MapNames::getDpctNamespace() +
                                            "sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        LITERAL("false"), ARG(7), ARG(8)))),
                          CASE(CheckArgCount(8, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceRadixSort::SortPairs",
                                   CALL(MapNames::getDpctNamespace() +
                                            "sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        LITERAL("false"), ARG(7)))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceRadixSort::SortPairs",
                              CALL(MapNames::getDpctNamespace() + "sort_pairs",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   LITERAL("false"))))))))))))

      // cub::DeviceRadixSort::SortPairsDescending
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY(
                   "cub::DeviceRadixSort::SortPairsDescending")),
          OTHERWISE(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckParamType(2, "cub::DoubleBuffer"),
                          CASE_FACTORY_ENTRY(
                              CASE(makeCheckAnd(
                                       CheckArgCount(
                                           8, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                                       makeCheckNot(
                                           CheckArgIsDefaultCudaStream(7))),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::"
                                       "SortPairsDescending",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_pairs",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 STREAM(7)),
                                            ARG(2), ARG(3), ARG(4),
                                            LITERAL("true"), LITERAL("true"),
                                            ARG(5), ARG(6)))),
                              CASE(CheckArgCount(7, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::"
                                       "SortPairsDescending",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_pairs",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 QUEUESTR),
                                            ARG(2), ARG(3), ARG(4),
                                            LITERAL("true"), LITERAL("true"),
                                            ARG(5), ARG(6)))),
                              CASE(CheckArgCount(6, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   CALL_FACTORY_ENTRY(
                                       "cub::DeviceRadixSort::"
                                       "SortPairsDescending",
                                       CALL(MapNames::getDpctNamespace() +
                                                "sort_pairs",
                                            CALL("oneapi::dpl::execution::"
                                                 "device_policy",
                                                 QUEUESTR),
                                            ARG(2), ARG(3), ARG(4),
                                            LITERAL("true"), LITERAL("true"),
                                            ARG(5)))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceRadixSort::SortPairsDescending",
                                  CALL(MapNames::getDpctNamespace() +
                                           "sort_pairs",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), ARG(4), LITERAL("true"),
                                       LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(makeCheckAnd(
                                   CheckArgCount(10, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(9))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceRadixSort::SortPairsDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(9)),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        LITERAL("true"), ARG(7), ARG(8)))),
                          CASE(CheckArgCount(9, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceRadixSort::SortPairsDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        LITERAL("true"), ARG(7), ARG(8)))),
                          CASE(CheckArgCount(8, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceRadixSort::SortPairsDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        LITERAL("true"), ARG(7)))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceRadixSort::SortPairsDescending",
                              CALL(MapNames::getDpctNamespace() + "sort_pairs",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   LITERAL("true"))))))))))))

  };
}
