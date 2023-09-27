//===--------------- RewriterDeviceSegmentedRadixSort.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createDeviceSegmentedRadixSortRewriterMap() {
  return RewriterMap{
      // cub::DeviceSegmentedRadixSort::SortKeys
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY(
                   "cub::DeviceSegmentedRadixSort::SortKeys")),
          OTHERWISE(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckParamType(2, "cub::DoubleBuffer"),
                          CASE_FACTORY_ENTRY(
                              CASE(
                                  makeCheckAnd(
                                      CheckArgCount(
                                          10, std::greater_equal<>(),
                                          /* IncludeDefaultArg */ false),
                                      makeCheckNot(
                                          CheckArgIsDefaultCudaStream(9))),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::SortKeys",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_keys",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               STREAM(9)),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), LITERAL("false"),
                                          LITERAL("true"), ARG(7), ARG(8)))),
                              CASE(
                                  CheckArgCount(9, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::SortKeys",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_keys",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               QUEUESTR),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), LITERAL("false"),
                                          LITERAL("true"), ARG(7), ARG(8)))),
                              CASE(
                                  CheckArgCount(8, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::SortKeys",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_keys",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               QUEUESTR),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), LITERAL("false"),
                                          LITERAL("true"), ARG(7)))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedRadixSort::SortKeys",
                                  CALL(MapNames::getDpctNamespace() +
                                           "segmented_sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       LITERAL("false"), LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(makeCheckAnd(
                                   CheckArgCount(11, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(10))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::SortKeys",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(10)),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), LITERAL("false"), ARG(8),
                                        ARG(9)))),
                          CASE(CheckArgCount(10, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::SortKeys",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), LITERAL("false"), ARG(8),
                                        ARG(9)))),
                          CASE(CheckArgCount(9, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::SortKeys",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), LITERAL("false"), ARG(8)))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedRadixSort::SortKeys",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_sort_keys",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), LITERAL("false"))))))))))))

      // cub::DeviceSegmentedRadixSort::SortKeysDescending
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY(
                   "cub::DeviceSegmentedRadixSort::SortKeysDescending")),
          OTHERWISE(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckParamType(2, "cub::DoubleBuffer"),
                          CASE_FACTORY_ENTRY(
                              CASE(
                                  makeCheckAnd(
                                      CheckArgCount(
                                          10, std::greater_equal<>(),
                                          /* IncludeDefaultArg */ false),
                                      makeCheckNot(
                                          CheckArgIsDefaultCudaStream(9))),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::"
                                      "SortKeysDescending",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_keys",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               STREAM(9)),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), LITERAL("true"),
                                          LITERAL("true"), ARG(7), ARG(8)))),
                              CASE(
                                  CheckArgCount(9, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::"
                                      "SortKeysDescending",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_keys",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               QUEUESTR),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), LITERAL("true"),
                                          LITERAL("true"), ARG(7), ARG(8)))),
                              CASE(
                                  CheckArgCount(8, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::"
                                      "SortKeysDescending",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_keys",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               QUEUESTR),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), LITERAL("true"),
                                          LITERAL("true"), ARG(7)))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedRadixSort::"
                                  "SortKeysDescending",
                                  CALL(MapNames::getDpctNamespace() +
                                           "segmented_sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       LITERAL("true"), LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(makeCheckAnd(
                                   CheckArgCount(11, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(10))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::"
                                   "SortKeysDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(10)),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), LITERAL("true"), ARG(8),
                                        ARG(9)))),
                          CASE(CheckArgCount(10, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::"
                                   "SortKeysDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), LITERAL("true"), ARG(8),
                                        ARG(9)))),
                          CASE(CheckArgCount(9, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::"
                                   "SortKeysDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), LITERAL("true"), ARG(8)))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedRadixSort::"
                              "SortKeysDescending",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_sort_keys",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), LITERAL("true"))))))))))))

      // cub::DeviceSegmentedRadixSort::SortPairs
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY(
                   "cub::DeviceSegmentedRadixSort::SortPairs")),
          OTHERWISE(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckParamType(2, "cub::DoubleBuffer"),
                          CASE_FACTORY_ENTRY(
                              CASE(
                                  makeCheckAnd(
                                      CheckArgCount(
                                          11, std::greater_equal<>(),
                                          /* IncludeDefaultArg */ false),
                                      makeCheckNot(
                                          CheckArgIsDefaultCudaStream(10))),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::"
                                      "SortPairs",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_pairs",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               STREAM(10)),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), ARG(7), LITERAL("false"),
                                          LITERAL("true"), ARG(8), ARG(9)))),
                              CASE(
                                  CheckArgCount(10, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::"
                                      "SortPairs",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_pairs",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               QUEUESTR),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), ARG(7), LITERAL("false"),
                                          LITERAL("true"), ARG(8), ARG(9)))),
                              CASE(
                                  CheckArgCount(9, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::"
                                      "SortPairs",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_pairs",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               QUEUESTR),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), ARG(7), LITERAL("false"),
                                          LITERAL("true"), ARG(8)))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedRadixSort::SortPairs",
                                  CALL(MapNames::getDpctNamespace() +
                                           "segmented_sort_pairs",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       ARG(7), LITERAL("false"),
                                       LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(makeCheckAnd(
                                   CheckArgCount(13, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(12))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::SortPairs",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(12)),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), ARG(8), ARG(9),
                                        LITERAL("false"), ARG(10), ARG(11)))),
                          CASE(CheckArgCount(12, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::SortPairs",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), ARG(8), ARG(9),
                                        LITERAL("false"), ARG(10), ARG(11)))),
                          CASE(CheckArgCount(11, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::SortPairs",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), ARG(8), ARG(9),
                                        LITERAL("false"), ARG(10)))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedRadixSort::SortPairs",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_sort_pairs",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9),
                                   LITERAL("false"))))))))))))

      // cub::DeviceSegmentedRadixSort::SortPairsDescending
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY(
                   "cub::DeviceSegmentedRadixSort::SortPairsDescending")),
          OTHERWISE(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckParamType(2, "cub::DoubleBuffer"),
                          CASE_FACTORY_ENTRY(
                              CASE(
                                  makeCheckAnd(
                                      CheckArgCount(
                                          11, std::greater_equal<>(),
                                          /* IncludeDefaultArg */ false),
                                      makeCheckNot(
                                          CheckArgIsDefaultCudaStream(10))),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::"
                                      "SortPairsDescending",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_pairs",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               STREAM(10)),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), ARG(7), LITERAL("true"),
                                          LITERAL("true"), ARG(8), ARG(9)))),
                              CASE(
                                  CheckArgCount(10, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::"
                                      "SortPairsDescending",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_pairs",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               QUEUESTR),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), ARG(7), LITERAL("true"),
                                          LITERAL("true"), ARG(8), ARG(9)))),
                              CASE(
                                  CheckArgCount(9, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedRadixSort::"
                                      "SortPairsDescending",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_pairs",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               QUEUESTR),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), ARG(7), LITERAL("true"),
                                          LITERAL("true"), ARG(8)))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedRadixSort::"
                                  "SortPairsDescending",
                                  CALL(MapNames::getDpctNamespace() +
                                           "segmented_sort_pairs",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       ARG(7), LITERAL("true"),
                                       LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(makeCheckAnd(
                                   CheckArgCount(13, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(12))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::"
                                   "SortPairsDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(12)),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), ARG(8), ARG(9), LITERAL("true"),
                                        ARG(10), ARG(11)))),
                          CASE(CheckArgCount(12, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::"
                                   "SortPairsDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), ARG(8), ARG(9), LITERAL("true"),
                                        ARG(10), ARG(11)))),
                          CASE(CheckArgCount(11, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedRadixSort::"
                                   "SortPairsDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             QUEUESTR),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), ARG(8), ARG(9), LITERAL("true"),
                                        ARG(10)))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedRadixSort::"
                              "SortPairsDescending",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_sort_pairs",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9),
                                   LITERAL("true"))))))))))))

  };
}
