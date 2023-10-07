//===--------------- RewriterDeviceSegmentedSort.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createDeviceSegmentedSortRewriterMap() {
  return RewriterMap{
      // cub::DeviceSegmentedSort::SortKeys
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedSort::SortKeys")),
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
                                          8, std::greater_equal<>(),
                                          /* IncludeDefaultArg */ false),
                                      makeCheckNot(
                                          CheckArgIsDefaultCudaStream(7))),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedSort::SortKeys",
                                      CALL(
                                          "dpct::segmented_sort_keys",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               STREAM(7)),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), LITERAL("false"),
                                          LITERAL("true")))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedSort::SortKeys",
                                  CALL("dpct::segmented_sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       LITERAL("false"), LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(
                              makeCheckAnd(
                                  CheckArgCount(9, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  makeCheckNot(CheckArgIsDefaultCudaStream(8))),
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedSort::SortKeys",
                                  CALL("dpct::segmented_sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            STREAM(8)),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       ARG(7), LITERAL("false")))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedSort::SortKeys",
                              CALL("dpct::segmented_sort_keys",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), LITERAL("false"))))))))))))

      // cub::DeviceSegmentedSort::SortKeysDescending
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY(
                   "cub::DeviceSegmentedSort::SortKeysDescending")),
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
                                          8, std::greater_equal<>(),
                                          /* IncludeDefaultArg */ false),
                                      makeCheckNot(
                                          CheckArgIsDefaultCudaStream(7))),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedSort::"
                                      "SortKeysDescending",
                                      CALL(
                                          "dpct::segmented_sort_keys",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               STREAM(7)),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), LITERAL("true"),
                                          LITERAL("true")))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedSort::"
                                  "SortKeysDescending",
                                  CALL("dpct::segmented_sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       LITERAL("true"), LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(makeCheckAnd(
                                   CheckArgCount(9, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(8))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedSort::"
                                   "SortKeysDescending",
                                   CALL("dpct::segmented_sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(8)),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), LITERAL("true")))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedSort::SortKeysDescending",
                              CALL("dpct::segmented_sort_keys",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), LITERAL("true"))))))))))))

      // cub::DeviceSegmentedSort::SortPairs
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedSort::SortPairs")),
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
                                          9, std::greater_equal<>(),
                                          /* IncludeDefaultArg */ false),
                                      makeCheckNot(
                                          CheckArgIsDefaultCudaStream(8))),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedSort::SortPairs",
                                      CALL(
                                          MapNames::getDpctNamespace() +
                                              "segmented_sort_pairs",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               STREAM(8)),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), ARG(7), LITERAL("false"),
                                          LITERAL("true")))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedSort::SortPairs",
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
                                   CheckArgCount(11, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(10))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedSort::SortPairs",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(10)),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), ARG(8), ARG(9),
                                        LITERAL("false")))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedSort::SortPairs",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_sort_pairs",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9),
                                   LITERAL("false"))))))))))))

      // cub::DeviceSegmentedSort::SortPairsDescending
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY(
                   "cub::DeviceSegmentedSort::SortPairsDescending")),
          OTHERWISE(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(CheckParamType(2, "cub::DoubleBuffer"),
                           CASE_FACTORY_ENTRY(
                               CASE(makeCheckAnd(
                                        CheckArgCount(
                                            9, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                                        makeCheckNot(
                                            CheckArgIsDefaultCudaStream(8))),
                                    CALL_FACTORY_ENTRY(
                                        "cub::DeviceSegmentedSort::"
                                        "SortPairsDescending",
                                        CALL(MapNames::getDpctNamespace() +
                                                 "segmented_sort_pairs",
                                             CALL("oneapi::dpl::execution::"
                                                  "device_policy",
                                                  STREAM(8)),
                                             ARG(2), ARG(3), ARG(4), ARG(5),
                                             ARG(6), ARG(7), LITERAL("true"),
                                             LITERAL("true")))),
                               OTHERWISE(CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedSort::"
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
                                   CheckArgCount(11, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(10))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedSort::"
                                   "SortPairsDescending",
                                   CALL(MapNames::getDpctNamespace() +
                                            "segmented_sort_pairs",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(10)),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), ARG(8), ARG(9),
                                        LITERAL("true")))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedSort::SortPairsDescending",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_sort_pairs",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9),
                                   LITERAL("true"))))))))))))
      // cub::DeviceSegmentedSort::StableSortKeys
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY(
                   "cub::DeviceSegmentedSort::StableSortKeys")),
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
                                          8, std::greater_equal<>(),
                                          /* IncludeDefaultArg */ false),
                                      makeCheckNot(
                                          CheckArgIsDefaultCudaStream(7))),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedSort::"
                                      "StableSortKeys",
                                      CALL(
                                          "dpct::segmented_sort_keys",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               STREAM(7)),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), LITERAL("false"),
                                          LITERAL("true")))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedSort::StableSortKeys",
                                  CALL("dpct::segmented_sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       LITERAL("false"), LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(makeCheckAnd(
                                   CheckArgCount(9, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(8))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedSort::StableSortKeys",
                                   CALL("dpct::segmented_sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(8)),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), LITERAL("false")))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedSort::StableSortKeys",
                              CALL("dpct::segmented_sort_keys",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), LITERAL("false"))))))))))))

      // cub::DeviceSegmentedSort::StableSortKeysDescending
      CASE_FACTORY_ENTRY(
          CASE(CheckCubRedundantFunctionCall(),
               REMOVE_API_FACTORY_ENTRY(
                   "cub::DeviceSegmentedSort::StableSortKeysDescending")),
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
                                          8, std::greater_equal<>(),
                                          /* IncludeDefaultArg */ false),
                                      makeCheckNot(
                                          CheckArgIsDefaultCudaStream(7))),
                                  CALL_FACTORY_ENTRY(
                                      "cub::DeviceSegmentedSort::"
                                      "StableSortKeysDescending",
                                      CALL(
                                          "dpct::segmented_sort_keys",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               STREAM(7)),
                                          ARG(2), ARG(3), ARG(4), ARG(5),
                                          ARG(6), LITERAL("true"),
                                          LITERAL("true")))),
                              OTHERWISE(CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedSort::"
                                  "StableSortKeysDescending",
                                  CALL("dpct::segmented_sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       LITERAL("true"), LITERAL("true")))))),
                      OTHERWISE(CASE_FACTORY_ENTRY(
                          CASE(makeCheckAnd(
                                   CheckArgCount(9, std::greater_equal<>(),
                                                 /* IncludeDefaultArg */ false),
                                   makeCheckNot(
                                       CheckArgIsDefaultCudaStream(8))),
                               CALL_FACTORY_ENTRY(
                                   "cub::DeviceSegmentedSort::"
                                   "StableSortKeysDescending",
                                   CALL("dpct::segmented_sort_keys",
                                        CALL("oneapi::dpl::execution::device_"
                                             "policy",
                                             STREAM(8)),
                                        ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                        ARG(7), LITERAL("true")))),
                          OTHERWISE(CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedSort::"
                              "StableSortKeysDescending",
                              CALL("dpct::segmented_sort_keys",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), LITERAL("true"))))))))))))

      // cub::DeviceSegmentedSort::StableSortPairs
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedSort::StableSortPairs"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckArgCount(11, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedSort::StableSortPairs",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_sort_pairs",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(10)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9)))),
                      CASE(makeCheckAnd(
                               CheckArgCount(10, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CheckParamType(9, "_Bool")),
                           CALL_FACTORY_ENTRY(
                               "cub::DeviceSegmentedSort::StableSortPairs",
                               CALL(MapNames::getDpctNamespace() +
                                        "segmented_sort_pairs",
                                    CALL("oneapi::dpl::execution::device_"
                                         "policy",
                                         STREAM(8)),
                                    ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                    ARG(7), LITERAL("false"),
                                    LITERAL("true")))),
                      CASE(
                          CheckArgCount(10, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedSort::StableSortPairs",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_sort_pairs",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9)))),
                      CASE(
                          CheckArgCount(9, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedSort::StableSortPairs",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_sort_pairs",
                                   CALL("oneapi::dpl::execution::device_"
                                        "policy",
                                        STREAM(8)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), LITERAL("false"), LITERAL("true")))),
                      OTHERWISE(CALL_FACTORY_ENTRY(
                          "cub::DeviceSegmentedSort::StableSortPairs",
                          CALL(MapNames::getDpctNamespace() +
                                   "segmented_sort_pairs",
                               CALL("oneapi::dpl::execution::device_policy",
                                    QUEUESTR),
                               ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                               LITERAL("false"), LITERAL("true"))))))))))

      // cub::DeviceSegmentedSort::StableSortPairsDescending
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY(
              "cub::DeviceSegmentedSort::StableSortPairsDescending"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckArgCount(11, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedSort::"
                              "StableSortPairsDescending",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_sort_pairs",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(10)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9), LITERAL("true")))),
                      CASE(makeCheckAnd(
                               CheckArgCount(10, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               CheckParamType(9, "_Bool")),
                           CALL_FACTORY_ENTRY(
                               "cub::DeviceSegmentedSort::"
                               "StableSortPairsDescending",
                               CALL(MapNames::getDpctNamespace() +
                                        "segmented_sort_pairs",
                                    CALL("oneapi::dpl::execution::device_"
                                         "policy",
                                         STREAM(8)),
                                    ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                    ARG(7), LITERAL("true"), LITERAL("true")))),
                      CASE(
                          CheckArgCount(10, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceSegmentedSort::"
                              "StableSortPairsDescending",
                              CALL(MapNames::getDpctNamespace() +
                                       "segmented_sort_pairs",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9), LITERAL("true")))),
                      CASE(CheckArgCount(9, std::greater_equal<>(),
                                         /* IncludeDefaultArg */ false),
                           CALL_FACTORY_ENTRY(
                               "cub::DeviceSegmentedSort::"
                               "StableSortPairsDescending",
                               CALL(MapNames::getDpctNamespace() +
                                        "segmented_sort_pairs",
                                    CALL("oneapi::dpl::execution::device_"
                                         "policy",
                                         STREAM(8)),
                                    ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                    ARG(7), LITERAL("true"), LITERAL("true")))),
                      OTHERWISE(CALL_FACTORY_ENTRY(
                          "cub::DeviceSegmentedSort::StableSortPairsDescending",
                          CALL(MapNames::getDpctNamespace() +
                                   "segmented_sort_pairs",
                               CALL("oneapi::dpl::execution::device_policy",
                                    QUEUESTR),
                               ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                               LITERAL("true"), LITERAL("true"))))))))))};
}
