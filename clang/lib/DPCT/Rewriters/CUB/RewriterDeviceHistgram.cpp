//===--------------- RewriterDeviceHistgram.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createDeviceHistgramRewriterMap() {
  return RewriterMap{
      // cub::DeviceHistogram::HistogramEven
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceHistogram::HistogramEven"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckArgCount(11, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::HistogramEven",
                              CALL(MapNames::getDpctNamespace() +
                                       "histogram_even_roi",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(10)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9)))),
                      CASE(
                          CheckArgCount(10, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::HistogramEven",
                              CALL(MapNames::getDpctNamespace() +
                                       "histogram_even_roi",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9)))),
                      CASE(
                          CheckArgCount(9, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::HistogramEven",
                              CALL(MapNames::getDpctNamespace() +
                                       "histogram_even",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(8)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7)))),
                      OTHERWISE(CALL_FACTORY_ENTRY(
                          "cub::DeviceHistogram::HistogramEven",
                          CALL(MapNames::getDpctNamespace() + "histogram_even",
                               CALL("oneapi::dpl::execution::device_policy",
                                    QUEUESTR),
                               ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                               ARG(7))))))))))

      // cub::DeviceHistogram::MultiHistogramEven
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceHistogram::MultiHistogramEven"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckArgCount(11, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::MultiHistogramEven",
                              CALL(TEMPLATED_CALLEE(
                                       MapNames::getDpctNamespace() +
                                           "multi_histogram_even_roi",
                                       0, 1),
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(10)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9)))),
                      CASE(
                          CheckArgCount(10, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::MultiHistogramEven",
                              CALL(TEMPLATED_CALLEE(
                                       MapNames::getDpctNamespace() +
                                           "multi_histogram_even_roi",
                                       0, 1),
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8), ARG(9)))),
                      CASE(
                          CheckArgCount(9, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::MultiHistogramEven",
                              CALL(TEMPLATED_CALLEE(
                                       MapNames::getDpctNamespace() +
                                           "multi_histogram_even",
                                       0, 1),
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(8)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7)))),
                      OTHERWISE(CALL_FACTORY_ENTRY(
                          "cub::DeviceHistogram::MultiHistogramEven",
                          CALL(TEMPLATED_CALLEE(MapNames::getDpctNamespace() +
                                                    "multi_histogram_even",
                                                0, 1),
                               CALL("oneapi::dpl::execution::device_policy",
                                    QUEUESTR),
                               ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                               ARG(7))))))))))

      // cub::DeviceHistogram::HistogramRange
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceHistogram::HistogramRange"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckArgCount(10, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::HistogramRange",
                              CALL(MapNames::getDpctNamespace() +
                                       "histogram_range_roi",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(9)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8)))),
                      CASE(
                          CheckArgCount(9, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::HistogramRange",
                              CALL(MapNames::getDpctNamespace() +
                                       "histogram_range_roi",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8)))),
                      CASE(
                          CheckArgCount(8, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::HistogramRange",
                              CALL(MapNames::getDpctNamespace() +
                                       "histogram_range",
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(7)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))),
                      OTHERWISE(CALL_FACTORY_ENTRY(
                          "cub::DeviceHistogram::HistogramRange",
                          CALL(MapNames::getDpctNamespace() + "histogram_range",
                               CALL("oneapi::dpl::execution::device_policy",
                                    QUEUESTR),
                               ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))))))))))

      // cub::DeviceHistogram::MultiHistogramRange
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceHistogram::MultiHistogramRange"),
          REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_DPCT_DPL_Utils,
                  REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                      CASE(
                          CheckArgCount(10, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::MultiHistogramRange",
                              CALL(TEMPLATED_CALLEE(
                                       MapNames::getDpctNamespace() +
                                           "multi_histogram_range_roi",
                                       0, 1),
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(9)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8)))),
                      CASE(
                          CheckArgCount(9, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::MultiHistogramRange",
                              CALL(TEMPLATED_CALLEE(
                                       MapNames::getDpctNamespace() +
                                           "multi_histogram_range_roi",
                                       0, 1),
                                   CALL("oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                   ARG(7), ARG(8)))),
                      CASE(
                          CheckArgCount(8, std::greater_equal<>(),
                                        /* IncludeDefaultArg */ false),
                          CALL_FACTORY_ENTRY(
                              "cub::DeviceHistogram::MultiHistogramRange",
                              CALL(TEMPLATED_CALLEE(
                                       MapNames::getDpctNamespace() +
                                           "multi_histogram_range",
                                       0, 1),
                                   CALL("oneapi::dpl::execution::device_policy",
                                        STREAM(7)),
                                   ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))),
                      OTHERWISE(CALL_FACTORY_ENTRY(
                          "cub::DeviceHistogram::MultiHistogramRange",
                          CALL(TEMPLATED_CALLEE(MapNames::getDpctNamespace() +
                                                    "multi_histogram_range",
                                                0, 1),
                               CALL("oneapi::dpl::execution::device_policy",
                                    QUEUESTR),
                               ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))))))))))};
}
