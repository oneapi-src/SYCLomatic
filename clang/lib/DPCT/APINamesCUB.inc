//===------------------------- APINamesCUB.inc ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off

// cub::DeviceReduce::Sum
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::Sum"),
    HEADER_INSERT_FACTORY(
        HeaderType::HT_DPL_Execution,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPL_Algorithm,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                makeCheckAnd(CheckArgCount(6, std::greater_equal<>(), /* IncludeDefaultArg */false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                MEMBER_CALL_FACTORY_ENTRY(
                    "cub::DeviceReduce::Sum",
                    MEMBER_CALL(
                        ARG(5), true, "fill", ARG(3),
                        CALL("oneapi::dpl::reduce",
                             CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(5)),
                             ARG(2),
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
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
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
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
                makeCheckAnd(CheckArgCount(6, std::greater_equal<>(),
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
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
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
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
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
                makeCheckAnd(CheckArgCount(6, std::greater_equal<>(),
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
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
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
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
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
        HelperFeatureEnum::DplExtrasAlgorithm_reduce_argmin,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPL_Execution,
            HEADER_INSERT_FACTORY(
                HeaderType::HT_DPCT_DPL_Utils,
                REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                    makeCheckAnd(CheckArgCount(6, std::greater_equal<>(),
                                               /* IncludeDefaultArg */ false),
                                 makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                    CALL_FACTORY_ENTRY(
                        "cub::DeviceReduce::ArgMin",
                        CALL(MapNames::getDpctNamespace() + "reduce_argmin",
                             CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(5)),
                             ARG(2), ARG(3), ARG(4))),
                    CALL_FACTORY_ENTRY(
                        "cub::DeviceReduce::ArgMin",
                        CALL(MapNames::getDpctNamespace() + "reduce_argmin",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4)))))))))

// cub::DeviceReduce::ArgMax
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::ArgMax"),
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasAlgorithm_reduce_argmax,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPL_Execution,
            HEADER_INSERT_FACTORY(
                HeaderType::HT_DPCT_DPL_Utils,
                REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                    makeCheckAnd(CheckArgCount(6, std::greater_equal<>(),
                                               /* IncludeDefaultArg */ false),
                                 makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                    CALL_FACTORY_ENTRY(
                        "cub::DeviceReduce::ArgMax",
                        CALL(MapNames::getDpctNamespace() + "reduce_argmax",
                             CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(5)),
                             ARG(2), ARG(3), ARG(4))),
                    CALL_FACTORY_ENTRY(
                        "cub::DeviceReduce::ArgMax",
                        CALL(MapNames::getDpctNamespace() + "reduce_argmax",
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
                makeCheckAnd(CheckArgCount(8, std::greater_equal<>(), /* IncludeDefaultArg */false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                MEMBER_CALL_FACTORY_ENTRY(
                    "cub::DeviceReduce::Reduce",
                    MEMBER_CALL(
                        ARG(7), true, "fill", ARG(3),
                        CALL("oneapi::dpl::reduce",
                             CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(7)),
                             ARG(2),
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
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
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
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
                makeCheckAnd(CheckArgCount(10, std::greater_equal<>(), /* IncludeDefaultArg */false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                MEMBER_CALL_FACTORY_ENTRY(
                    "cub::DeviceReduce::ReduceByKey",
                    MEMBER_CALL(
                        ARG(9), true, "fill", ARG(6),
                        CALL(
                            "std::distance", ARG(3),
                            MEMBER_EXPR(
                                CALL(
                                    "oneapi::dpl::reduce_by_key",
                                    CALL(
                                        "oneapi::dpl::execution::device_policy",
                                        STREAM(9)),
                                    ARG(2),
                                    BO(BinaryOperatorKind::BO_Add, ARG(2),
                                       ARG(8)),
                                    ARG(4), ARG(3), ARG(5),
                                    CALL(TEMPLATED_NAME(
                                        "std::equal_to",
                                        TYPENAME(STATIC_MEMBER_EXPR(
                                            TEMPLATED_NAME(
                                                "std::iterator_traits",
                                                CALL("decltype", ARG(2))),
                                            LITERAL("value_type"))))),
                                    ARG(7)),
                                false, LITERAL("first"))),
                        LITERAL("1")),
                    false, "wait"),
                MEMBER_CALL_FACTORY_ENTRY(
                    "cub::DeviceReduce::ReduceByKey",
                    MEMBER_CALL(
                        QUEUESTR, false, "fill", ARG(6),
                        CALL(
                            "std::distance", ARG(3),
                            MEMBER_EXPR(
                                CALL(
                                    "oneapi::dpl::reduce_by_key",
                                    CALL(
                                        "oneapi::dpl::execution::device_policy",
                                        QUEUESTR),
                                    ARG(2),
                                    BO(BinaryOperatorKind::BO_Add, ARG(2),
                                       ARG(8)),
                                    ARG(4), ARG(3), ARG(5),
                                    CALL(TEMPLATED_NAME(
                                        "std::equal_to",
                                        TYPENAME(STATIC_MEMBER_EXPR(
                                            TEMPLATED_NAME(
                                                "std::iterator_traits",
                                                CALL("decltype", ARG(2))),
                                            LITERAL("value_type"))))),
                                    ARG(7)),
                                false, LITERAL("first"))),
                        LITERAL("1")),
                    false, "wait"))))))

// cub::DeviceScan::ExclusiveSum
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::ExclusiveSum"),
    HEADER_INSERT_FACTORY(
        HeaderType::HT_DPL_Execution,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPL_Algorithm,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                makeCheckAnd(CheckArgCount(6, std::greater_equal<>(), /* IncludeDefaultArg */false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                CALL_FACTORY_ENTRY(
                    "cub::DeviceScan::ExclusiveSum",
                    CALL("oneapi::dpl::exclusive_scan",
                         CALL("oneapi::dpl::execution::device_policy",
                              STREAM(5)),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
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
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
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
                makeCheckAnd(CheckArgCount(6, std::greater_equal<>(), /* IncludeDefaultArg */false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(5))),
                CALL_FACTORY_ENTRY(
                    "cub::DeviceScan::InclusiveSum",
                    CALL("oneapi::dpl::inclusive_scan",
                         CALL("oneapi::dpl::execution::device_policy",
                              STREAM(5)),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
                         ARG(3))),
                CALL_FACTORY_ENTRY(
                    "cub::DeviceScan::InclusiveSum",
                    CALL("oneapi::dpl::inclusive_scan",
                         CALL("oneapi::dpl::execution::device_policy",
                              QUEUESTR),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
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
                makeCheckAnd(CheckArgCount(8, std::greater_equal<>(), /* IncludeDefaultArg */false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                CALL_FACTORY_ENTRY(
                    "cub::DeviceScan::ExclusiveScan",
                    CALL("oneapi::dpl::exclusive_scan",
                         CALL("oneapi::dpl::execution::device_policy",
                              STREAM(7)),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
                         ARG(3), ARG(5), ARG(4))),
                CALL_FACTORY_ENTRY(
                    "cub::DeviceScan::ExclusiveScan",
                    CALL("oneapi::dpl::exclusive_scan",
                         CALL("oneapi::dpl::execution::device_policy",
                              QUEUESTR),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
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
                makeCheckAnd(CheckArgCount(7, std::greater_equal<>(), /* IncludeDefaultArg */false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(6))),
                CALL_FACTORY_ENTRY(
                    "cub::DeviceScan::InclusiveScan",
                    CALL("oneapi::dpl::inclusive_scan",
                         CALL("oneapi::dpl::execution::device_policy",
                              STREAM(6)),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
                         ARG(3), ARG(4))),
                CALL_FACTORY_ENTRY(
                    "cub::DeviceScan::InclusiveScan",
                    CALL("oneapi::dpl::inclusive_scan",
                         CALL("oneapi::dpl::execution::device_policy",
                              QUEUESTR),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
                         ARG(3), ARG(4))))))))

// cub::DeviceSelect::Flagged
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceSelect::Flagged"),
    HEADER_INSERT_FACTORY(
        HeaderType::HT_DPCT_DPL_Utils,
        REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
            makeCheckAnd(CheckArgCount(8, std::greater_equal<>(), /* IncludeDefaultArg */false),
                         makeCheckNot(CheckArgIsDefaultCudaStream(7))),
            MEMBER_CALL_FACTORY_ENTRY(
                "cub::DeviceSelect::Flagged",
                MEMBER_CALL(
                    ARG(7), true, "fill", ARG(5),
                    CALL("std::distance", ARG(4),
                         CALL(MapNames::getDpctNamespace() + "copy_if",
                              CALL("oneapi::dpl::execution::device_policy",
                                   STREAM(7)),
                              ARG(2),
                              BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
                              ARG(3), ARG(4),
                              LITERAL(
                                  "[](const auto &t) -> bool { return t; }"))),
                    LITERAL("1")),
                false, "wait"),
            MEMBER_CALL_FACTORY_ENTRY(
                "cub::DeviceSelect::Flagged",
                MEMBER_CALL(
                    QUEUESTR, false, "fill", ARG(5),
                    CALL("std::distance", ARG(4),
                         CALL(MapNames::getDpctNamespace() + "copy_if",
                              CALL("oneapi::dpl::execution::device_policy",
                                   QUEUESTR),
                              ARG(2),
                              BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
                              ARG(3), ARG(4),
                              LITERAL(
                                  "[](const auto &t) -> bool { return t; }"))),
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
                makeCheckAnd(CheckArgCount(7, std::greater_equal<>(), /* IncludeDefaultArg */false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(6))),
                MEMBER_CALL_FACTORY_ENTRY(
                    "cub::DeviceSelect::Unique",
                    MEMBER_CALL(
                        ARG(6), true, "fill", ARG(4),
                        CALL("std::distance", ARG(3),
                             CALL("oneapi::dpl::unique_copy",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       STREAM(6)),
                                  ARG(2),
                                  BO(BinaryOperatorKind::BO_Add, ARG(2),
                                     ARG(5)),
                                  ARG(3))),
                        LITERAL("1")),
                    false, "wait"),
                MEMBER_CALL_FACTORY_ENTRY(
                    "cub::DeviceSelect::Unique",
                    MEMBER_CALL(
                        QUEUESTR, false, "fill", ARG(4),
                        CALL("std::distance", ARG(3),
                             CALL("oneapi::dpl::unique_copy",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       QUEUESTR),
                                  ARG(2),
                                  BO(BinaryOperatorKind::BO_Add, ARG(2),
                                     ARG(5)),
                                  ARG(3))),     
                        LITERAL("1")),
                    false, "wait"))))))

// cub::DeviceSelect::UniqueByKey
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceSelect::UniqueByKey"),
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasAlgorithm_unique_copy,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPL_Execution,
            HEADER_INSERT_FACTORY(
                HeaderType::HT_DPCT_DPL_Utils,
                REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                    makeCheckAnd(CheckArgCount(9, std::greater_equal<>(),
                                               /* IncludeDefaultArg */ false),
                                 makeCheckNot(CheckArgIsDefaultCudaStream(8))),
                    MEMBER_CALL_FACTORY_ENTRY(
                        "cub::DeviceSelect::UniqueByKey",
                        MEMBER_CALL(
                            ARG(8), true, "fill", ARG(6),
                            CALL("std::distance", ARG(4),
                                 CALL(TEMPLATED_CALLEE_WITH_ARGS("std::get",
                                                                 LITERAL("0")),
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
                                 CALL(TEMPLATED_CALLEE_WITH_ARGS("std::get",
                                                                 LITERAL("0")),
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
                    CALL("std::distance", ARG(3),
                         CALL("oneapi::dpl::copy_if",
                              CALL("oneapi::dpl::execution::device_policy",
                                   STREAM(7)),
                              ARG(2),
                              BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
                              ARG(3), ARG(6))),
                    LITERAL("1")),
                false, "wait"),
            MEMBER_CALL_FACTORY_ENTRY(
                "cub::DeviceSelect::If",
                MEMBER_CALL(
                    QUEUESTR, false, "fill", ARG(4),
                    CALL("std::distance", ARG(3),
                         CALL("oneapi::dpl::copy_if",
                              CALL("oneapi::dpl::execution::device_policy",
                                   QUEUESTR),
                              ARG(2),
                              BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
                              ARG(3), ARG(6))),
                    LITERAL("1")),
                false, "wait")))))

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
                    makeCheckAnd(CheckArgCount(8, std::greater_equal<>(), /* IncludeDefaultArg */false),
                                 makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                    MEMBER_CALL_FACTORY_ENTRY(
                        "cub::DeviceRunLengthEncode::Encode",
                        MEMBER_CALL(
                            ARG(7), true, "fill", ARG(5),
                            CALL("std::distance", ARG(3),
                                 MEMBER_EXPR(
                                     CALL("oneapi::dpl::reduce_by_segment",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               STREAM(7)),
                                          ARG(2),
                                          BO(BinaryOperatorKind::BO_Add, ARG(2),
                                             ARG(6)),
                                          CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                                   MapNames::getDpctNamespace() + "constant_iterator",
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
                            CALL("std::distance", ARG(3),
                                 MEMBER_EXPR(
                                     CALL("oneapi::dpl::reduce_by_segment",
                                          CALL("oneapi::dpl::execution::device_"
                                               "policy",
                                               QUEUESTR),
                                          ARG(2),
                                          BO(BinaryOperatorKind::BO_Add, ARG(2),
                                             ARG(6)),
                                          CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                                   MapNames::getDpctNamespace() + "constant_iterator",
                                                   LITERAL("size_t")),
                                               LITERAL("1")),
                                          ARG(3), ARG(4)),
                                     false, LITERAL("first"))),
                            LITERAL("1")),
                        false, "wait")))))))

// cub::DeviceSegmentedReduce::Reduce
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::Reduce"),
    REMOVE_CUB_TEMP_STORAGE_FACTORY(HEADER_INSERT_FACTORY(
        HeaderType::HT_DPCT_DPL_Utils,
        WARNING_FACTORY_ENTRY(
            "cub::DeviceSegmentedReduce::Reduce",
            CONDITIONAL_FACTORY_ENTRY(
                makeCheckAnd(CheckArgCount(10, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                CONDITIONAL_FACTORY_ENTRY(
                    checkEnableUserDefineReductions(),
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::
                            DplExtrasDpcppExtensions_segmented_reduce_ext,
                        CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedReduce::Reduce",
                            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                     MapNames::getDpctNamespace() +
                                         "device::experimental::segmented_"
                                         "reduce",
                                     LITERAL("128")),
                                 STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5),
                                 ARG(6), ARG(7), ARG(8)))),
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::
                            DplExtrasDpcppExtensions_segmented_reduce,
                        CONDITIONAL_FACTORY_ENTRY(
                            checkArgCanMappingToSyclNativeBinaryOp(7),
                            CALL_FACTORY_ENTRY(
                                "cub::DeviceSegmentedReduce::Reduce",
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                         MapNames::getDpctNamespace() +
                                             "device::segmented_reduce",
                                         LITERAL("128")),
                                     STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5),
                                     ARG(6), ARG(7), ARG(8))),
                            WARNING_FACTORY_ENTRY(
                                "cub::DeviceSegmentedReduce::Reduce",
                                CALL_FACTORY_ENTRY(
                                    "cub::DeviceSegmentedReduce::Reduce",
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                             MapNames::getDpctNamespace() +
                                                 "device::segmented_reduce",
                                             LITERAL("128")),
                                         STREAM(9), ARG(2), ARG(3), ARG(4),
                                         ARG(5), ARG(6),
                                         LITERAL("dpct_placeholder"), ARG(8))),
                                Diagnostics::UNSUPPORTED_BINARY_OPERATION)))),
                CONDITIONAL_FACTORY_ENTRY(
                    checkEnableUserDefineReductions(),
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::
                            DplExtrasDpcppExtensions_segmented_reduce_ext,
                        CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedReduce::Reduce",
                            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                     MapNames::getDpctNamespace() +
                                         "device::experimental::segmented_"
                                         "reduce",
                                     LITERAL("128")),
                                 QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5),
                                 ARG(6), ARG(7), ARG(8)))),
                    FEATURE_REQUEST_FACTORY(
                        HelperFeatureEnum::
                            DplExtrasDpcppExtensions_segmented_reduce,
                        CONDITIONAL_FACTORY_ENTRY(
                            checkArgCanMappingToSyclNativeBinaryOp(7),
                            CALL_FACTORY_ENTRY(
                                "cub::DeviceSegmentedReduce::Reduce",
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                         MapNames::getDpctNamespace() +
                                             "device::segmented_reduce",
                                         LITERAL("128")),
                                     QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5),
                                     ARG(6), ARG(7), ARG(8))),
                            WARNING_FACTORY_ENTRY(
                                "cub::DeviceSegmentedReduce::Reduce",
                                CALL_FACTORY_ENTRY(
                                    "cub::DeviceSegmentedReduce::Reduce",
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                             MapNames::getDpctNamespace() +
                                                 "device::segmented_reduce",
                                             LITERAL("128")),
                                         QUEUESTR, ARG(2), ARG(3), ARG(4),
                                         ARG(5), ARG(6),
                                         LITERAL("dpct_placeholder"), ARG(8))),
                                Diagnostics::UNSUPPORTED_BINARY_OPERATION))))),
            Diagnostics::REDUCE_PERFORMANCE_TUNE))))

// cub::DeviceSegmentedReduce::Sum
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::Sum"),
    REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasDpcppExtensions_segmented_reduce,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            WARNING_FACTORY_ENTRY(
                "cub::DeviceSegmentedReduce::Sum",
                CONDITIONAL_FACTORY_ENTRY(
                    makeCheckAnd(CheckArgCount(10, std::greater_equal<>(), /* IncludeDefaultArg */false),
                                 makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                    CALL_FACTORY_ENTRY(
                        "cub::DeviceSegmentedReduce::Sum",
                        CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                 MapNames::getDpctNamespace() +
                                     "device::segmented_reduce",
                                 LITERAL("128")),
                             STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
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
                             QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
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
        HelperFeatureEnum::DplExtrasDpcppExtensions_segmented_reduce,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            HEADER_INSERT_FACTORY(
                HeaderType::HT_Limits,
                WARNING_FACTORY_ENTRY(
                    "cub::DeviceSegmentedReduce::Min",
                    CONDITIONAL_FACTORY_ENTRY(
                        makeCheckAnd(
                            CheckArgCount(10, std::greater_equal<>(), /* IncludeDefaultArg */false),
                            makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                        CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedReduce::Min",
                            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                     MapNames::getDpctNamespace() +
                                         "device::segmented_reduce",
                                     LITERAL("128")),
                                 STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5),
                                 ARG(6),
                                 CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                     MapNames::getClNamespace() + "minimum",
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
                            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                     MapNames::getDpctNamespace() +
                                         "device::segmented_reduce",
                                     LITERAL("128")),
                                 QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5),
                                 ARG(6),
                                 CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                     MapNames::getClNamespace() + "minimum",
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
        HelperFeatureEnum::DplExtrasDpcppExtensions_segmented_reduce,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            HEADER_INSERT_FACTORY(
                HeaderType::HT_Limits,
                WARNING_FACTORY_ENTRY(
                    "cub::DeviceSegmentedReduce::Max",
                    CONDITIONAL_FACTORY_ENTRY(
                        makeCheckAnd(
                            CheckArgCount(10, std::greater_equal<>(), /* IncludeDefaultArg */false),
                            makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                        CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedReduce::Max",
                            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                     MapNames::getDpctNamespace() +
                                         "device::segmented_reduce",
                                     LITERAL("128")),
                                 STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5),
                                 ARG(6),
                                 CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                     MapNames::getClNamespace() + "maximum",
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
                            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                     MapNames::getDpctNamespace() +
                                         "device::segmented_reduce",
                                     LITERAL("128")),
                                 QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5),
                                 ARG(6),
                                 CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                     MapNames::getClNamespace() + "maximum",
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

// cub::DeviceRadixSort::SortKeys
CASE_FACTORY_ENTRY(
  CASE(CheckCubRedundantFunctionCall(),
       REMOVE_API_FACTORY_ENTRY("cub::DeviceRadixSort::SortKeys")),
  OTHERWISE(
       FEATURE_REQUEST_FACTORY(
       HelperFeatureEnum::DplExtrasAlgorithm_sort_keys,
       HEADER_INSERT_FACTORY(
       HeaderType::HT_DPCT_DPL_Utils,
       REMOVE_CUB_TEMP_STORAGE_FACTORY(
       CASE_FACTORY_ENTRY(
         CASE(CheckArgType(2, "cub::DoubleBuffer"),
              CASE_FACTORY_ENTRY(
                CASE(makeCheckAnd(
                     CheckArgCount(7, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     makeCheckNot(CheckArgIsDefaultCudaStream(6))),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeys",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(6)),
                             ARG(2), ARG(3),
                             LITERAL("false"), LITERAL("true"),
                             ARG(4), ARG(5)))),
                CASE(CheckArgCount(6, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeys",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3),
                            LITERAL("false"), LITERAL("true"),
                            ARG(4), ARG(5)))),
                CASE(CheckArgCount(5, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeys",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3),
                            LITERAL("false"), LITERAL("true"),
                            ARG(4)))),
                OTHERWISE(
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeys",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3),
                            LITERAL("false"), LITERAL("true")))))),
         OTHERWISE(
              CASE_FACTORY_ENTRY(                   
                CASE(makeCheckAnd(
                              CheckArgCount(8, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeys",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                            CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(7)),
                            ARG(2), ARG(3), ARG(4), LITERAL("false"), ARG(5), ARG(6)))),
                CASE(CheckArgCount(7, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeys",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), LITERAL("false"), ARG(5), ARG(6)))),
                CASE(CheckArgCount(6, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeys",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), LITERAL("false"), ARG(5)))),
                OTHERWISE(
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeys",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), LITERAL("false"))))))))))))

// cub::DeviceRadixSort::SortKeysDescending
CASE_FACTORY_ENTRY(
  CASE(CheckCubRedundantFunctionCall(),
       REMOVE_API_FACTORY_ENTRY("cub::DeviceRadixSort::SortKeysDescending")),

OTHERWISE(
       FEATURE_REQUEST_FACTORY(
       HelperFeatureEnum::DplExtrasAlgorithm_sort_keys,
       HEADER_INSERT_FACTORY(
       HeaderType::HT_DPCT_DPL_Utils,
       REMOVE_CUB_TEMP_STORAGE_FACTORY(
       CASE_FACTORY_ENTRY(
         CASE(CheckArgType(2, "cub::DoubleBuffer"),
              CASE_FACTORY_ENTRY(
                CASE(makeCheckAnd(
                     CheckArgCount(7, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     makeCheckNot(CheckArgIsDefaultCudaStream(6))),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeysDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(6)),
                             ARG(2), ARG(3),
                             LITERAL("true"), LITERAL("true"),
                             ARG(4), ARG(5)))),
                CASE(CheckArgCount(6, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeysDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3),
                            LITERAL("true"), LITERAL("true"),
                            ARG(4), ARG(5)))),
                CASE(CheckArgCount(5, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeysDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3),
                            LITERAL("true"), LITERAL("true"),
                            ARG(4)))),
                OTHERWISE(
                     MULTI_STMTS_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeysDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3),
                            LITERAL("true"), LITERAL("true")))))),
         OTHERWISE(
              CASE_FACTORY_ENTRY(                   
                CASE(makeCheckAnd(
                              CheckArgCount(8, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeysDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                            CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(7)),
                            ARG(2), ARG(3), ARG(4), LITERAL("true"), ARG(5), ARG(6)))),
                CASE(CheckArgCount(7, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeysDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), LITERAL("true"), ARG(5), ARG(6)))),
                CASE(CheckArgCount(6, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeysDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), LITERAL("true"), ARG(5)))),
                OTHERWISE(
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortKeysDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), LITERAL("true"))))))))))))

// cub::DeviceRadixSort::SortPairs
CASE_FACTORY_ENTRY(
  CASE(CheckCubRedundantFunctionCall(),
       REMOVE_API_FACTORY_ENTRY("cub::DeviceRadixSort::SortPairs")),
  OTHERWISE(
       FEATURE_REQUEST_FACTORY(
       HelperFeatureEnum::DplExtrasAlgorithm_sort_pairs,
       HEADER_INSERT_FACTORY(
       HeaderType::HT_DPCT_DPL_Utils,
       REMOVE_CUB_TEMP_STORAGE_FACTORY(
       CASE_FACTORY_ENTRY(
         CASE(CheckArgType(2, "cub::DoubleBuffer"),
              CASE_FACTORY_ENTRY(
                CASE(makeCheckAnd(
                     CheckArgCount(8, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairs",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(7)),
                             ARG(2), ARG(3), ARG(4),
                             LITERAL("false"), LITERAL("true"),
                             ARG(5), ARG(6)))),
                CASE(CheckArgCount(7, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairs",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3), ARG(4),
                            LITERAL("false"), LITERAL("true"),
                            ARG(5), ARG(6)))),
                CASE(CheckArgCount(6, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairs",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3), ARG(4),
                            LITERAL("false"), LITERAL("true"),
                            ARG(5)))),
                OTHERWISE(
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairs",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3), ARG(4),
                            LITERAL("false"), LITERAL("true")))))),
         OTHERWISE(
              CASE_FACTORY_ENTRY(                   
                CASE(makeCheckAnd(
                              CheckArgCount(10, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairs",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                            CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(9)),
                            ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), LITERAL("false"), ARG(7), ARG(8)))),
                CASE(CheckArgCount(9, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairs",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), LITERAL("false"), ARG(7), ARG(8)))),
                CASE(CheckArgCount(8, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairs",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), LITERAL("false"), ARG(7)))),
                OTHERWISE(
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairs",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), LITERAL("false"))))))))))))

// cub::DeviceRadixSort::SortPairsDescending
CASE_FACTORY_ENTRY(
  CASE(CheckCubRedundantFunctionCall(),
       REMOVE_API_FACTORY_ENTRY("cub::DeviceRadixSort::SortPairsDescending")),
  OTHERWISE(
       FEATURE_REQUEST_FACTORY(
       HelperFeatureEnum::DplExtrasAlgorithm_sort_pairs,
       HEADER_INSERT_FACTORY(
       HeaderType::HT_DPCT_DPL_Utils,
       REMOVE_CUB_TEMP_STORAGE_FACTORY(
       CASE_FACTORY_ENTRY(
         CASE(CheckArgType(2, "cub::DoubleBuffer"),
              CASE_FACTORY_ENTRY(
                CASE(makeCheckAnd(
                     CheckArgCount(8, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairsDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(7)),
                             ARG(2), ARG(3), ARG(4),
                             LITERAL("true"), LITERAL("true"),
                             ARG(5), ARG(6)))),
                CASE(CheckArgCount(7, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairsDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3), ARG(4),
                            LITERAL("true"), LITERAL("true"),
                            ARG(5), ARG(6)))),
                CASE(CheckArgCount(6, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairsDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3), ARG(4),
                            LITERAL("true"), LITERAL("true"),
                            ARG(5)))),
                OTHERWISE(
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairsDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                            CALL("oneapi::dpl::execution::device_policy",
                                 QUEUESTR),
                            ARG(2), ARG(3), ARG(4),
                            LITERAL("true"), LITERAL("true")))))),
         OTHERWISE(
              CASE_FACTORY_ENTRY(                   
                CASE(makeCheckAnd(
                              CheckArgCount(10, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairsDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                            CALL("oneapi::dpl::execution::device_policy",
                                  STREAM(9)),
                            ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), LITERAL("true"), ARG(7), ARG(8)))),
                CASE(CheckArgCount(9, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairsDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), LITERAL("true"), ARG(7), ARG(8)))),
                CASE(CheckArgCount(8, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairsDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), LITERAL("true"), ARG(7)))),
                OTHERWISE(
                     CALL_FACTORY_ENTRY(
                       "cub::DeviceRadixSort::SortPairsDescending",
                       CALL(MapNames::getDpctNamespace() + "sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), LITERAL("true"))))))))))))

// cub::DeviceSegmentedRadixSort::SortKeys
CASE_FACTORY_ENTRY(
    CASE(CheckCubRedundantFunctionCall(),
         REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedRadixSort::SortKeys")),
    OTHERWISE(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasAlgorithm_sort_keys,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                CASE(CheckArgType(2, "cub::DoubleBuffer"),
                     CASE_FACTORY_ENTRY(
                         CASE(makeCheckAnd(
                                  CheckArgCount(10, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedRadixSort::SortKeys",
                                  CALL(MapNames::getDpctNamespace() +
                                           "segmented_sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            STREAM(9)),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       LITERAL("false"), LITERAL("true"),
                                       ARG(7), ARG(8)))),
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
                                       LITERAL("false"), LITERAL("true"),
                                       ARG(7), ARG(8)))),
                         CASE(CheckArgCount(8, std::greater_equal<>(),
                                            /* IncludeDefaultArg */ false),
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedRadixSort::SortKeys",
                                  CALL(MapNames::getDpctNamespace() +
                                           "segmented_sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            QUEUESTR),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       LITERAL("false"), LITERAL("true"),
                                       ARG(7)))),
                         OTHERWISE(CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedRadixSort::SortKeys",
                             CALL(MapNames::getDpctNamespace() +
                                      "segmented_sort_keys",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       QUEUESTR),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  LITERAL("false"), LITERAL("true")))))),
                OTHERWISE(CASE_FACTORY_ENTRY(
                    CASE(makeCheckAnd(
                             CheckArgCount(11, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(10))),
                         CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedRadixSort::SortKeys",
                             CALL(MapNames::getDpctNamespace() +
                                      "segmented_sort_keys",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       STREAM(10)),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), LITERAL("false"), ARG(8), ARG(9)))),
                    CASE(CheckArgCount(10, std::greater_equal<>(),
                                       /* IncludeDefaultArg */ false),
                         CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedRadixSort::SortKeys",
                             CALL(MapNames::getDpctNamespace() +
                                      "segmented_sort_keys",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       QUEUESTR),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), LITERAL("false"), ARG(8), ARG(9)))),
                    CASE(CheckArgCount(9, std::greater_equal<>(),
                                       /* IncludeDefaultArg */ false),
                         CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedRadixSort::SortKeys",
                             CALL(MapNames::getDpctNamespace() +
                                      "segmented_sort_keys",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       QUEUESTR),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), LITERAL("false"), ARG(8)))),
                    OTHERWISE(CALL_FACTORY_ENTRY(
                        "cub::DeviceSegmentedRadixSort::SortKeys",
                        CALL(MapNames::getDpctNamespace() +
                                 "segmented_sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                             LITERAL("false"))))))))))))

// cub::DeviceSegmentedRadixSort::SortKeysDescending
CASE_FACTORY_ENTRY(
    CASE(CheckCubRedundantFunctionCall(),
         REMOVE_API_FACTORY_ENTRY(
             "cub::DeviceSegmentedRadixSort::SortKeysDescending")),
    OTHERWISE(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasAlgorithm_sort_keys,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                CASE(
                    CheckArgType(2, "cub::DoubleBuffer"),
                    CASE_FACTORY_ENTRY(
                        CASE(makeCheckAnd(
                                 CheckArgCount(10, std::greater_equal<>(),
                                               /* IncludeDefaultArg */ false),
                                 makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                             CALL_FACTORY_ENTRY(
                                 "cub::DeviceSegmentedRadixSort::"
                                 "SortKeysDescending",
                                 CALL(MapNames::getDpctNamespace() +
                                          "segmented_sort_keys",
                                      CALL("oneapi::dpl::execution::device_"
                                           "policy",
                                           STREAM(9)),
                                      ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                      LITERAL("true"), LITERAL("true"), ARG(7),
                                      ARG(8)))),
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
                                      LITERAL("true"), LITERAL("true"), ARG(7),
                                      ARG(8)))),
                        CASE(CheckArgCount(8, std::greater_equal<>(),
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
                                      LITERAL("true"), LITERAL("true"),
                                      ARG(7)))),
                        OTHERWISE(CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedRadixSort::SortKeysDescending",
                            CALL(MapNames::getDpctNamespace() +
                                     "segmented_sort_keys",
                                 CALL("oneapi::dpl::execution::device_policy",
                                      QUEUESTR),
                                 ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                 LITERAL("true"), LITERAL("true")))))),
                OTHERWISE(CASE_FACTORY_ENTRY(
                    CASE(
                        makeCheckAnd(
                            CheckArgCount(11, std::greater_equal<>(),
                                          /* IncludeDefaultArg */ false),
                            makeCheckNot(CheckArgIsDefaultCudaStream(10))),
                        CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedRadixSort::SortKeysDescending",
                            CALL(MapNames::getDpctNamespace() +
                                     "segmented_sort_keys",
                                 CALL("oneapi::dpl::execution::device_policy",
                                      STREAM(10)),
                                 ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                                 LITERAL("true"), ARG(8), ARG(9)))),
                    CASE(
                        CheckArgCount(10, std::greater_equal<>(),
                                      /* IncludeDefaultArg */ false),
                        CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedRadixSort::SortKeysDescending",
                            CALL(MapNames::getDpctNamespace() +
                                     "segmented_sort_keys",
                                 CALL("oneapi::dpl::execution::device_policy",
                                      QUEUESTR),
                                 ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                                 LITERAL("true"), ARG(8), ARG(9)))),
                    CASE(
                        CheckArgCount(9, std::greater_equal<>(),
                                      /* IncludeDefaultArg */ false),
                        CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedRadixSort::SortKeysDescending",
                            CALL(MapNames::getDpctNamespace() +
                                     "segmented_sort_keys",
                                 CALL("oneapi::dpl::execution::device_policy",
                                      QUEUESTR),
                                 ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                                 LITERAL("true"), ARG(8)))),
                    OTHERWISE(CALL_FACTORY_ENTRY(
                        "cub::DeviceSegmentedRadixSort::SortKeysDescending",
                        CALL(MapNames::getDpctNamespace() +
                                 "segmented_sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                             LITERAL("true"))))))))))))

// cub::DeviceSegmentedRadixSort::SortPairs
CASE_FACTORY_ENTRY(
    CASE(CheckCubRedundantFunctionCall(),
         REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedRadixSort::SortPairs")),
    OTHERWISE(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasAlgorithm_sort_pairs,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                CASE(
                    CheckArgType(2, "cub::DoubleBuffer"),
                    CASE_FACTORY_ENTRY(
                        CASE(makeCheckAnd(
                                 CheckArgCount(11, std::greater_equal<>(),
                                               /* IncludeDefaultArg */ false),
                                 makeCheckNot(CheckArgIsDefaultCudaStream(10))),
                             CALL_FACTORY_ENTRY(
                                 "cub::DeviceSegmentedRadixSort::SortPairs",
                                 CALL(MapNames::getDpctNamespace() +
                                          "segmented_sort_pairs",
                                      CALL("oneapi::dpl::execution::device_"
                                           "policy",
                                           STREAM(10)),
                                      ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                      ARG(7), LITERAL("false"), LITERAL("true"),
                                      ARG(8), ARG(9)))),
                        CASE(CheckArgCount(10, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             CALL_FACTORY_ENTRY(
                                 "cub::DeviceSegmentedRadixSort::SortPairs",
                                 CALL(MapNames::getDpctNamespace() +
                                          "segmented_sort_pairs",
                                      CALL("oneapi::dpl::execution::device_"
                                           "policy",
                                           QUEUESTR),
                                      ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                      ARG(7), LITERAL("false"), LITERAL("true"),
                                      ARG(8), ARG(9)))),
                        CASE(CheckArgCount(9, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             CALL_FACTORY_ENTRY(
                                 "cub::DeviceSegmentedRadixSort::SortPairs",
                                 CALL(MapNames::getDpctNamespace() +
                                          "segmented_sort_pairs",
                                      CALL("oneapi::dpl::execution::device_"
                                           "policy",
                                           QUEUESTR),
                                      ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                      ARG(7), LITERAL("false"), LITERAL("true"),
                                      ARG(8)))),
                        OTHERWISE(CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedRadixSort::SortPairs",
                            CALL(MapNames::getDpctNamespace() +
                                     "segmented_sort_pairs",
                                 CALL("oneapi::dpl::execution::device_policy",
                                      QUEUESTR),
                                 ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                                 LITERAL("false"), LITERAL("true")))))),
                OTHERWISE(CASE_FACTORY_ENTRY(
                    CASE(makeCheckAnd(
                             CheckArgCount(13, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(12))),
                         CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedRadixSort::SortPairs",
                             CALL(MapNames::getDpctNamespace() +
                                      "segmented_sort_pairs",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       STREAM(12)),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), ARG(8), ARG(9), LITERAL("false"),
                                  ARG(10), ARG(11)))),
                    CASE(CheckArgCount(12, std::greater_equal<>(),
                                       /* IncludeDefaultArg */ false),
                         CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedRadixSort::SortPairs",
                             CALL(MapNames::getDpctNamespace() +
                                      "segmented_sort_pairs",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       QUEUESTR),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), ARG(8), ARG(9), LITERAL("false"),
                                  ARG(10), ARG(11)))),
                    CASE(CheckArgCount(11, std::greater_equal<>(),
                                       /* IncludeDefaultArg */ false),
                         CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedRadixSort::SortPairs",
                             CALL(MapNames::getDpctNamespace() +
                                      "segmented_sort_pairs",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       QUEUESTR),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), ARG(8), ARG(9), LITERAL("false"),
                                  ARG(10)))),
                    OTHERWISE(CALL_FACTORY_ENTRY(
                        "cub::DeviceSegmentedRadixSort::SortPairs",
                        CALL(MapNames::getDpctNamespace() +
                                 "segmented_sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                             ARG(8), ARG(9), LITERAL("false"))))))))))))

// cub::DeviceSegmentedRadixSort::SortPairsDescending
CASE_FACTORY_ENTRY(
    CASE(CheckCubRedundantFunctionCall(),
         REMOVE_API_FACTORY_ENTRY(
             "cub::DeviceSegmentedRadixSort::SortPairsDescending")),
    OTHERWISE(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasAlgorithm_sort_pairs,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                CASE(
                    CheckArgType(2, "cub::DoubleBuffer"),
                    CASE_FACTORY_ENTRY(
                        CASE(makeCheckAnd(
                                 CheckArgCount(11, std::greater_equal<>(),
                                               /* IncludeDefaultArg */ false),
                                 makeCheckNot(CheckArgIsDefaultCudaStream(10))),
                             CALL_FACTORY_ENTRY(
                                 "cub::DeviceSegmentedRadixSort::"
                                 "SortPairsDescending",
                                 CALL(MapNames::getDpctNamespace() +
                                          "segmented_sort_pairs",
                                      CALL("oneapi::dpl::execution::device_"
                                           "policy",
                                           STREAM(10)),
                                      ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                      ARG(7), LITERAL("true"), LITERAL("true"),
                                      ARG(8), ARG(9)))),
                        CASE(CheckArgCount(10, std::greater_equal<>(),
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
                                      ARG(7), LITERAL("true"), LITERAL("true"),
                                      ARG(8), ARG(9)))),
                        CASE(CheckArgCount(9, std::greater_equal<>(),
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
                                      ARG(7), LITERAL("true"), LITERAL("true"),
                                      ARG(8)))),
                        OTHERWISE(CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedRadixSort::"
                            "SortPairsDescending",
                            CALL(MapNames::getDpctNamespace() +
                                     "segmented_sort_pairs",
                                 CALL("oneapi::dpl::execution::device_policy",
                                      QUEUESTR),
                                 ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                                 LITERAL("true"), LITERAL("true")))))),
                OTHERWISE(CASE_FACTORY_ENTRY(
                    CASE(makeCheckAnd(
                             CheckArgCount(13, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(12))),
                         CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedRadixSort::"
                             "SortPairsDescending",
                             CALL(MapNames::getDpctNamespace() +
                                      "segmented_sort_pairs",
                                  CALL("oneapi::dpl::execution::device_policy",
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
                                  CALL("oneapi::dpl::execution::device_policy",
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
                                  CALL("oneapi::dpl::execution::device_policy",
                                       QUEUESTR),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), ARG(8), ARG(9), LITERAL("true"),
                                  ARG(10)))),
                    OTHERWISE(CALL_FACTORY_ENTRY(
                        "cub::DeviceSegmentedRadixSort::SortPairsDescending",
                        CALL(MapNames::getDpctNamespace() +
                                 "segmented_sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                             ARG(8), ARG(9), LITERAL("true"))))))))))))

// cub::DeviceSegmentedSort::SortKeys
CASE_FACTORY_ENTRY(
    CASE(CheckCubRedundantFunctionCall(),
         REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedSort::SortKeys")),
    OTHERWISE(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasAlgorithm_sort_keys,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                CASE(CheckArgType(2, "cub::DoubleBuffer"),
                     CASE_FACTORY_ENTRY(
                         CASE(makeCheckAnd(
                                  CheckArgCount(8, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedSort::SortKeys",
                                  CALL("dpct::segmented_sort_keys",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            STREAM(7)),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       LITERAL("false"), LITERAL("true")))),
                         OTHERWISE(CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedSort::SortKeys",
                             CALL("dpct::segmented_sort_keys",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       QUEUESTR),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  LITERAL("false"), LITERAL("true")))))),
                OTHERWISE(CASE_FACTORY_ENTRY(
                    CASE(makeCheckAnd(
                             CheckArgCount(9, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(8))),
                         CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedSort::SortKeys",
                             CALL("dpct::segmented_sort_keys",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       STREAM(8)),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), LITERAL("false")))),
                    OTHERWISE(CALL_FACTORY_ENTRY(
                        "cub::DeviceSegmentedSort::SortKeys",
                        CALL("dpct::segmented_sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                             LITERAL("false"))))))))))))

// cub::DeviceSegmentedSort::SortKeysDescending
CASE_FACTORY_ENTRY(
    CASE(CheckCubRedundantFunctionCall(),
         REMOVE_API_FACTORY_ENTRY(
             "cub::DeviceSegmentedSort::SortKeysDescending")),
    OTHERWISE(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasAlgorithm_sort_keys,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                CASE(
                    CheckArgType(2, "cub::DoubleBuffer"),
                    CASE_FACTORY_ENTRY(
                        CASE(makeCheckAnd(
                                 CheckArgCount(8, std::greater_equal<>(),
                                               /* IncludeDefaultArg */ false),
                                 makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                             CALL_FACTORY_ENTRY(
                                 "cub::DeviceSegmentedSort::SortKeysDescending",
                                 CALL("dpct::segmented_sort_keys",
                                      CALL("oneapi::dpl::execution::device_"
                                           "policy",
                                           STREAM(7)),
                                      ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                      LITERAL("true"), LITERAL("true")))),
                        OTHERWISE(CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedSort::SortKeysDescending",
                            CALL("dpct::segmented_sort_keys",
                                 CALL("oneapi::dpl::execution::device_policy",
                                      QUEUESTR),
                                 ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                 LITERAL("true"), LITERAL("true")))))),
                OTHERWISE(CASE_FACTORY_ENTRY(
                    CASE(makeCheckAnd(
                             CheckArgCount(9, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(8))),
                         CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedSort::SortKeysDescending",
                             CALL("dpct::segmented_sort_keys",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       STREAM(8)),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), LITERAL("true")))),
                    OTHERWISE(CALL_FACTORY_ENTRY(
                        "cub::DeviceSegmentedSort::SortKeysDescending",
                        CALL("dpct::segmented_sort_keys",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                             LITERAL("true"))))))))))))

// cub::DeviceSegmentedSort::SortPairs
CASE_FACTORY_ENTRY(
    CASE(CheckCubRedundantFunctionCall(),
         REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedSort::SortPairs")),
    OTHERWISE(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasAlgorithm_sort_pairs,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                CASE(CheckArgType(2, "cub::DoubleBuffer"),
                     CASE_FACTORY_ENTRY(
                         CASE(makeCheckAnd(
                                  CheckArgCount(9, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  makeCheckNot(CheckArgIsDefaultCudaStream(8))),
                              CALL_FACTORY_ENTRY(
                                  "cub::DeviceSegmentedSort::SortPairs",
                                  CALL(MapNames::getDpctNamespace() +
                                           "segmented_sort_pairs",
                                       CALL("oneapi::dpl::execution::device_"
                                            "policy",
                                            STREAM(8)),
                                       ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                       ARG(7), LITERAL("false"),
                                       LITERAL("true")))),
                         OTHERWISE(CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedSort::SortPairs",
                             CALL(MapNames::getDpctNamespace() +
                                      "segmented_sort_pairs",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       QUEUESTR),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), LITERAL("false"),
                                  LITERAL("true")))))),
                OTHERWISE(CASE_FACTORY_ENTRY(
                    CASE(makeCheckAnd(
                             CheckArgCount(11, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(10))),
                         CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedSort::SortPairs",
                             CALL(MapNames::getDpctNamespace() +
                                      "segmented_sort_pairs",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       STREAM(10)),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), ARG(8), ARG(9), LITERAL("false")))),
                    OTHERWISE(CALL_FACTORY_ENTRY(
                        "cub::DeviceSegmentedSort::SortPairs",
                        CALL(MapNames::getDpctNamespace() +
                                 "segmented_sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                             ARG(8), ARG(9), LITERAL("false"))))))))))))

// cub::DeviceSegmentedSort::SortPairsDescending
CASE_FACTORY_ENTRY(
    CASE(CheckCubRedundantFunctionCall(),
         REMOVE_API_FACTORY_ENTRY(
             "cub::DeviceSegmentedSort::SortPairsDescending")),
    OTHERWISE(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasAlgorithm_sort_pairs,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                CASE(
                    CheckArgType(2, "cub::DoubleBuffer"),
                    CASE_FACTORY_ENTRY(
                        CASE(
                            makeCheckAnd(
                                CheckArgCount(9, std::greater_equal<>(),
                                              /* IncludeDefaultArg */ false),
                                makeCheckNot(CheckArgIsDefaultCudaStream(8))),
                            CALL_FACTORY_ENTRY(
                                "cub::DeviceSegmentedSort::SortPairsDescending",
                                CALL(
                                    MapNames::getDpctNamespace() +
                                        "segmented_sort_pairs",
                                    CALL(
                                        "oneapi::dpl::execution::device_policy",
                                        STREAM(8)),
                                    ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                    ARG(7), LITERAL("true"), LITERAL("true")))),
                        OTHERWISE(CALL_FACTORY_ENTRY(
                            "cub::DeviceSegmentedSort::SortPairsDescending",
                            CALL(MapNames::getDpctNamespace() +
                                     "segmented_sort_pairs",
                                 CALL("oneapi::dpl::execution::device_policy",
                                      QUEUESTR),
                                 ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                                 LITERAL("true"), LITERAL("true")))))),
                OTHERWISE(CASE_FACTORY_ENTRY(
                    CASE(makeCheckAnd(
                             CheckArgCount(11, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(10))),
                         CALL_FACTORY_ENTRY(
                             "cub::DeviceSegmentedSort::SortPairsDescending",
                             CALL(MapNames::getDpctNamespace() +
                                      "segmented_sort_pairs",
                                  CALL("oneapi::dpl::execution::device_policy",
                                       STREAM(10)),
                                  ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                                  ARG(7), ARG(8), ARG(9), LITERAL("true")))),
                    OTHERWISE(CALL_FACTORY_ENTRY(
                        "cub::DeviceSegmentedSort::SortPairsDescending",
                        CALL(MapNames::getDpctNamespace() +
                                 "segmented_sort_pairs",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                             ARG(8), ARG(9), LITERAL("true"))))))))))))

// cub::DeviceScan::InclusiveScanByKey
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::InclusiveScanByKey"),
    HEADER_INSERT_FACTORY(
        HeaderType::HT_DPL_Execution,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPL_Algorithm,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                makeCheckAnd(CheckArgCount(9, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(8))),
                CALL_FACTORY_ENTRY(
                    "cub::DeviceScan::InclusiveScanByKey",
                    CALL("oneapi::dpl::inclusive_scan_by_key",
                         CALL("oneapi::dpl::execution::device_policy",
                              STREAM(8)),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
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
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
                             ARG(3), ARG(4), ARG(7), ARG(5))),
                    CALL_FACTORY_ENTRY(
                        "cub::DeviceScan::InclusiveScanByKey",
                        CALL("oneapi::dpl::inclusive_scan_by_key",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2),
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
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
                makeCheckAnd(CheckArgCount(8, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                CALL_FACTORY_ENTRY(
                    "cub::DeviceScan::InclusiveSumByKey",
                    CALL("oneapi::dpl::inclusive_scan_by_key",
                         CALL("oneapi::dpl::execution::device_policy",
                              STREAM(7)),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
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
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
                             ARG(3), ARG(4), ARG(6))),
                    CALL_FACTORY_ENTRY(
                        "cub::DeviceScan::InclusiveSumByKey",
                        CALL("oneapi::dpl::inclusive_scan_by_key",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2),
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
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
                makeCheckAnd(CheckArgCount(10, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(9))),
                CALL_FACTORY_ENTRY(
                    "cub::DeviceScan::ExclusiveScanByKey",
                    CALL("oneapi::dpl::exclusive_scan_by_key",
                         CALL("oneapi::dpl::execution::device_policy",
                              STREAM(9)),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(7)),
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
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(7)),
                             ARG(3), ARG(4), ARG(6),
                             LITERAL("std::equal_to<>()"), ARG(5))),
                    CALL_FACTORY_ENTRY(
                        "cub::DeviceScan::ExclusiveScanByKey",
                        CALL("oneapi::dpl::exclusive_scan_by_key",
                             CALL("oneapi::dpl::execution::device_policy",
                                  QUEUESTR),
                             ARG(2),
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(7)),
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
                makeCheckAnd(CheckArgCount(8, std::greater_equal<>(),
                                           /* IncludeDefaultArg */ false),
                             makeCheckNot(CheckArgIsDefaultCudaStream(7))),
                CALL_FACTORY_ENTRY(
                    "cub::DeviceScan::ExclusiveSumByKey",
                    CALL("oneapi::dpl::exclusive_scan_by_key",
                         CALL("oneapi::dpl::execution::device_policy",
                              STREAM(7)),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
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
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
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
                             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)),
                             ARG(3), ARG(4)))))))))
#ifdef CLASS_METHOD_CALL
// cub::ArgIndexInputIterator.normalize
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::DplExtrasIterators_arg_index_input_iterator,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_DPCT_DPL_Utils,
        ASSIGN_FACTORY_ENTRY(
            "cub::ArgIndexInputIterator.normalize", MemberExprBase(),
            MEMBER_CALL(MemberExprBase(), false, LITERAL("create_normalize")))))
#endif

// cub::IADD3
CALL_FACTORY_ENTRY("cub::IADD3",
                   CALL("",
                        BO(BinaryOperatorKind::BO_Add,
                           BO(BinaryOperatorKind::BO_Add,
                              CAST_IF_NOT_SAME(LITERAL("unsigned int"), ARG(0)),
                              CAST_IF_NOT_SAME(LITERAL("unsigned int"), ARG(1))),
                           CAST_IF_NOT_SAME(LITERAL("unsigned int"), ARG(2)))))

// cub::LaneId
MEMBER_CALL_FACTORY_ENTRY("cub::LaneId", MEMBER_CALL(NDITEM, false, LITERAL("get_sub_group")), false, "get_local_linear_id")

// cub::WarpId
MEMBER_CALL_FACTORY_ENTRY("cub::WarpId", MEMBER_CALL(NDITEM, false, LITERAL("get_sub_group")), false, "get_group_linear_id")

// cub::DeviceSegmentedSort::StableSortPairs
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedSort::StableSortPairs"),
    REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::DplExtrasAlgorithm_segmented_sort_pairs,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                CASE(CheckArgCount(11, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                         "cub::DeviceSegmentedSort::StableSortPairs",
                         CALL(MapNames::getDpctNamespace() +
                                  "segmented_sort_pairs",
                              CALL("oneapi::dpl::execution::device_policy",
                                   STREAM(10)),
                              ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                              ARG(8), ARG(9)))),
                CASE(makeCheckAnd(CheckArgCount(10, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  CheckArgType(9, "_Bool")),
                     CALL_FACTORY_ENTRY(
                         "cub::DeviceSegmentedSort::StableSortPairs",
                         CALL(MapNames::getDpctNamespace() +
                                  "segmented_sort_pairs",
                              CALL("oneapi::dpl::execution::device_"
                                   "policy",
                                   STREAM(8)),
                              ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                              LITERAL("false"), LITERAL("true")))),
                CASE(CheckArgCount(10, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                         "cub::DeviceSegmentedSort::StableSortPairs",
                         CALL(MapNames::getDpctNamespace() +
                                  "segmented_sort_pairs",
                              CALL("oneapi::dpl::execution::device_policy",
                                   QUEUESTR),
                              ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                              ARG(8), ARG(9)))),
                CASE(CheckArgCount(9, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                         "cub::DeviceSegmentedSort::StableSortPairs",
                         CALL(MapNames::getDpctNamespace() +
                                  "segmented_sort_pairs",
                              CALL("oneapi::dpl::execution::device_"
                                   "policy",
                                   STREAM(8)),
                              ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                              LITERAL("false"), LITERAL("true")))),
                OTHERWISE(CALL_FACTORY_ENTRY(
                    "cub::DeviceSegmentedSort::StableSortPairs",
                    CALL(MapNames::getDpctNamespace() + "segmented_sort_pairs",
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
        HelperFeatureEnum::DplExtrasAlgorithm_segmented_sort_pairs,
        HEADER_INSERT_FACTORY(
            HeaderType::HT_DPCT_DPL_Utils,
            REMOVE_CUB_TEMP_STORAGE_FACTORY(CASE_FACTORY_ENTRY(
                CASE(CheckArgCount(11, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                         "cub::DeviceSegmentedSort::StableSortPairsDescending",
                         CALL(MapNames::getDpctNamespace() +
                                  "segmented_sort_pairs",
                              CALL("oneapi::dpl::execution::device_policy",
                                   STREAM(10)),
                              ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                              ARG(8), ARG(9), LITERAL("true")))),
                CASE(makeCheckAnd(CheckArgCount(10, std::greater_equal<>(),
                                                /* IncludeDefaultArg */ false),
                                  CheckArgType(9, "_Bool")),
                     CALL_FACTORY_ENTRY(
                         "cub::DeviceSegmentedSort::StableSortPairsDescending",
                         CALL(MapNames::getDpctNamespace() +
                                  "segmented_sort_pairs",
                              CALL("oneapi::dpl::execution::device_"
                                   "policy",
                                   STREAM(8)),
                              ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                              LITERAL("true"), LITERAL("true")))),
                CASE(CheckArgCount(10, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                         "cub::DeviceSegmentedSort::StableSortPairsDescending",
                         CALL(MapNames::getDpctNamespace() +
                                  "segmented_sort_pairs",
                              CALL("oneapi::dpl::execution::device_policy",
                                   QUEUESTR),
                              ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                              ARG(8), ARG(9), LITERAL("true")))),
                CASE(CheckArgCount(9, std::greater_equal<>(),
                                   /* IncludeDefaultArg */ false),
                     CALL_FACTORY_ENTRY(
                         "cub::DeviceSegmentedSort::StableSortPairsDescending",
                         CALL(MapNames::getDpctNamespace() +
                                  "segmented_sort_pairs",
                              CALL("oneapi::dpl::execution::device_"
                                   "policy",
                                   STREAM(8)),
                              ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                              LITERAL("true"), LITERAL("true")))),
                OTHERWISE(CALL_FACTORY_ENTRY(
                    "cub::DeviceSegmentedSort::StableSortPairsDescending",
                    CALL(MapNames::getDpctNamespace() + "segmented_sort_pairs",
                         CALL("oneapi::dpl::execution::device_policy",
                              QUEUESTR),
                         ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                         LITERAL("true"), LITERAL("true"))))))))))

// clang-format on
