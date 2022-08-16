//===------------------------- APINamesCUB.inc ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cub::DeviceReduce::Sum
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::Sum"),
    HEADER_INSERT_FACTORY(
        HeaderType::HT_DPL_Execution,
        HEADER_INSERT_FACTORY(
        HeaderType::HT_DPL_Algorithm,
        REMOVE_CUB_TEMP_STORAGE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
            "cub::DeviceReduce::Sum", QUEUESTR, false, "fill", ARG(3),
            CALL("oneapi::dpl::reduce",
                 CALL("oneapi::dpl::execution::device_policy", QUEUESTR),
                 ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4))),
            LITERAL("1"))))))

// cub::DeviceScan::ExclusiveSum
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::ExclusiveSum"),
    HEADER_INSERT_FACTORY(
        HeaderType::HT_DPL_Execution,
        HEADER_INSERT_FACTORY(
        HeaderType::HT_DPL_Algorithm,
        REMOVE_CUB_TEMP_STORAGE_FACTORY(CALL_FACTORY_ENTRY(
            "cub::DeviceScan::ExclusiveSum",
            CALL("oneapi::dpl::exclusive_scan",
                 CALL("oneapi::dpl::execution::device_policy", QUEUESTR),
                 ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)), ARG(3),
                 LITERAL("0")))))))

// cub::DeviceScan::InclusiveSum
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::InclusiveSum"),
    HEADER_INSERT_FACTORY(
        HeaderType::HT_DPL_Execution,
        HEADER_INSERT_FACTORY(
        HeaderType::HT_DPL_Algorithm,
        REMOVE_CUB_TEMP_STORAGE_FACTORY(CALL_FACTORY_ENTRY(
            "cub::DeviceScan::InclusiveSum",
            CALL("oneapi::dpl::inclusive_scan",
                 CALL("oneapi::dpl::execution::device_policy", QUEUESTR),
                 ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
                 ARG(3)))))))

// cub::DeviceSelect::Flagged
CONDITIONAL_FACTORY_ENTRY(
    CheckCubRedundantFunctionCall(),
    REMOVE_API_FACTORY_ENTRY("cub::DeviceSelect::Flagged"),
    HEADER_INSERT_FACTORY(
        HeaderType::HT_DPL_Utils,
        REMOVE_CUB_TEMP_STORAGE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
            "cub::DeviceSelect::Flagged", QUEUESTR, false, "fill", ARG(5),
            BO(BinaryOperatorKind::BO_Sub, CALL("dpct::copy_if",
                 CALL("oneapi::dpl::execution::device_policy", QUEUESTR),
                 ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)), ARG(3),
                 ARG(4), LITERAL("[](const auto &t) -> bool { return t; }")), ARG(4)),
            LITERAL("1")))))
