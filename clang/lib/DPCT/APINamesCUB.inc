//===--------------- APINamesComplex.inc ----------------------------------===//
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
        HeaderType::HT_DPL_Algorithm_Execution,
        REMOVE_CUB_TEMP_STORAGE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
            "cub::DeviceReduce::Sum", QUEUESTR, false, "fill", ARG(3),
            CALL("oneapi::dpl::reduce",
                 CALL("oneapi::dpl::execution::device_policy", QUEUESTR),
                 ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4))),
            LITERAL("1")))))
