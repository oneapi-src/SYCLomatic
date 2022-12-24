//===--------------- APINamesMathRewrite.inc -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

MATH_API_REWRITER_HOST_DEVICE(
    MATH_API_REWRITER_HOST_2(
        "rsqrt",
        CALL_FACTORY_ENTRY(
            "rsqrt", CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                          ARG(0)))),
    MATH_API_REWRITER_DEVICE_2(
        "rsqrt",
        {
            CALL_FACTORY_ENTRY(
                "rsqrt", CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                              ARG(0)))
            EMPTY_FACTORY_ENTRY("rsqrt"),
            EMPTY_FACTORY_ENTRY("rsqrt")
        }
    )
)

CONDITIONAL_FACTORY_ENTRY(
    makeCheckOr(
        makeCheckAnd(
            HasDirectCallee(),
            makeCheckAnd(
                IsDirectCalleeHasAttribute<CUDADeviceAttr>(),
                makeCheckNot(IsDirectCalleeHasAttribute<CUDAHostAttr>()))),
        makeCheckOr(IsContextCallHasAttribute<CUDADeviceAttr>(),
                    IsContextCallHasAttribute<CUDAGlobalAttr>())),
    CONDITIONAL_FACTORY_ENTRY(
        math::IsStdLibdevice,
        HEADER_INSERT_FACTORY(HeaderType::HT_Stdlib, CALL_FACTORY_ENTRY("std::abs", CALL("std::abs", ARG(0)))),
        CONDITIONAL_FACTORY_ENTRY(
            IsParameterIntegerType(0),
            CALL_FACTORY_ENTRY(
                "std::abs",
                CALL(MapNames::getClNamespace(false, true) + "abs", ARG(0))),
            CALL_FACTORY_ENTRY(
                "std::abs",
                CALL(MapNames::getClNamespace(false, true) + "fabs", ARG(0))))),
    CONDITIONAL_FACTORY_ENTRY(
        math::IsDefinedInCUDA(),
        HEADER_INSERT_FACTORY(HeaderType::HT_Stdlib, NO_REWRITE_FUNCNAME_FACTORY_ENTRY("std::abs", "std::abs")),
        NO_REWRITE_FUNCNAME_FACTORY_ENTRY("std::abs", "std::abs")
    ))

CONDITIONAL_FACTORY_ENTRY(
    makeCheckOr(
        makeCheckAnd(
            HasDirectCallee(),
            makeCheckAnd(
                IsDirectCalleeHasAttribute<CUDADeviceAttr>(),
                makeCheckNot(IsDirectCalleeHasAttribute<CUDAHostAttr>()))),
        makeCheckOr(IsContextCallHasAttribute<CUDADeviceAttr>(),
                    IsContextCallHasAttribute<CUDAGlobalAttr>())),
    CONDITIONAL_FACTORY_ENTRY(
        math::IsStdLibdevice,
        HEADER_INSERT_FACTORY(HeaderType::HT_Stdlib, CALL_FACTORY_ENTRY("abs", CALL("std::abs", ARG(0)))),
        CONDITIONAL_FACTORY_ENTRY(
            IsParameterIntegerType(0),
            CALL_FACTORY_ENTRY(
                "abs",
                CALL(MapNames::getClNamespace(false, true) + "abs", ARG(0))),
            CALL_FACTORY_ENTRY(
                "abs",
                CALL(MapNames::getClNamespace(false, true) + "fabs", ARG(0))))),
    CONDITIONAL_FACTORY_ENTRY(
        math::IsDefinedInCUDA(),
        HEADER_INSERT_FACTORY(HeaderType::HT_Stdlib, NO_REWRITE_FUNCNAME_FACTORY_ENTRY("abs", "std::abs")),
        NO_REWRITE_FUNCNAME_FACTORY_ENTRY("abs", "abs")
    ))

CONDITIONAL_FACTORY_ENTRY(
    math::IsDefinedInCUDA(),
    CALL_FACTORY_ENTRY("saturate",
                       CALL(MapNames::getClNamespace(false, true) + "clamp<float>",
                            ARG(0), ARG("0.0f"), ARG("1.0f"))),
    NO_REWRITE_FUNCNAME_FACTORY_ENTRY("saturate", "saturate"))

MATH_API_REWRITER_HOST_DEVICE(
    MATH_API_REWRITER_HOST_1(
        "pow",
        makeCheckOr(IsArgumentIntegerType(1),
                    makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                                 makeCheckNot(math::HasSideEffects(0)))),
        CONDITIONAL_FACTORY_ENTRY(
            makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                         makeCheckNot(math::HasSideEffects(0))),
            BINARY_OP_FACTORY_ENTRY("pow", BinaryOperatorKind::BO_Mul,
                                    makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(0)),
            CALL_FACTORY_ENTRY("pow",
                               CALL(MapNames::getClNamespace(false, true) +
                                        "pown<double>",
                                    ARG(0), ARG(1)))),
        CALL_FACTORY_ENTRY("pow", CALL(MapNames::getClNamespace(false, true) +
                                           "pow<double>",
                                       ARG(0), ARG(1)))),
    MATH_API_REWRITER_DEVICE_1(
        "pow",
        makeCheckOr(IsArgumentIntegerType(1),
                    makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                                 makeCheckNot(math::HasSideEffects(0)))),
        CONDITIONAL_FACTORY_ENTRY(
            makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                         makeCheckNot(math::HasSideEffects(0))),
            BINARY_OP_FACTORY_ENTRY("pow", BinaryOperatorKind::BO_Mul,
                                    makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(0)),
            CALL_FACTORY_ENTRY("pow",
                               CALL(MapNames::getClNamespace(false, true) +
                                        "pown<double>",
                                    ARG(0), ARG(1)))),
        {
            CALL_FACTORY_ENTRY(
                "pow", CALL(MapNames::getClNamespace(false, true) + "pow<double>",
                            ARG(0), ARG(1)))
            EMPTY_FACTORY_ENTRY("pow"),
            EMPTY_FACTORY_ENTRY("pow")
        }
))

MATH_API_REWRITER_HOST_DEVICE(
    MATH_API_REWRITER_HOST_1(
        "powf",
        makeCheckOr(IsArgumentIntegerType(1),
                    makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                                 makeCheckNot(math::HasSideEffects(0)))),
        CONDITIONAL_FACTORY_ENTRY(
            makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                         makeCheckNot(math::HasSideEffects(0))),
            BINARY_OP_FACTORY_ENTRY("powf", BinaryOperatorKind::BO_Mul,
                                    makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(0)),
            CALL_FACTORY_ENTRY("powf",
                               CALL(MapNames::getClNamespace(false, true) +
                                        "pown<float>",
                                    ARG(0), ARG(1)))),
        CALL_FACTORY_ENTRY("powf", CALL(MapNames::getClNamespace(false, true) +
                                           "pow<float>",
                                       ARG(0), ARG(1)))),
    MATH_API_REWRITER_DEVICE_1(
        "powf",
        makeCheckOr(IsArgumentIntegerType(1),
                    makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                                 makeCheckNot(math::HasSideEffects(0)))),
        CONDITIONAL_FACTORY_ENTRY(
            makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                         makeCheckNot(math::HasSideEffects(0))),
            BINARY_OP_FACTORY_ENTRY("powf", BinaryOperatorKind::BO_Mul,
                                    makeCallArgCreatorWithCall(0),
                                    makeCallArgCreatorWithCall(0)),
            CALL_FACTORY_ENTRY("powf",
                               CALL(MapNames::getClNamespace(false, true) +
                                        "pown<float>",
                                    ARG(0), ARG(1)))),
        {
            CALL_FACTORY_ENTRY(
                "powf", CALL(MapNames::getClNamespace(false, true) + "pow<float>",
                            ARG(0), ARG(1)))
            EMPTY_FACTORY_ENTRY("powf"),
            EMPTY_FACTORY_ENTRY("powf")
        }
))

MATH_API_REWRITER_DEVICE_1(
    "__powf",
    makeCheckOr(IsArgumentIntegerType(1),
                makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                             makeCheckNot(math::HasSideEffects(0)))),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                     makeCheckNot(math::HasSideEffects(0))),
        BINARY_OP_FACTORY_ENTRY("__powf", BinaryOperatorKind::BO_Mul,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(0)),
        CALL_FACTORY_ENTRY("__powf",
                           CALL(MapNames::getClNamespace(false, true) +
                                    "pown<float>",
                                ARG(0), ARG(1)))),
    {
        CALL_FACTORY_ENTRY(
            "__powf", CALL(MapNames::getClNamespace(false, true) + "pow<float>",
                        ARG(0), ARG(1)))
        EMPTY_FACTORY_ENTRY("__powf"),
        EMPTY_FACTORY_ENTRY("__powf")
    }
)
