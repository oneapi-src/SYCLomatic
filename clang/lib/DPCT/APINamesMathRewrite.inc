//===--------------- APINamesMathRewrite.inc -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Policies to migrate math functions with new rewriter:
// 1) Functions with __device__ attribute but without __host__
//    attribute are treated as device functions;
// 2) Functions whose calling functions are augmented with __device__
//    or __global__ attributes are treated as device functions;
// 3) Other functions are treated as host functions.
//    eg. "__host__ __device__ fabs()" falls in 3) if fabs is not called in
//    device or kernel

CONDITIONAL_FACTORY_ENTRY(
    makeCheckOr(
        makeCheckAnd(
            HasDirectCallee(),
            makeCheckAnd(
                IsDirectCalleeHasAttribute<CUDADeviceAttr>(),
                makeCheckNot(IsDirectCalleeHasAttribute<CUDAHostAttr>()))),
        makeCheckOr(IsContextCallHasAttribute<CUDADeviceAttr>(),
                    IsContextCallHasAttribute<CUDAGlobalAttr>())),
    CALL_FACTORY_ENTRY("rsqrt",
                       CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                            CAST_IF_NEED(makeLiteral("double"), ARG(0)))),
    NO_REWRITE_FUNCNAME_FACTORY_ENTRY("rsqrt", "rsqrt"))

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
        math::UseCAndCXXStandardLibrariesExt("std::abs"),
        HEADER_INSERT_FACTORY(HeaderType::HT_STDLIB, CALL_FACTORY_ENTRY("std::abs", CALL("std::abs", ARG(0)))),
        CONDITIONAL_FACTORY_ENTRY(
            IsIntegerType(0),
            CALL_FACTORY_ENTRY(
                "std::abs",
                CALL(MapNames::getClNamespace(false, true) + "abs", ARG(0))),
            CALL_FACTORY_ENTRY(
                "std::abs",
                CALL(MapNames::getClNamespace(false, true) + "fabs", ARG(0))))),
    CONDITIONAL_FACTORY_ENTRY(
        math::IsDefinedInCUDA(),
        HEADER_INSERT_FACTORY(HeaderType::HT_STDLIB, NO_REWRITE_FUNCNAME_FACTORY_ENTRY("std::abs", "std::abs")),
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
        math::UseCAndCXXStandardLibrariesExt("abs"),
        HEADER_INSERT_FACTORY(HeaderType::HT_STDLIB, CALL_FACTORY_ENTRY("abs", CALL("std::abs", ARG(0)))),
        CONDITIONAL_FACTORY_ENTRY(
            IsIntegerType(0),
            CALL_FACTORY_ENTRY(
                "abs",
                CALL(MapNames::getClNamespace(false, true) + "abs", ARG(0))),
            CALL_FACTORY_ENTRY(
                "abs",
                CALL(MapNames::getClNamespace(false, true) + "fabs", ARG(0))))),
    CONDITIONAL_FACTORY_ENTRY(
        math::IsDefinedInCUDA(),
        HEADER_INSERT_FACTORY(HeaderType::HT_STDLIB, NO_REWRITE_FUNCNAME_FACTORY_ENTRY("abs", "std::abs")),
        NO_REWRITE_FUNCNAME_FACTORY_ENTRY("abs", "abs")
    ))

CONDITIONAL_FACTORY_ENTRY(
    math::IsDefinedInCUDA(),
    CALL_FACTORY_ENTRY("saturate",
                       CALL(MapNames::getClNamespace(false, true) + "clamp<float>",
                            ARG(0), ARG("0.0f"), ARG("1.0f"))),
    NO_REWRITE_FUNCNAME_FACTORY_ENTRY("saturate", "saturate"))

// Eg, optmized migration
//MATH_API_REWRITER_WITH_OPT(
//"pow",
//CONDITIONAL_FACTORY_ENTRY(
//    makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
//                 makeCheckAnd(makeCheckNot(math::HasSideEffects(0)),
//                              makeCheckNot(math::HasSideEffects(1)))),
//    BINARY_OP_FACTORY_ENTRY("pow", BinaryOperatorKind::BO_Mul,
//                            makeCallArgCreatorWithCall(0),
//                            makeCallArgCreatorWithCall(0)),
//    CONDITIONAL_FACTORY_ENTRY(
//        IsIntegerType(1),
//        CALL_FACTORY_ENTRY("pow",
//                           CALL(MapNames::getClNamespace(false, true) + "pown",
//                                CAST_IF_NEED(makeLiteral("double"), ARG(0)),
//                                CAST_IF_NEED(makeLiteral("int"), ARG(0)))),
//        NO_REWRITE_FUNCNAME_FACTORY_ENTRY("pow", "pow"))),
//HEADER_INSERT_FACTORY(HeaderType::HT_Math, CALL_FACTORY_ENTRY("pow", CALL("std::pow", ARG(0), ARG(1)))),
//HEADER_INSERT_FACTORY(HeaderType::HT_Math, CALL_FACTORY_ENTRY("pow", CALL("std::pow", ARG(0), ARG(1)))),
//HEADER_INSERT_FACTORY(HeaderType::HT_Math, CALL_FACTORY_ENTRY("pow", CALL("std::pow", ARG(0), ARG(1)))),
//CONDITIONAL_FACTORY_ENTRY(
//    makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
//                 makeCheckAnd(makeCheckNot(math::HasSideEffects(0)),
//                              makeCheckNot(math::HasSideEffects(1)))),
//    BINARY_OP_FACTORY_ENTRY("pow", BinaryOperatorKind::BO_Mul,
//                            makeCallArgCreatorWithCall(0),
//                            makeCallArgCreatorWithCall(0)),
//    CONDITIONAL_FACTORY_ENTRY(
//        IsIntegerType(1),
//        CALL_FACTORY_ENTRY("pow",
//                           CALL(MapNames::getClNamespace(false, true) + "pown",
//                                CAST_IF_NEED(makeLiteral("double"), ARG(0)),
//                                CAST_IF_NEED(makeLiteral("int"), ARG(0)))),
//        CALL_FACTORY_ENTRY("pow",
//                           CALL(MapNames::getClNamespace(false, true) + "pow<double>",
//                                ARG(0), ARG(1)))))
//)
MATH_API_REWRITER_WITH_OPT(
"pow",
CALL_FACTORY_ENTRY("pow", CALL("std::pow1", ARG(0), ARG(1))),
CALL_FACTORY_ENTRY("pow", CALL("std::pow2", ARG(0), ARG(1))),
CALL_FACTORY_ENTRY("pow", CALL("std::pow3", ARG(0), ARG(1))),
CALL_FACTORY_ENTRY("pow", CALL("std::pow4", ARG(0), ARG(1))),
CALL_FACTORY_ENTRY("pow", CALL("std::pow5", ARG(0), ARG(1)))
)

MATH_API_REWRITER_WITH_OPT(
"powf",
CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                 makeCheckAnd(makeCheckNot(math::HasSideEffects(0)),
                              makeCheckNot(math::HasSideEffects(1)))),
    BINARY_OP_FACTORY_ENTRY("powf", BinaryOperatorKind::BO_Mul,
                            makeCallArgCreatorWithCall(0),
                            makeCallArgCreatorWithCall(0)),
    CONDITIONAL_FACTORY_ENTRY(
        IsIntegerType(1),
        CALL_FACTORY_ENTRY("powf",
                           CALL(MapNames::getClNamespace(false, true) + "pown",
                                CAST_IF_NEED(makeLiteral("float"), ARG(0)),
                                CAST_IF_NEED(makeLiteral("int"), ARG(0)))),
        NO_REWRITE_FUNCNAME_FACTORY_ENTRY("powf", "powf"))),
HEADER_INSERT_FACTORY(HeaderType::HT_Math, CALL_FACTORY_ENTRY("powf", CALL("std::pow", ARG(0), ARG(1)))),
HEADER_INSERT_FACTORY(HeaderType::HT_Math, CALL_FACTORY_ENTRY("powf", CALL("std::pow", ARG(0), ARG(1)))),
HEADER_INSERT_FACTORY(HeaderType::HT_Math, CALL_FACTORY_ENTRY("powf", CALL("std::pow", ARG(0), ARG(1)))),
CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                 makeCheckAnd(makeCheckNot(math::HasSideEffects(0)),
                              makeCheckNot(math::HasSideEffects(1)))),
    BINARY_OP_FACTORY_ENTRY("powf", BinaryOperatorKind::BO_Mul,
                            makeCallArgCreatorWithCall(0),
                            makeCallArgCreatorWithCall(0)),
    CONDITIONAL_FACTORY_ENTRY(
        IsIntegerType(1),
        CALL_FACTORY_ENTRY("powf",
                           CALL(MapNames::getClNamespace(false, true) + "pown",
                                CAST_IF_NEED(makeLiteral("float"), ARG(0)),
                                CAST_IF_NEED(makeLiteral("int"), ARG(0)))),
        CALL_FACTORY_ENTRY("powf",
                           CALL(MapNames::getClNamespace(false, true) + "pow<float>",
                                ARG(0), ARG(1)))))
)

MATH_API_REWRITER_WITH_OPT(
"__powf",
CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                 makeCheckAnd(makeCheckNot(math::HasSideEffects(0)),
                              makeCheckNot(math::HasSideEffects(1)))),
    BINARY_OP_FACTORY_ENTRY("__powf", BinaryOperatorKind::BO_Mul,
                            makeCallArgCreatorWithCall(0),
                            makeCallArgCreatorWithCall(0)),
    CONDITIONAL_FACTORY_ENTRY(
        IsIntegerType(1),
        CALL_FACTORY_ENTRY("__powf",
                           CALL(MapNames::getClNamespace(false, true) + "pown",
                                CAST_IF_NEED(makeLiteral("float"), ARG(0)),
                                CAST_IF_NEED(makeLiteral("int"), ARG(0)))),
        HEADER_INSERT_FACTORY(HeaderType::HT_Math, CALL_FACTORY_ENTRY("__powf", CALL("std::pow", ARG(0), ARG(1)))))),
HEADER_INSERT_FACTORY(HeaderType::HT_Math, CALL_FACTORY_ENTRY("__powf", CALL("std::pow", ARG(0), ARG(1)))),
HEADER_INSERT_FACTORY(HeaderType::HT_Math, CALL_FACTORY_ENTRY("__powf", CALL("std::pow", ARG(0), ARG(1)))),
HEADER_INSERT_FACTORY(HeaderType::HT_Math, CALL_FACTORY_ENTRY("__powf", CALL("std::pow", ARG(0), ARG(1)))),
CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckArgIsConstantIntWithValue(1, 2),
                 makeCheckAnd(makeCheckNot(math::HasSideEffects(0)),
                              makeCheckNot(math::HasSideEffects(1)))),
    BINARY_OP_FACTORY_ENTRY("__powf", BinaryOperatorKind::BO_Mul,
                            makeCallArgCreatorWithCall(0),
                            makeCallArgCreatorWithCall(0)),
    CONDITIONAL_FACTORY_ENTRY(
        IsIntegerType(1),
        CALL_FACTORY_ENTRY("__powf",
                           CALL(MapNames::getClNamespace(false, true) + "pown",
                                CAST_IF_NEED(makeLiteral("float"), ARG(0)),
                                CAST_IF_NEED(makeLiteral("int"), ARG(0)))),
        CALL_FACTORY_ENTRY("__powf",
                           CALL(MapNames::getClNamespace(false, true) + "pow<float>",
                                ARG(0), ARG(1)))))
)

