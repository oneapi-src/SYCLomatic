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
        UseCAndCXXStandardLibrariesExt,
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
        IsDefinedInCUDA(),
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
        UseCAndCXXStandardLibrariesExt,
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
        IsDefinedInCUDA(),
        HEADER_INSERT_FACTORY(HeaderType::HT_STDLIB, NO_REWRITE_FUNCNAME_FACTORY_ENTRY("abs", "std::abs")),
        NO_REWRITE_FUNCNAME_FACTORY_ENTRY("abs", "abs")
    ))

CONDITIONAL_FACTORY_ENTRY(
    IsDefinedInCUDA(),
    CALL_FACTORY_ENTRY("saturate",
                       CALL(MapNames::getClNamespace(false, true) + "clamp<float>",
                            ARG(0), ARG("0.0f"), ARG("1.0f"))),
    NO_REWRITE_FUNCNAME_FACTORY_ENTRY("saturate", "saturate"))

CONDITIONAL_FACTORY_ENTRY(
    GetUsingGenericSpace,
    CALL_FACTORY_ENTRY(
        "atomicAdd",
        CALL(MapNames::getDpctNamespace() + "atomic_fetch_add", ARG(0), ARG(1))),
    /* TODO: Handle case where (GetUsingGenericSpace == false) */
    CALL_FACTORY_ENTRY(
        "atomicAdd",
        CALL(MapNames::getDpctNamespace() + "atomic_fetch_add", ARG(0), ARG(1)))
)
