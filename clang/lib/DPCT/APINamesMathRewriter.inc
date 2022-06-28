//===--------------- APINamesMathRewriter.inc -----------------------------===//
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
    NO_REWRITE_FUNCNAME_FACTORY_ENTRY("rsqrt", ""))
