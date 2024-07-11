//===--------------- RewriterCXXAPIRoutines.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap
dpct::createCXXAPIRoutinesRewriterMap() {
  return RewriterMap{
      // saturate
      MATH_API_REWRITER_DEVICE(
          "saturate",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "saturate",
                  CALL(MapNames::getDpctNamespace() + "clamp<float>", ARG(0),
                       ARG("0.0f"), ARG("1.0f"))),
              EMPTY_FACTORY_ENTRY("saturate"), EMPTY_FACTORY_ENTRY("saturate"),
              EMPTY_FACTORY_ENTRY("saturate")))};
}
