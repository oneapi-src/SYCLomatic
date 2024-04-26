//===--------------- RewriterSTDFunctions.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createSTDFunctionsRewriterMap() {
  return RewriterMap{
      // std::abs
      MATH_API_REWRITERS_V2(
          "std::abs",
          MATH_API_REWRITER_PAIR(
              math::Tag::host_normal,
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_Stdlib,
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_Math,
                      CALL_FACTORY_ENTRY("std::abs",
                                         CALL("std::abs", ARG(0)))))),
          MATH_API_REWRITER_PAIR(
              math::Tag::device_normal,
              CONDITIONAL_FACTORY_ENTRY(
                  IsParameterIntegerType(0),
                  CALL_FACTORY_ENTRY(
                      "std::abs",
                      CALL(MapNames::getClNamespace(false, true) + "abs",
                           ARG(0))),
                  CALL_FACTORY_ENTRY(
                      "std::abs",
                      CALL(MapNames::getClNamespace(false, true) + "fabs",
                           ARG(0))))))};
}
