//===--------------- RewriterTypeCastingIntrinsics.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createTypeCastingIntrinsicsRewriterMap() {
  return RewriterMap{
      // __double2hiint
      MATH_API_REWRITERS_V2(
          "__double2hiint",
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              WARNING_FACTORY_ENTRY(
                  "__double2hiint",
                  CALL_FACTORY_ENTRY(
                      "__double2hiint",
                      CALL(MapNames::getDpctNamespace() + "cast_double_to_int",
                           ARG(0))),
                  Diagnostics::MATH_EMULATION, std::string("__double2hiint"),
                  MapNames::getDpctNamespace() + "cast_double_to_int")))
      // __hiloint2double
      MATH_API_REWRITERS_V2(
          "__hiloint2double",
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              WARNING_FACTORY_ENTRY(
                  "__hiloint2double",
                  CALL_FACTORY_ENTRY(
                      "__hiloint2double",
                      CALL(MapNames::getDpctNamespace() + "cast_ints_to_double",
                           ARG(0), ARG(1))),
                  Diagnostics::MATH_EMULATION, std::string("__hiloint2double"),
                  MapNames::getDpctNamespace() + "cast_ints_to_double")))};
}
