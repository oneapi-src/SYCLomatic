//===--------------- RewriterIntegerIntrinsics.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createIntegerIntrinsicsRewriterMap() {
  return RewriterMap{
      // __dp2a_lo
      MATH_API_REWRITER_DEVICE(
          "__dp2a_lo",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dp2a_lo"),
              EMPTY_FACTORY_ENTRY("__dp2a_lo"),
              EMPTY_FACTORY_ENTRY("__dp2a_lo"),
              CALL_FACTORY_ENTRY("__dp2a_lo",
                                 CALL(MapNames::getDpctNamespace() + "dp2a_lo",
                                      ARG(0), ARG(1), ARG(2)))))
      // __dp2a_hi
      MATH_API_REWRITER_DEVICE(
          "__dp2a_hi",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dp2a_hi"),
              EMPTY_FACTORY_ENTRY("__dp2a_hi"),
              EMPTY_FACTORY_ENTRY("__dp2a_hi"),
              CALL_FACTORY_ENTRY("__dp2a_hi",
                                 CALL(MapNames::getDpctNamespace() + "dp2a_hi",
                                      ARG(0), ARG(1), ARG(2)))))
      // __dp4a
      MATH_API_REWRITER_DEVICE(
          "__dp4a",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dp4a"), EMPTY_FACTORY_ENTRY("__dp4a"),
              EMPTY_FACTORY_ENTRY("__dp4a"),
              CALL_FACTORY_ENTRY("__dp4a",
                                 CALL(MapNames::getDpctNamespace() + "dp4a",
                                      ARG(0), ARG(1), ARG(2)))))
      // __umulhi
      MATH_API_REWRITERS_V2(
          "__umulhi",
          MATH_API_REWRITER_PAIR(
              math::Tag::device_normal,
              CALL_FACTORY_ENTRY("__umulhi",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "mul_hi<unsigned>",
                                      ARG(0), ARG(1)))))};
}
