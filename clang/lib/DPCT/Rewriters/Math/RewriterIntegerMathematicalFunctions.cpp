//===--------------- RewriterIntegerMathematicalFunctions.cpp ---------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createIntegerMathematicalFunctionsRewriterMap() {
  return RewriterMap{
      // abs
      MATH_API_REWRITER_HOST_DEVICE(
          MATH_API_REWRITER_HOST(
              "abs",
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_Stdlib,
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_Math,
                      CALL_FACTORY_ENTRY("abs", CALL("std::abs", ARG(0)))))),
          MATH_API_REWRITER_DEVICE(
              "abs",
              MATH_API_DEVICE_NODES(
                  CONDITIONAL_FACTORY_ENTRY(
                      IsParameterIntegerType(0),
                      CALL_FACTORY_ENTRY(
                          "abs",
                          CALL(MapNames::getClNamespace(false, true) + "abs",
                               ARG(0))),
                      CALL_FACTORY_ENTRY(
                          "abs",
                          CALL(MapNames::getClNamespace(false, true) + "fabs",
                               ARG(0)))),
                  EMPTY_FACTORY_ENTRY("abs"), EMPTY_FACTORY_ENTRY("abs"),
                  EMPTY_FACTORY_ENTRY("abs"))))
      // min
      MATH_API_REWRITER_HOST_DEVICE(
          MATH_API_REWRITER_HOST(
              "min",
              CONDITIONAL_FACTORY_ENTRY(
                  math::UsingDpctMinMax,
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "min", CALL(MapNames::getDpctNamespace() + "min",
                                      CAST_IF_SPECIAL(0), CAST_IF_SPECIAL(1)))),
                  CALL_FACTORY_ENTRY("min", CALL("std::min", CAST_IF_SPECIAL(0),
                                                 CAST_IF_SPECIAL(1))))),
          MATH_API_REWRITER_DEVICE(
              "min",
              MATH_API_DEVICE_NODES(
                  CONDITIONAL_FACTORY_ENTRY(
                      math::UsingDpctMinMax,
                      FEATURE_REQUEST_FACTORY(
                          HelperFeatureEnum::device_ext,
                          CALL_FACTORY_ENTRY(
                              "min",
                              CALL(MapNames::getDpctNamespace() + "min",
                                   CAST_IF_SPECIAL(0), CAST_IF_SPECIAL(1)))),
                      CALL_FACTORY_ENTRY(
                          "min", CALL(MapNames::getClNamespace() + "min",
                                      CAST_IF_SPECIAL(0), CAST_IF_SPECIAL(1)))),
                  EMPTY_FACTORY_ENTRY("min"), EMPTY_FACTORY_ENTRY("min"),
                  EMPTY_FACTORY_ENTRY("min"))))
      // max
      MATH_API_REWRITER_HOST_DEVICE(
          MATH_API_REWRITER_HOST(
              "max",
              CONDITIONAL_FACTORY_ENTRY(
                  math::UsingDpctMinMax,
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "max", CALL(MapNames::getDpctNamespace() + "max",
                                      CAST_IF_SPECIAL(0), CAST_IF_SPECIAL(1)))),
                  CALL_FACTORY_ENTRY("max", CALL("std::max", CAST_IF_SPECIAL(0),
                                                 CAST_IF_SPECIAL(1))))),
          MATH_API_REWRITER_DEVICE(
              "max",
              MATH_API_DEVICE_NODES(
                  CONDITIONAL_FACTORY_ENTRY(
                      math::UsingDpctMinMax,
                      FEATURE_REQUEST_FACTORY(
                          HelperFeatureEnum::device_ext,
                          CALL_FACTORY_ENTRY(
                              "max",
                              CALL(MapNames::getDpctNamespace() + "max",
                                   CAST_IF_SPECIAL(0), CAST_IF_SPECIAL(1)))),
                      CALL_FACTORY_ENTRY(
                          "max", CALL(MapNames::getClNamespace() + "max",
                                      CAST_IF_SPECIAL(0), CAST_IF_SPECIAL(1)))),
                  EMPTY_FACTORY_ENTRY("max"), EMPTY_FACTORY_ENTRY("max"),
                  EMPTY_FACTORY_ENTRY("max"))))
      // llmax
      MATH_API_REWRITER_DEVICE(
          "llmax",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "llmax",
                  CALL(MapNames::getDpctNamespace() + "max",
                       CAST_IF_NOT_SAME(makeLiteral("long long"), ARG(0)),
                       CAST_IF_NOT_SAME(makeLiteral("long long"), ARG(1)))),
              EMPTY_FACTORY_ENTRY("llmax"), EMPTY_FACTORY_ENTRY("llmax"),
              EMPTY_FACTORY_ENTRY("llmax")))
      // llmin
      MATH_API_REWRITER_DEVICE(
          "llmin",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "llmin",
                  CALL(MapNames::getDpctNamespace() + "min",
                       CAST_IF_NOT_SAME(makeLiteral("long long"), ARG(0)),
                       CAST_IF_NOT_SAME(makeLiteral("long long"), ARG(1)))),
              EMPTY_FACTORY_ENTRY("llmin"), EMPTY_FACTORY_ENTRY("llmin"),
              EMPTY_FACTORY_ENTRY("llmin")))
      // ullmax
      MATH_API_REWRITER_DEVICE(
          "ullmax",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "ullmax",
                  CALL(MapNames::getDpctNamespace() + "max",
                       CAST_IF_NOT_SAME(makeLiteral("unsigned long long"),
                                        ARG(0)),
                       CAST_IF_NOT_SAME(makeLiteral("unsigned long long"),
                                        ARG(1)))),
              EMPTY_FACTORY_ENTRY("ullmax"), EMPTY_FACTORY_ENTRY("ullmax"),
              EMPTY_FACTORY_ENTRY("ullmax")))
      // ullmin
      MATH_API_REWRITER_DEVICE(
          "ullmin",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "ullmin",
                  CALL(MapNames::getDpctNamespace() + "min",
                       CAST_IF_NOT_SAME(makeLiteral("unsigned long long"),
                                        ARG(0)),
                       CAST_IF_NOT_SAME(makeLiteral("unsigned long long"),
                                        ARG(1)))),
              EMPTY_FACTORY_ENTRY("ullmin"), EMPTY_FACTORY_ENTRY("ullmin"),
              EMPTY_FACTORY_ENTRY("ullmin")))
      // umax
      MATH_API_REWRITER_DEVICE(
          "umax",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "umax",
                  CALL(MapNames::getDpctNamespace() + "max",
                       CAST_IF_NOT_SAME(makeLiteral("unsigned int"), ARG(0)),
                       CAST_IF_NOT_SAME(makeLiteral("unsigned int"), ARG(1)))),
              EMPTY_FACTORY_ENTRY("umax"), EMPTY_FACTORY_ENTRY("umax"),
              EMPTY_FACTORY_ENTRY("umax")))
      // umin
      MATH_API_REWRITER_DEVICE(
          "umin",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "umin",
                  CALL(MapNames::getDpctNamespace() + "min",
                       CAST_IF_NOT_SAME(makeLiteral("unsigned int"), ARG(0)),
                       CAST_IF_NOT_SAME(makeLiteral("unsigned int"), ARG(1)))),
              EMPTY_FACTORY_ENTRY("umin"), EMPTY_FACTORY_ENTRY("umin"),
              EMPTY_FACTORY_ENTRY("umin")))};
}
