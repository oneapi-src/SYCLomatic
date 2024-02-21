//===--------------- RewriterHalfMathFunctions.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createHalfMathFunctionsRewriterMap() {
  return RewriterMap{
      // hceil
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hceil",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hceil"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hceil",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::ceil",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hceil"),
                  CALL_FACTORY_ENTRY(
                      "hceil",
                      CALL(MapNames::getClNamespace(false, true) + "ceil",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hceil",
              CALL_FACTORY_ENTRY("hceil",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::ceil",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hceil", CALL(MapNames::getClNamespace(false, true) + "ceil",
                                CALL("float", ARG(0))))))
      // hcos
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hcos",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hcos"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hcos",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::cos",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hcos"),
                  CALL_FACTORY_ENTRY(
                      "hcos",
                      CALL(MapNames::getClNamespace(false, true) + "cos",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hcos",
              CALL_FACTORY_ENTRY("hcos",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::cos",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hcos", CALL(MapNames::getClNamespace(false, true) + "cos",
                               CALL("float", ARG(0))))))
      // hexp
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hexp",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hexp"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hexp",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::exp",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hexp"),
                  CALL_FACTORY_ENTRY(
                      "hexp",
                      CALL(MapNames::getClNamespace(false, true) + "exp",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hexp",
              CALL_FACTORY_ENTRY("hexp",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::exp",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hexp", CALL(MapNames::getClNamespace(false, true) + "exp",
                               CALL("float", ARG(0))))))
      // hexp10
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hexp10",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hexp10"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hexp10",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::exp10",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hexp10"),
                  CALL_FACTORY_ENTRY(
                      "hexp10",
                      CALL(MapNames::getClNamespace(false, true) + "exp10",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hexp10",
              CALL_FACTORY_ENTRY("hexp10",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::exp10",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hexp10",
                  CALL(MapNames::getClNamespace(false, true) + "exp10",
                       CALL("float", ARG(0))))))
      // hexp2
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hexp2",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hexp2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hexp2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::exp2",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hexp2"),
                  CALL_FACTORY_ENTRY(
                      "hexp2",
                      CALL(MapNames::getClNamespace(false, true) + "exp2",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hexp2",
              CALL_FACTORY_ENTRY("hexp2",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::exp2",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hexp2", CALL(MapNames::getClNamespace(false, true) + "exp2",
                                CALL("float", ARG(0))))))
      // hfloor
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hfloor",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hfloor"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hfloor",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::floor",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hfloor"),
                  CALL_FACTORY_ENTRY(
                      "hfloor",
                      CALL(MapNames::getClNamespace(false, true) + "floor",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hfloor",
              CALL_FACTORY_ENTRY("hfloor",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::floor",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hfloor",
                  CALL(MapNames::getClNamespace(false, true) + "floor",
                       CALL("float", ARG(0))))))
      // hlog
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hlog",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hlog"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hlog",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::log",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hlog"),
                  CALL_FACTORY_ENTRY(
                      "hlog",
                      CALL(MapNames::getClNamespace(false, true) + "log",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hlog",
              CALL_FACTORY_ENTRY("hlog",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::log",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hlog", CALL(MapNames::getClNamespace(false, true) + "log",
                               CALL("float", ARG(0))))))
      // hlog10
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hlog10",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hlog10"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hlog10",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::log10",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hlog10"),
                  CALL_FACTORY_ENTRY(
                      "hlog10",
                      CALL(MapNames::getClNamespace(false, true) + "log10",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hlog10",
              CALL_FACTORY_ENTRY("hlog10",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::log10",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hlog10",
                  CALL(MapNames::getClNamespace(false, true) + "log10",
                       CALL("float", ARG(0))))))
      // hlog2
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hlog2",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hlog2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hlog2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::log2",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hlog2"),
                  CALL_FACTORY_ENTRY(
                      "hlog2",
                      CALL(MapNames::getClNamespace(false, true) + "log2",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hlog2",
              CALL_FACTORY_ENTRY("hlog2",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::log2",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hlog2", CALL(MapNames::getClNamespace(false, true) + "log2",
                                CALL("float", ARG(0))))))
      // hrcp
      MATH_API_REWRITER_DEVICE(
          "hrcp", MATH_API_DEVICE_NODES(
                      EMPTY_FACTORY_ENTRY("hrcp"),
                      MATH_API_SPECIFIC_ELSE_EMU(
                          CheckArgType(0, "__half"),
                          HEADER_INSERT_FACTORY(
                              HeaderType::HT_SYCL_Math,
                              CALL_FACTORY_ENTRY(
                                  "hrcp", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::inv",
                                               ARG(0))))),
                      EMPTY_FACTORY_ENTRY("hrcp"),
                      CALL_FACTORY_ENTRY(
                          "hrcp", CALL(MapNames::getClNamespace(false, true) +
                                           "half_precision::recip",
                                       CALL("float", ARG(0))))))
      // hrint
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hrint",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hrint"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hrint",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::rint",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hrint"),
                  CALL_FACTORY_ENTRY(
                      "hrint",
                      CALL(MapNames::getClNamespace(false, true) + "rint",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hrint",
              CALL_FACTORY_ENTRY("hrint",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::rint",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hrint", CALL(MapNames::getClNamespace(false, true) + "rint",
                                CALL("float", ARG(0))))))
      // hrsqrt
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hrsqrt",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hrsqrt"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hrsqrt",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::rsqrt",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hrsqrt"),
                  CALL_FACTORY_ENTRY(
                      "hrsqrt",
                      CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hrsqrt",
              CALL_FACTORY_ENTRY("hrsqrt",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::rsqrt",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hrsqrt",
                  CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                       CALL("float", ARG(0))))))
      // hsin
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hsin",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hsin"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hsin",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::sin",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hsin"),
                  CALL_FACTORY_ENTRY(
                      "hsin",
                      CALL(MapNames::getClNamespace(false, true) + "sin",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hsin",
              CALL_FACTORY_ENTRY("hsin",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::sin",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hsin", CALL(MapNames::getClNamespace(false, true) + "sin",
                               CALL("float", ARG(0))))))
      // hsqrt
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "hsqrt",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("hsqrt"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("hsqrt",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::sqrt",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("hsqrt"),
                  CALL_FACTORY_ENTRY(
                      "hsqrt",
                      CALL(MapNames::getClNamespace(false, true) + "sqrt",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "hsqrt",
              CALL_FACTORY_ENTRY("hsqrt",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::sqrt",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "hsqrt", CALL(MapNames::getClNamespace(false, true) + "sqrt",
                                CALL("float", ARG(0))))))
      // htrunc
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "htrunc",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("htrunc"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("htrunc",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::trunc",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("htrunc"),
                  CALL_FACTORY_ENTRY(
                      "htrunc",
                      CALL(MapNames::getClNamespace(false, true) + "trunc",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "htrunc",
              CALL_FACTORY_ENTRY("htrunc",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::trunc",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "htrunc",
                  CALL(MapNames::getClNamespace(false, true) + "trunc",
                       CALL("float", ARG(0))))))};
}
