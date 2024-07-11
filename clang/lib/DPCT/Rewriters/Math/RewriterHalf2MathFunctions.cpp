//===--------------- RewriterHalf2MathFunctions.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createHalf2MathFunctionsRewriterMap() {
  return RewriterMap{
      // h2ceil
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2ceil",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2ceil"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2ceil",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::ceil",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2ceil"),
                  CALL_FACTORY_ENTRY(
                      "h2ceil",
                      CALL(MapNames::getClNamespace(false, true) + "ceil",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2ceil",
              CALL_FACTORY_ENTRY("h2ceil",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::ceil",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2ceil",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "ceil",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "ceil",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2cos
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2cos",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2cos"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2cos",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::cos",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2cos"),
                  CALL_FACTORY_ENTRY(
                      "h2cos",
                      CALL(MapNames::getClNamespace(false, true) + "cos",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2cos",
              CALL_FACTORY_ENTRY("h2cos",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::cos",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2cos",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "cos",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "cos",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2exp
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2exp",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2exp"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2exp",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::exp",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2exp"),
                  CALL_FACTORY_ENTRY(
                      "h2exp",
                      CALL(MapNames::getClNamespace(false, true) + "exp",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2exp",
              CALL_FACTORY_ENTRY("h2exp",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::exp",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2exp",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "exp",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "exp",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2exp10
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2exp10",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2exp10"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2exp10",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::exp10",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2exp10"),
                  CALL_FACTORY_ENTRY(
                      "h2exp10",
                      CALL(MapNames::getClNamespace(false, true) + "exp10",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2exp10",
              CALL_FACTORY_ENTRY("h2exp10",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::exp10",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2exp10",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "exp10",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "exp10",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2exp2
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2exp2",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2exp2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2exp2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::exp2",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2exp2"),
                  CALL_FACTORY_ENTRY(
                      "h2exp2",
                      CALL(MapNames::getClNamespace(false, true) + "exp2",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2exp2",
              CALL_FACTORY_ENTRY("h2exp2",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::exp2",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2exp2",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "exp2",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "exp2",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2floor
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2floor",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2floor"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2floor",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::floor",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2floor"),
                  CALL_FACTORY_ENTRY(
                      "h2floor",
                      CALL(MapNames::getClNamespace(false, true) + "floor",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2floor",
              CALL_FACTORY_ENTRY("h2floor",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::floor",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2floor",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "floor",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "floor",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2log
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2log",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2log"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2log",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::log",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2log"),
                  CALL_FACTORY_ENTRY(
                      "h2log",
                      CALL(MapNames::getClNamespace(false, true) + "log",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2log",
              CALL_FACTORY_ENTRY("h2log",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::log",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2log",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "log",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "log",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2log10
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2log10",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2log10"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2log10",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::log10",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2log10"),
                  CALL_FACTORY_ENTRY(
                      "h2log10",
                      CALL(MapNames::getClNamespace(false, true) + "log10",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2log10",
              CALL_FACTORY_ENTRY("h2log10",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::log10",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2log10",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "log10",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "log10",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2log2
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2log2",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2log2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2log2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::log2",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2log2"),
                  CALL_FACTORY_ENTRY(
                      "h2log2",
                      CALL(MapNames::getClNamespace(false, true) + "log2",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2log2",
              CALL_FACTORY_ENTRY("h2log2",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::log2",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2log2",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "log2",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "log2",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2rcp
      MATH_API_REWRITER_DEVICE(
          "h2rcp",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("h2rcp"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2rcp",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::inv",
                                              ARG(0))))),
              EMPTY_FACTORY_ENTRY("h2rcp"),
              CONDITIONAL_FACTORY_ENTRY(
                  CheckArgType(0, "__half2"),
                  CALL_FACTORY_ENTRY(
                      "h2rcp",
                      CALL(MapNames::getClNamespace() + "half2",
                           CALL(MapNames::getClNamespace(false, true) +
                                    "half_precision::recip",
                                CALL("float",
                                     ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                           CALL(MapNames::getClNamespace(false, true) +
                                    "half_precision::recip",
                                CALL("float",
                                     ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")))))),
                  CONDITIONAL_FACTORY_ENTRY(
                      math::UseBFloat16,
                      CALL_FACTORY_ENTRY(
                          "h2rcp",
                          CALL(MapNames::getClNamespace() + "marray<" +
                                   MapNames::getClNamespace() +
                                   "ext::oneapi::bfloat16, 2>",
                               CALL(MapNames::getClNamespace(false, true) +
                                        "half_precision::recip",
                                    CALL("float", ARRAY_SUBSCRIPT(
                                                      ARG(0), LITERAL("0")))),
                               CALL(MapNames::getClNamespace(false, true) +
                                        "half_precision::recip",
                                    CALL("float", ARRAY_SUBSCRIPT(
                                                      ARG(0), LITERAL("1")))))),
                      UNSUPPORT_FACTORY_ENTRY("h2rcp",
                                              Diagnostics::API_NOT_MIGRATED,
                                              ARG("h2rcp"))))))
      // h2rint
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2rint",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2rint"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2rint",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::rint",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2rint"),
                  CALL_FACTORY_ENTRY(
                      "h2rint",
                      CALL(MapNames::getClNamespace(false, true) + "rint",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2rint",
              CALL_FACTORY_ENTRY("h2rint",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::rint",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2rint",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "rint",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "rint",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2rsqrt
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2rsqrt",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2rsqrt"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2rsqrt",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::rsqrt",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2rsqrt"),
                  CALL_FACTORY_ENTRY(
                      "h2rsqrt",
                      CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2rsqrt",
              CALL_FACTORY_ENTRY("h2rsqrt",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::rsqrt",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2rsqrt",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2sin
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2sin",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2sin"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2sin",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::sin",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2sin"),
                  CALL_FACTORY_ENTRY(
                      "h2sin",
                      CALL(MapNames::getClNamespace(false, true) + "sin",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2sin",
              CALL_FACTORY_ENTRY("h2sin",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::sin",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2sin",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "sin",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "sin",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2sqrt
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2sqrt",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2sqrt"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2sqrt",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::sqrt",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2sqrt"),
                  CALL_FACTORY_ENTRY(
                      "h2sqrt",
                      CALL(MapNames::getClNamespace(false, true) + "sqrt",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2sqrt",
              CALL_FACTORY_ENTRY("h2sqrt",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::sqrt",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2sqrt",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "sqrt",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "sqrt",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // h2trunc
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "h2trunc",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("h2trunc"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("h2trunc",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::trunc",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("h2trunc"),
                  CALL_FACTORY_ENTRY(
                      "h2trunc",
                      CALL(MapNames::getClNamespace(false, true) + "trunc",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "h2trunc",
              CALL_FACTORY_ENTRY("h2trunc",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::trunc",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "h2trunc",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "trunc",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "trunc",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))};
}
