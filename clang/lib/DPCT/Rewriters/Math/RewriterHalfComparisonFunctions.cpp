//===--------------- RewriterHalfComparisonFunctions.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createHalfComparisonFunctionsRewriterMap() {
  return RewriterMap{
      // __heq
      MATH_API_REWRITER_DEVICE(
          "__heq",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__heq"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__heq",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::heq",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__heq"),
              BINARY_OP_FACTORY_ENTRY("__heq", BinaryOperatorKind::BO_EQ,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __hequ
      MATH_API_REWRITER_DEVICE(
          "__hequ",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hequ"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hequ",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hequ",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hequ"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hequ",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1), LITERAL("std::equal_to<>()"))))))
      // __hge
      MATH_API_REWRITER_DEVICE(
          "__hge",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hge"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hge",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hge",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hge"),
              BINARY_OP_FACTORY_ENTRY("__hge", BinaryOperatorKind::BO_GE,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __hgeu
      MATH_API_REWRITER_DEVICE(
          "__hgeu",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hgeu"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hgeu",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgeu",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hgeu"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hgeu",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1),
                           LITERAL("std::greater_equal<>()"))))))
      // __hgt
      MATH_API_REWRITER_DEVICE(
          "__hgt",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hgt"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hgt",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgt",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hgt"),
              BINARY_OP_FACTORY_ENTRY("__hgt", BinaryOperatorKind::BO_GT,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __hgtu
      MATH_API_REWRITER_DEVICE(
          "__hgtu",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hgtu"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hgtu",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgtu",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hgtu"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hgtu",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1), LITERAL("std::greater<>()"))))))
      // __hisinf
      MATH_API_REWRITER_DEVICE(
          "__hisinf",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hisinf"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hisinf",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hisinf",
                                              ARG(0))))),
              EMPTY_FACTORY_ENTRY("__hisinf"),
              CONDITIONAL_FACTORY_ENTRY(
                  CheckArgType(0, "__half"),
                  CALL_FACTORY_ENTRY(
                      "__hisinf",
                      CALL(MapNames::getClNamespace(false, true) + "isinf",
                           ARG(0))),
                  CALL_FACTORY_ENTRY(
                      "__hisinf",
                      CALL(MapNames::getClNamespace(false, true) + "isinf",
                           CALL("float", ARG(0)))))))
      // __hisnan
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "__hisnan",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hisnan"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hisnan",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hisnan",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("__hisnan"),
                  CALL_FACTORY_ENTRY(
                      "__hisnan",
                      CALL(MapNames::getClNamespace(false, true) + "isnan",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hisnan",
              CALL_FACTORY_ENTRY("__hisnan",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::isnan",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "__hisnan",
                  CALL(MapNames::getClNamespace(false, true) + "isnan",
                       CALL("float", ARG(0))))))
      // __hle
      MATH_API_REWRITER_DEVICE(
          "__hle",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hle"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hle",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hle",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hle"),
              BINARY_OP_FACTORY_ENTRY("__hle", BinaryOperatorKind::BO_LE,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __hleu
      MATH_API_REWRITER_DEVICE(
          "__hleu",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hleu"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hleu",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hleu",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hleu"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hleu",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))
      // __hlt
      MATH_API_REWRITER_DEVICE(
          "__hlt",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hlt"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hlt",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hlt",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hlt"),
              BINARY_OP_FACTORY_ENTRY("__hlt", BinaryOperatorKind::BO_LT,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __hltu
      MATH_API_REWRITER_DEVICE(
          "__hltu",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hltu"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hltu",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hltu",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hltu"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hltu",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1), LITERAL("std::less<>()"))))))
      // __hmax
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "__hmax",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hmax"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmax",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmax",
                                              ARG(0), ARG(1)))),
                  EMPTY_FACTORY_ENTRY("__hmax"),
                  CALL_FACTORY_ENTRY("__hmax",
                                     CALL(MapNames::getClNamespace() + "fmax",
                                          ARG(0), ARG(1))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hmax",
              CALL_FACTORY_ENTRY("__hmax",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::fmax",
                                      ARG(0), ARG(1))),
              CALL_FACTORY_ENTRY(
                  "__hmax",
                  CALL(MapNames::getClNamespace(false, true) + "fmax",
                       CALL("float", ARG(0)), CALL("float", ARG(1))))))
      // __hmax_nan
      MATH_API_REWRITER_DEVICE(
          "__hmax_nan",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hmax_nan"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmax_nan",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmax_nan",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hmax_nan"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hmax_nan",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hmax_nan")),
                  CALL_FACTORY_ENTRY(
                      "__hmax_nan",
                      CALL(MapNames::getDpctNamespace() + "fmax_nan", ARG(0),
                           ARG(1))))))
      // __hmin
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "__hmin",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hmin"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmin",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmin",
                                              ARG(0), ARG(1)))),
                  EMPTY_FACTORY_ENTRY("__hmin"),
                  CALL_FACTORY_ENTRY("__hmin",
                                     CALL(MapNames::getClNamespace() + "fmin",
                                          ARG(0), ARG(1))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hmin",
              CALL_FACTORY_ENTRY("__hmin",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::fmin",
                                      ARG(0), ARG(1))),
              CALL_FACTORY_ENTRY(
                  "__hmin",
                  CALL(MapNames::getClNamespace(false, true) + "fmin",
                       CALL("float", ARG(0)), CALL("float", ARG(1))))))
      // __hmin_nan
      MATH_API_REWRITER_DEVICE(
          "__hmin_nan",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hmin_nan"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmin_nan",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmin_nan",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hmin_nan"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hmin_nan",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hmin_nan")),
                  CALL_FACTORY_ENTRY(
                      "__hmin_nan",
                      CALL(MapNames::getDpctNamespace() + "fmin_nan", ARG(0),
                           ARG(1))))))
      // __hne
      MATH_API_REWRITER_DEVICE(
          "__hne",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hne"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hne",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hne",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hne"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  // Notice: not equal compare need consider 'isnan'.
                  CALL_FACTORY_ENTRY(
                      "__hne",
                      CALL(MapNames::getDpctNamespace() + "compare", ARG(0),
                           ARG(1), LITERAL("std::not_equal_to<>()"))))))
      // __hneu
      MATH_API_REWRITER_DEVICE(
          "__hneu",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hneu"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hneu",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hneu",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hneu"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hneu",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1),
                           LITERAL("std::not_equal_to<>()"))))))};
}
