//===--------------- RewriterHalfArithmeticFunctions.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createHalfArithmeticFunctionsRewriterMap() {
  return RewriterMap{
      // __habs
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITERS_V2(
              "__habs",
              MATH_API_REWRITER_PAIR(
                  math::Tag::math_libdevice,
                  CALL_FACTORY_ENTRY("__habs",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::habs",
                                          ARG(0)))),
              MATH_API_REWRITER_PAIR(
                  math::Tag::emulation,
                  CALL_FACTORY_ENTRY(
                      "__habs",
                      CALL(MapNames::getClNamespace(false, true) + "fabs",
                           ARG(0))))),
          MATH_API_REWRITERS_V2(
              "__habs",
              MATH_API_REWRITER_PAIR(
                  math::Tag::ext_experimental,
                  CALL_FACTORY_ENTRY(
                      "__habs", CALL(MapNames::getClNamespace(false, true) +
                                         "ext::oneapi::experimental::fabs",
                                     ARG(0)))),
              MATH_API_REWRITER_PAIR(
                  math::Tag::emulation,
                  CALL_FACTORY_ENTRY(
                      "__habs",
                      CALL(MapNames::getClNamespace(false, true) + "fabs",
                           CALL("float", ARG(0)))))))
      // __hadd_rn
      MATH_API_REWRITERS_V2(
          "__hadd_rn",
          MATH_API_REWRITER_PAIR_WITH_COND(
              math::Tag::math_libdevice, CheckArgType(0, "__half"),
              CALL_FACTORY_ENTRY("__hadd_rn", CALL(MapNames::getClNamespace() +
                                                       "ext::intel::math::hadd",
                                                   ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              BINARY_OP_FACTORY_ENTRY("__hadd_rn", BinaryOperatorKind::BO_Add,
                                      ARG(0), ARG(1))))
      // __hadd_sat
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITERS_V2(
              "__hadd_sat",
              MATH_API_REWRITER_PAIR(
                  math::Tag::math_libdevice,
                  CALL_FACTORY_ENTRY("__hadd_sat",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::hadd_sat",
                                          ARG(0), ARG(1)))),
              MATH_API_REWRITER_PAIR(
                  math::Tag::emulation,
                  CALL_FACTORY_ENTRY(
                      "__hadd_sat",
                      CALL(MapNames::getDpctNamespace() + "clamp<" +
                               MapNames::getClNamespace() + "half>",
                           BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                           LITERAL("0.f"), LITERAL("1.0f"))))),
          MATH_API_REWRITERS_V2(
              "__hadd_sat",
              MATH_API_REWRITER_PAIR_WITH_COND(
                  math::Tag::emulation, math::UseBFloat16,
                  CALL_FACTORY_ENTRY(
                      "__hadd_sat",
                      CALL(MapNames::getDpctNamespace() + "clamp<" +
                               MapNames::getClNamespace() +
                               "ext::oneapi::bfloat16>",
                           BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                           LITERAL("0.f"), LITERAL("1.0f")))),
              MATH_API_REWRITER_PAIR(
                  math::Tag::unsupported_warning,
                  UNSUPPORT_FACTORY_ENTRY("__hadd_sat",
                                          Diagnostics::API_NOT_MIGRATED,
                                          ARG("__hadd_sat")))))
      // __hdiv
      MATH_API_REWRITERS_V2(
          "__hdiv",
          MATH_API_REWRITER_PAIR_WITH_COND(
              math::Tag::math_libdevice, CheckArgType(0, "__half"),
              CALL_FACTORY_ENTRY("__hdiv", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hdiv",
                                                ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              BINARY_OP_FACTORY_ENTRY("__hdiv", BinaryOperatorKind::BO_Div,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __hfma
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "__hfma",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hfma"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hfma",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hfma",
                                              ARG(0), ARG(1)))),
                  EMPTY_FACTORY_ENTRY("__hfma"),
                  CALL_FACTORY_ENTRY(
                      "__hfma",
                      CALL(MapNames::getClNamespace(false, true) + "fma",
                           ARG(0), ARG(1), ARG(2))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hfma",
              CALL_FACTORY_ENTRY("__hfma",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::fma",
                                      ARG(0), ARG(1), ARG(2))),
              BINARY_OP_FACTORY_ENTRY("__hfma", BinaryOperatorKind::BO_Add,
                                      BO(BinaryOperatorKind::BO_Mul,
                                         makeCallArgCreatorWithCall(0),
                                         makeCallArgCreatorWithCall(1)),
                                      makeCallArgCreatorWithCall(2))))
      // __hfma_relu
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "__hfma_relu",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hfma_relu"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hfma_relu",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hfma_relu",
                                              ARG(0), ARG(1), ARG(2)))),
                  EMPTY_FACTORY_ENTRY("__hfma_relu"),
                  CALL_FACTORY_ENTRY(
                      "__hfma_relu",
                      CALL(MapNames::getDpctNamespace() + "relu",
                           CALL(MapNames::getClNamespace() + "fma", ARG(0),
                                ARG(1), ARG(2)))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hfma_relu",
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hfma_relu",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hfma_relu")),
                  CALL_FACTORY_ENTRY(
                      "__hfma_relu",
                      CALL(MapNames::getDpctNamespace() + "relu",
                           CALL(MapNames::getClNamespace(false, true) +
                                    "ext::oneapi::experimental::fma",
                                ARG(0), ARG(1), ARG(2))))),
              EMPTY_FACTORY_ENTRY("__hfma_relu")))
      // __hfma_sat
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half"),
          MATH_API_REWRITER_DEVICE(
              "__hfma_sat",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hfma_sat"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hfma_sat",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hfma_sat",
                                              ARG(0), ARG(1), ARG(2)))),
                  EMPTY_FACTORY_ENTRY("__hfma_sat"),
                  CALL_FACTORY_ENTRY(
                      "__hfma_sat",
                      CALL(MapNames::getDpctNamespace() + "clamp<" +
                               MapNames::getClNamespace() + "half>",
                           CALL(MapNames::getClNamespace() + "fma", ARG(0),
                                ARG(1), ARG(2)),
                           LITERAL("0.f"), LITERAL("1.0f"))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hfma_sat",
              CALL_FACTORY_ENTRY(
                  "__hfma_sat",
                  CALL(MapNames::getDpctNamespace() + "clamp<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16>",
                       CALL(MapNames::getClNamespace(false, true) +
                                "ext::oneapi::experimental::fma",
                            ARG(0), ARG(1), ARG(2)),
                       LITERAL("0.f"), LITERAL("1.0f"))),
              CALL_FACTORY_ENTRY("__hfma_sat",
                                 CALL(MapNames::getDpctNamespace() + "clamp<" +
                                          MapNames::getClNamespace() +
                                          "ext::oneapi::bfloat16>",
                                      BO(BinaryOperatorKind::BO_Add,
                                         BO(BinaryOperatorKind::BO_Mul,
                                            makeCallArgCreatorWithCall(0),
                                            makeCallArgCreatorWithCall(1)),
                                         makeCallArgCreatorWithCall(2)),
                                      LITERAL("0.f"), LITERAL("1.0f")))))
      // __hmul
      MATH_API_REWRITER_DEVICE(
          "__hmul",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hmul"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmul",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmul",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hmul"),
              BINARY_OP_FACTORY_ENTRY("__hmul", BinaryOperatorKind::BO_Mul,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __hmul_rn
      MATH_API_REWRITER_DEVICE(
          "__hmul_rn",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hmul_rn"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmul_rn",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmul",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hmul_rn"),
              BINARY_OP_FACTORY_ENTRY("__hmul_rn", BinaryOperatorKind::BO_Mul,
                                      ARG(0), ARG(1))))
      // __hmul_sat
      MATH_API_REWRITER_DEVICE(
          "__hmul_sat",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hmul_sat"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmul_sat",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmul_sat",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hmul_sat"),
              CONDITIONAL_FACTORY_ENTRY(
                  CheckArgType(0, "__half"),
                  CALL_FACTORY_ENTRY(
                      "__hmul_sat",
                      CALL(MapNames::getDpctNamespace() + "clamp<" +
                               MapNames::getClNamespace() + "half>",
                           BO(BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                           LITERAL("0.f"), LITERAL("1.0f"))),
                  CONDITIONAL_FACTORY_ENTRY(
                      math::UseBFloat16,
                      CALL_FACTORY_ENTRY(
                          "__hmul_sat",
                          CALL(MapNames::getDpctNamespace() + "clamp<" +
                                   MapNames::getClNamespace() +
                                   "ext::oneapi::bfloat16>",
                               BO(BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                               LITERAL("0.f"), LITERAL("1.0f"))),
                      UNSUPPORT_FACTORY_ENTRY("__hmul_sat",
                                              Diagnostics::API_NOT_MIGRATED,
                                              ARG("__hmul_sat"))))))
      // __hneg
      MATH_API_REWRITER_DEVICE(
          "__hneg",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hneg"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hneg",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hneg",
                                              ARG(0))))),
              EMPTY_FACTORY_ENTRY("__hneg"),
              UNARY_OP_FACTORY_ENTRY("__hneg", UnaryOperatorKind::UO_Minus,
                                     makeCallArgCreatorWithCall(0))))
      // __hsub
      MATH_API_REWRITER_DEVICE(
          "__hsub",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hsub"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hsub",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hsub",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hsub"),
              BINARY_OP_FACTORY_ENTRY("__hsub", BinaryOperatorKind::BO_Sub,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __hsub_rn
      MATH_API_REWRITER_DEVICE(
          "__hsub_rn",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hsub_rn"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hsub_rn",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hsub",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hsub_rn"),
              BINARY_OP_FACTORY_ENTRY("__hsub_rn", BinaryOperatorKind::BO_Sub,
                                      ARG(0), ARG(1))))
      // __hsub_sat
      MATH_API_REWRITER_DEVICE(
          "__hsub_sat",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hsub_sat"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hsub_sat",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hsub_sat",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hsub_sat"),
              CONDITIONAL_FACTORY_ENTRY(
                  CheckArgType(0, "__half"),
                  CALL_FACTORY_ENTRY(
                      "__hsub_sat",
                      CALL(MapNames::getDpctNamespace() + "clamp<" +
                               MapNames::getClNamespace() + "half>",
                           BO(BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                           LITERAL("0.f"), LITERAL("1.0f"))),
                  CONDITIONAL_FACTORY_ENTRY(
                      math::UseBFloat16,
                      CALL_FACTORY_ENTRY(
                          "__hsub_sat",
                          CALL(MapNames::getDpctNamespace() + "clamp<" +
                                   MapNames::getClNamespace() +
                                   "ext::oneapi::bfloat16>",
                               BO(BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                               LITERAL("0.f"), LITERAL("1.0f"))),
                      UNSUPPORT_FACTORY_ENTRY("__hsub_sat",
                                              Diagnostics::API_NOT_MIGRATED,
                                              ARG("__hsub_sat"))))))
      // hdiv
      MATH_API_REWRITER_DEVICE(
          "hdiv",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("hdiv"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("hdiv", CALL(MapNames::getClNamespace() +
                                                      "ext::intel::math::hdiv",
                                                  ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("hdiv"),
              BINARY_OP_FACTORY_ENTRY("hdiv", BinaryOperatorKind::BO_Div,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))};
}
