//===--------------- RewriterHalf2ArithmeticFunctions.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createHalf2ArithmeticFunctionsRewriterMap() {
  return RewriterMap{
      // __h2div
      MATH_API_REWRITER_DEVICE(
          "__h2div",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__h2div"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__h2div",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::h2div",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__h2div"),
              BINARY_OP_FACTORY_ENTRY("__h2div", BinaryOperatorKind::BO_Div,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __habs2
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "__habs2",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__habs2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__habs2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::habs2",
                                              ARG(0)))),
                  EMPTY_FACTORY_ENTRY("__habs2"),
                  CALL_FACTORY_ENTRY(
                      "__habs2",
                      CALL(MapNames::getClNamespace(false, true) + "fabs",
                           ARG(0))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__habs2",
              CALL_FACTORY_ENTRY("__habs2",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::fabs",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "__habs2",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace(false, true) + "fabs",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace(false, true) + "fabs",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // __hadd2
      MATH_API_REWRITER_DEVICE(
          "__hadd2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hadd2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hadd2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hadd2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hadd2"),
              BINARY_OP_FACTORY_ENTRY("__hadd2", BinaryOperatorKind::BO_Add,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __hadd2_rn
      MATH_API_REWRITER_DEVICE(
          "__hadd2_rn",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hadd2_rn"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hadd2_rn",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hadd2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hadd2_rn"),
              BINARY_OP_FACTORY_ENTRY("__hadd2_rn", BinaryOperatorKind::BO_Add,
                                      ARG(0), ARG(1))))
      // __hadd2_sat
      MATH_API_REWRITER_DEVICE(
          "__hadd2_sat",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hadd2_sat"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hadd2_sat",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hadd2_sat",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hadd2_sat"),
              CONDITIONAL_FACTORY_ENTRY(
                  CheckArgType(0, "__half2"),
                  CALL_FACTORY_ENTRY(
                      "__hadd2_sat",
                      CALL(MapNames::getDpctNamespace() + "clamp<" +
                               MapNames::getClNamespace() + "half2>",
                           BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                           LITERAL("{0.f, 0.f}"), LITERAL("{1.f, 1.f}"))),
                  CONDITIONAL_FACTORY_ENTRY(
                      UseSYCLCompat,
                      UNSUPPORT_FACTORY_ENTRY("__hadd2_sat",
                                              Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                              LITERAL("__hadd2_sat")),
                      CALL_FACTORY_ENTRY(
                          "__hadd2_sat",
                          CALL(MapNames::getDpctNamespace() + "clamp",
                               BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                               LITERAL("{0.f, 0.f}"),
                               LITERAL("{1.f, 1.f}")))))))
      // __hcmadd
      MATH_API_REWRITER_DEVICE(
          "__hcmadd",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hcmadd"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hcmadd",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hcmadd",
                                              ARG(0), ARG(1), ARG(2))))),
              EMPTY_FACTORY_ENTRY("__hcmadd"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hcmadd",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hcmadd")),
                  CALL_FACTORY_ENTRY("__hcmadd",
                                     CALL(MapNames::getDpctNamespace() +
                                              (DpctGlobalInfo::useSYCLCompat()
                                                   ? "cmul_add"
                                                   : "complex_mul_add"),
                                          ARG(0), ARG(1), ARG(2))))))
      // __hfma2
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "__hfma2",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hfma2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hfma2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hfma2",
                                              ARG(0), ARG(1), ARG(2)))),
                  EMPTY_FACTORY_ENTRY("__hfma2"),
                  CALL_FACTORY_ENTRY(
                      "__hfma2",
                      CALL(MapNames::getClNamespace(false, true) + "fma",
                           ARG(0), ARG(1), ARG(2))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hfma2",
              CALL_FACTORY_ENTRY("__hfma2",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::fma",
                                      ARG(0), ARG(1), ARG(2))),
              BINARY_OP_FACTORY_ENTRY("__hfma2", BinaryOperatorKind::BO_Add,
                                      BO(BinaryOperatorKind::BO_Mul,
                                         makeCallArgCreatorWithCall(0),
                                         makeCallArgCreatorWithCall(1)),
                                      makeCallArgCreatorWithCall(2))))
      // __hfma2_relu
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "__hfma2_relu",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hfma2_relu"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY(
                          "__hfma2_relu",
                          CALL(MapNames::getClNamespace() +
                                   "ext::intel::math::hfma2_relu",
                               ARG(0), ARG(1), ARG(2)))),
                  EMPTY_FACTORY_ENTRY("__hfma2_relu"),
                  CALL_FACTORY_ENTRY(
                      "__hfma2_relu",
                      CALL(MapNames::getDpctNamespace() + "relu",
                           CALL(MapNames::getClNamespace() + "fma", ARG(0),
                                ARG(1), ARG(2)))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hfma2_relu",
              CALL_FACTORY_ENTRY(
                  "__hfma2_relu",
                  CALL(MapNames::getDpctNamespace() + "relu",
                       CALL(MapNames::getClNamespace(false, true) +
                                "ext::oneapi::experimental::fma",
                            ARG(0), ARG(1), ARG(2)))),
              EMPTY_FACTORY_ENTRY("__hfma2_relu")))
      // __hfma2_sat
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "__hfma2_sat",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hfma2_sat"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hfma2_sat",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hfma2_sat",
                                              ARG(0), ARG(1), ARG(2)))),
                  EMPTY_FACTORY_ENTRY("__hfma2_sat"),
                  CALL_FACTORY_ENTRY(
                      "__hfma2_sat",
                      CALL(MapNames::getDpctNamespace() + "clamp<" +
                               MapNames::getClNamespace() + "half2>",
                           CALL(MapNames::getClNamespace() + "fma", ARG(0),
                                ARG(1), ARG(2)),
                           LITERAL("{0.f, 0.f}"), LITERAL("{1.f, 1.f}"))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hfma2_sat",
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hfma2_sat",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hfma2_sat")),
                  CALL_FACTORY_ENTRY(
                      "__hfma2_sat",
                      CALL(MapNames::getDpctNamespace() + "clamp",
                           CALL(MapNames::getClNamespace(false, true) +
                                    "ext::oneapi::experimental::fma",
                                ARG(0), ARG(1), ARG(2)),
                           LITERAL("{0.f, 0.f}"), LITERAL("{1.f, 1.f}")))),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hfma2_sat",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hfma2_sat")),
                  CALL_FACTORY_ENTRY(
                      "__hfma2_sat",
                      CALL(MapNames::getDpctNamespace() + "clamp",
                           BO(BinaryOperatorKind::BO_Add,
                              BO(BinaryOperatorKind::BO_Mul,
                                 makeCallArgCreatorWithCall(0),
                                 makeCallArgCreatorWithCall(1)),
                              makeCallArgCreatorWithCall(2)),
                           LITERAL("{0.f, 0.f}"), LITERAL("{1.f, 1.f}"))))))
      // __hmul2
      MATH_API_REWRITER_DEVICE(
          "__hmul2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hmul2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmul2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmul2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hmul2"),
              BINARY_OP_FACTORY_ENTRY("__hmul2", BinaryOperatorKind::BO_Mul,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __hmul2_rn
      MATH_API_REWRITER_DEVICE(
          "__hmul2_rn",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hmul2_rn"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmul2_rn",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmul2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hmul2_rn"),
              BINARY_OP_FACTORY_ENTRY("__hmul2_rn", BinaryOperatorKind::BO_Mul,
                                      ARG(0), ARG(1))))
      // __hmul2_sat
      MATH_API_REWRITER_DEVICE(
          "__hmul2_sat",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hmul2_sat"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmul2_sat",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmul2_sat",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hmul2_sat"),
              CONDITIONAL_FACTORY_ENTRY(
                  CheckArgType(0, "__half2"),
                  CALL_FACTORY_ENTRY(
                      "__hmul2_sat",
                      CALL(MapNames::getDpctNamespace() + "clamp<" +
                               MapNames::getClNamespace() + "half2>",
                           BO(BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                           LITERAL("{0.f, 0.f}"), LITERAL("{1.f, 1.f}"))),
                  CONDITIONAL_FACTORY_ENTRY(
                      UseSYCLCompat,
                      UNSUPPORT_FACTORY_ENTRY("__hmul2_sat",
                                              Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                              LITERAL("__hmul2_sat")),
                      CALL_FACTORY_ENTRY(
                          "__hmul2_sat",
                          CALL(MapNames::getDpctNamespace() + "clamp",
                               BO(BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                               LITERAL("{0.f, 0.f}"),
                               LITERAL("{1.f, 1.f}")))))))
      // __hneg2
      MATH_API_REWRITER_DEVICE(
          "__hneg2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hneg2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hneg2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hneg2",
                                              ARG(0))))),
              EMPTY_FACTORY_ENTRY("__hneg2"),
              UNARY_OP_FACTORY_ENTRY("__hneg2", UnaryOperatorKind::UO_Minus,
                                     makeCallArgCreatorWithCall(0))))
      // __hsub2
      MATH_API_REWRITER_DEVICE(
          "__hsub2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hsub2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hsub2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hsub2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hsub2"),
              BINARY_OP_FACTORY_ENTRY("__hsub2", BinaryOperatorKind::BO_Sub,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))
      // __hsub2_rn
      MATH_API_REWRITER_DEVICE(
          "__hsub2_rn",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hsub2_rn"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hsub2_rn",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hsub2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hsub2_rn"),
              BINARY_OP_FACTORY_ENTRY("__hsub2_rn", BinaryOperatorKind::BO_Sub,
                                      ARG(0), ARG(1))))
      // __hsub2_sat
      MATH_API_REWRITER_DEVICE(
          "__hsub2_sat",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hsub2_sat"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hsub2_sat",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hsub2_sat",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hsub2_sat"),
              CONDITIONAL_FACTORY_ENTRY(
                  CheckArgType(0, "__half2"),
                  CALL_FACTORY_ENTRY(
                      "__hsub2_sat",
                      CALL(MapNames::getDpctNamespace() + "clamp<" +
                               MapNames::getClNamespace() + "half2>",
                           BO(BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                           LITERAL("{0.f, 0.f}"), LITERAL("{1.f, 1.f}"))),
                  CONDITIONAL_FACTORY_ENTRY(
                      UseSYCLCompat,
                      UNSUPPORT_FACTORY_ENTRY("__hsub2_sat",
                                              Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                              LITERAL("__hsub2_sat")),
                      CALL_FACTORY_ENTRY(
                          "__hsub2_sat",
                          CALL(MapNames::getDpctNamespace() + "clamp",
                               BO(BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                               LITERAL("{0.f, 0.f}"),
                               LITERAL("{1.f, 1.f}")))))))
      // h2div
      MATH_API_REWRITER_DEVICE(
          "h2div",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("h2div"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("h2div",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::h2div",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("h2div"),
              BINARY_OP_FACTORY_ENTRY("h2div", BinaryOperatorKind::BO_Div,
                                      makeCallArgCreatorWithCall(0),
                                      makeCallArgCreatorWithCall(1))))};
}
