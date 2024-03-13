//===------------------- RewriterOverload.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createOverloadRewriterMap() {
  return RewriterMap{
      //__hadd
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckParamType(0, "int"),
          MATH_API_REWRITER_DEVICE(
              "__hadd",
              MATH_API_DEVICE_NODES(
                  CALL_FACTORY_ENTRY(
                      "__hadd",
                      CALL(MapNames::getClNamespace(false, true) + "hadd",
                           CAST_IF_NOT_SAME(LITERAL("int"), ARG(0)),
                           CAST_IF_NOT_SAME(LITERAL("int"), ARG(1)))),
                  EMPTY_FACTORY_ENTRY("__hadd"), EMPTY_FACTORY_ENTRY("__hadd"),
                  EMPTY_FACTORY_ENTRY("__hadd"))),
          MATH_API_REWRITER_DEVICE(
              "__hadd",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hadd"),
                  MATH_API_SPECIFIC_ELSE_EMU(
                      CheckArgType(0, "__half"),
                      HEADER_INSERT_FACTORY(
                          HeaderType::HT_SYCL_Math,
                          CALL_FACTORY_ENTRY("__hadd",
                                             CALL(MapNames::getClNamespace() +
                                                      "ext::intel::math::hadd",
                                                  ARG(0), ARG(1))))),
                  EMPTY_FACTORY_ENTRY("__hadd"),
                  BINARY_OP_FACTORY_ENTRY("__hadd", BinaryOperatorKind::BO_Add,
                                          ARG(0), ARG(1)))))

#define SINCOS_REWRITER(FuncName, CastType)                                    \
  MATH_API_REWRITER_HOST_DEVICE(                                               \
      MATH_API_REWRITER_HOST(                                                  \
          FuncName,                                                            \
          CONDITIONAL_FACTORY_ENTRY(                                           \
              CheckArgType(0, "int"),                                          \
              MULTI_STMTS_FACTORY_ENTRY(                                       \
                  FuncName, false, true, false, false,                         \
                  BO(BinaryOperatorKind::BO_Assign, DEREF(ARG_WC(1)),          \
                     CALL(MapNames::getClNamespace() + "sincos",               \
                          CAST(makeLiteral(CastType), ARG_WC(0)),              \
                          makeArgWithAddressSpaceCast(2)))),                   \
              MULTI_STMTS_FACTORY_ENTRY(                                       \
                  FuncName, false, true, false, false,                         \
                  BO(BinaryOperatorKind::BO_Assign, DEREF(ARG_WC(1)),          \
                     CALL(MapNames::getClNamespace() + "sincos", ARG_WC(0),    \
                          makeArgWithAddressSpaceCast(2)))))),                 \
      MATH_API_REWRITER_DEVICE_WITH_PERF(                                      \
          FuncName, SinCosPerfPred(),                                          \
          WARNING_FACTORY_ENTRY(                                               \
              FuncName,                                                        \
              CONDITIONAL_FACTORY_ENTRY(                                       \
                  CheckArgType(0, "int"),                                      \
                  MULTI_STMTS_FACTORY_ENTRY(                                   \
                      FuncName, false, true, false, true,                      \
                      BO(BinaryOperatorKind::BO_Assign, DEREF(ARG_WC(1)),      \
                         CALL(MapNames::getClNamespace() + "sin",              \
                              CAST(makeLiteral(CastType), ARG_WC(0)))),        \
                      BO(BinaryOperatorKind::BO_Assign, DEREF(ARG_WC(2)),      \
                         CALL(MapNames::getClNamespace() + "cos",              \
                              CAST(makeLiteral(CastType), ARG_WC(0))))),       \
                  MULTI_STMTS_FACTORY_ENTRY(                                   \
                      FuncName, false, true, false, true,                      \
                      BO(BinaryOperatorKind::BO_Assign, DEREF(ARG_WC(1)),      \
                         CALL(MapNames::getClNamespace() + "sin", ARG_WC(0))), \
                      BO(BinaryOperatorKind::BO_Assign, DEREF(ARG_WC(2)),      \
                         CALL(MapNames::getClNamespace() + "cos",              \
                              ARG_WC(0))))),                                   \
              Diagnostics::MATH_EMULATION, std::string(FuncName),              \
              MapNames::getClNamespace() + std::string("sin/cos")),            \
          MATH_API_DEVICE_NODES(                                               \
              CONDITIONAL_FACTORY_ENTRY(                                       \
                  CheckArgType(0, "int"),                                      \
                  MULTI_STMTS_FACTORY_ENTRY(                                   \
                      FuncName, false, true, false, false,                     \
                      BO(BinaryOperatorKind::BO_Assign, DEREF(ARG_WC(1)),      \
                         CALL(MapNames::getClNamespace() + "sincos",           \
                              CAST(makeLiteral(CastType), ARG_WC(0)),          \
                              makeArgWithAddressSpaceCast(2)))),               \
                  MULTI_STMTS_FACTORY_ENTRY(                                   \
                      FuncName, false, true, false, false,                     \
                      BO(BinaryOperatorKind::BO_Assign, DEREF(ARG_WC(1)),      \
                         CALL(MapNames::getClNamespace() + "sincos",           \
                              ARG_WC(0), makeArgWithAddressSpaceCast(2))))),   \
              EMPTY_FACTORY_ENTRY(FuncName), EMPTY_FACTORY_ENTRY(FuncName),    \
              EMPTY_FACTORY_ENTRY(FuncName))))
      // sincos
      SINCOS_REWRITER("sincos", "double")
      // sincosf
      SINCOS_REWRITER("sincosf", "float")
      // __sincosf
      SINCOS_REWRITER("__sincosf", "float")
      // ceil
      MATH_API_REWRITER_HOST_DEVICE(
          MATH_API_REWRITER_HOST(
              "ceil",
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_Math,
                  CALL_FACTORY_ENTRY("ceil", CALL("std::ceil", ARG(0))))),
          MATH_API_REWRITER_DEVICE(
              "ceil",
              MATH_API_DEVICE_NODES(
                  CALL_FACTORY_ENTRY(
                      "ceil",
                      CALL(MapNames::getClNamespace(false, true) + "ceil",
                           ARG(0))),
                  EMPTY_FACTORY_ENTRY("ceil"), EMPTY_FACTORY_ENTRY("ceil"),
                  EMPTY_FACTORY_ENTRY("ceil"))))

          MATH_API_REWRITER_HOST_DEVICE(
              MATH_API_REWRITER_HOST(
                  "ceilf",
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_Math,
                      CALL_FACTORY_ENTRY(
                          "ceilf",
                          CALL("std::ceil",
                               CAST_IF_NOT_SAME(LITERAL("float"), ARG(0)))))),
              MATH_API_REWRITER_DEVICE(
                  "ceilf",
                  MATH_API_DEVICE_NODES(
                      CALL_FACTORY_ENTRY(
                          "ceilf",
                          CALL(MapNames::getClNamespace(false, true) + "ceil",
                               CAST_IF_NOT_SAME(LITERAL("float"), ARG(0)))),
                      EMPTY_FACTORY_ENTRY("ceilf"),
                      EMPTY_FACTORY_ENTRY("ceilf"),
                      EMPTY_FACTORY_ENTRY("ceilf"))))

  };
}
