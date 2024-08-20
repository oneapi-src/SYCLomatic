//===--------------- RewriterSinglePrecisionIntrinsics.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createSinglePrecisionIntrinsicsRewriterMap() {
  return RewriterMap{
      // __expf
      MATH_API_REWRITER_DEVICE(
          "__expf",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "__expf",
                  CALL(MapNames::getClNamespace(false, true) + "native::exp",
                       CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0)))),
              EMPTY_FACTORY_ENTRY("__expf"), EMPTY_FACTORY_ENTRY("__expf"),
              EMPTY_FACTORY_ENTRY("__expf")))
      // __fadd_rd
      MATH_API_REWRITER_DEVICE(
          "__fadd_rd",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fadd_rd"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fadd_rd",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fadd_rd",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fadd_rd"),
              WARNING_FACTORY_ENTRY(
                  "__fadd_rd",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fadd_rd", BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fadd_rn
      MATH_API_REWRITER_DEVICE(
          "__fadd_rn",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fadd_rn"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fadd_rn",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fadd_rn",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fadd_rn"),
              WARNING_FACTORY_ENTRY(
                  "__fadd_rn",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fadd_rn", BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fadd_ru
      MATH_API_REWRITER_DEVICE(
          "__fadd_ru",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fadd_ru"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fadd_ru",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fadd_ru",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fadd_ru"),
              WARNING_FACTORY_ENTRY(
                  "__fadd_ru",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fadd_ru", BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fadd_rz
      MATH_API_REWRITER_DEVICE(
          "__fadd_rz",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fadd_rz"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fadd_rz",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fadd_rz",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fadd_rz"),
              WARNING_FACTORY_ENTRY(
                  "__fadd_rz",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fadd_rz", BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fmaf_rd
      MATH_API_REWRITERS_V2(
          "__fmaf_rd",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__fmaf_rd",
                  CALL(MapNames::getClNamespace() + "ext::intel::math::fmaf_rd",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              WARNING_FACTORY_ENTRY(
                  "__fmaf_rd",
                  CALL_FACTORY_ENTRY(
                      "__fmaf_rd",
                      CALL(MapNames::getClNamespace(false, true) + "fma",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0)),
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(1)),
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(2)))),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fmaf_rn
      MATH_API_REWRITERS_V2(
          "__fmaf_rn",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__fmaf_rn",
                  CALL(MapNames::getClNamespace() + "ext::intel::math::fmaf_rn",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              WARNING_FACTORY_ENTRY(
                  "__fmaf_rn",
                  CALL_FACTORY_ENTRY(
                      "__fmaf_rn",
                      CALL(MapNames::getClNamespace(false, true) + "fma",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0)),
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(1)),
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(2)))),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fmaf_ru
      MATH_API_REWRITERS_V2(
          "__fmaf_ru",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__fmaf_ru",
                  CALL(MapNames::getClNamespace() + "ext::intel::math::fmaf_ru",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              WARNING_FACTORY_ENTRY(
                  "__fmaf_ru",
                  CALL_FACTORY_ENTRY(
                      "__fmaf_ru",
                      CALL(MapNames::getClNamespace(false, true) + "fma",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0)),
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(1)),
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(2)))),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fmaf_rz
      MATH_API_REWRITERS_V2(
          "__fmaf_rz",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__fmaf_rz",
                  CALL(MapNames::getClNamespace() + "ext::intel::math::fmaf_rz",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              WARNING_FACTORY_ENTRY(
                  "__fmaf_rz",
                  CALL_FACTORY_ENTRY(
                      "__fmaf_rz",
                      CALL(MapNames::getClNamespace(false, true) + "fma",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0)),
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(1)),
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(2)))),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fmul_rd
      MATH_API_REWRITER_DEVICE(
          "__fmul_rd",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fmul_rd"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fmul_rd",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fmul_rd",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fmul_rd"),
              WARNING_FACTORY_ENTRY(
                  "__fmul_rd",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fmul_rd", BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fmul_rn
      MATH_API_REWRITER_DEVICE(
          "__fmul_rn",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fmul_rn"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fmul_rn",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fmul_rn",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fmul_rn"),
              WARNING_FACTORY_ENTRY(
                  "__fmul_rn",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fmul_rn", BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fmul_ru
      MATH_API_REWRITER_DEVICE(
          "__fmul_ru",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fmul_ru"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fmul_ru",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fmul_ru",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fmul_ru"),
              WARNING_FACTORY_ENTRY(
                  "__fmul_ru",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fmul_ru", BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fmul_rz
      MATH_API_REWRITER_DEVICE(
          "__fmul_rz",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fmul_rz"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fmul_rz",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fmul_rz",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fmul_rz"),
              WARNING_FACTORY_ENTRY(
                  "__fmul_rz",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fmul_rz", BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fsub_rd
      MATH_API_REWRITER_DEVICE(
          "__fsub_rd",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fsub_rd"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fsub_rd",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fsub_rd",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fsub_rd"),
              WARNING_FACTORY_ENTRY(
                  "__fsub_rd",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fsub_rd", BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fsub_rn
      MATH_API_REWRITER_DEVICE(
          "__fsub_rn",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fsub_rn"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fsub_rn",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fsub_rn",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fsub_rn"),
              WARNING_FACTORY_ENTRY(
                  "__fsub_rn",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fsub_rn", BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fsub_ru
      MATH_API_REWRITER_DEVICE(
          "__fsub_ru",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fsub_ru"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fsub_ru",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fsub_ru",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fsub_ru"),
              WARNING_FACTORY_ENTRY(
                  "__fsub_ru",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fsub_ru", BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fsub_rz
      MATH_API_REWRITER_DEVICE(
          "__fsub_rz",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__fsub_rz"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__fsub_rz",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::fsub_rz",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__fsub_rz"),
              WARNING_FACTORY_ENTRY(
                  "__fsub_rz",
                  BINARY_OP_FACTORY_ENTRY(
                      "__fsub_rz", BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))};
}
