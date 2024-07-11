//===--------------- RewriterDoublePrecisionIntrinsics.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createDoublePrecisionIntrinsicsRewriterMap() {
  return RewriterMap{
      // __dadd_rd
      MATH_API_REWRITER_DEVICE(
          "__dadd_rd",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dadd_rd"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dadd_rd",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dadd_rd",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dadd_rd"),
              WARNING_FACTORY_ENTRY(
                  "__dadd_rd",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dadd_rd", BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __dadd_rn
      MATH_API_REWRITER_DEVICE(
          "__dadd_rn",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dadd_rn"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dadd_rn",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dadd_rn",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dadd_rn"),
              WARNING_FACTORY_ENTRY(
                  "__dadd_rn",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dadd_rn", BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __dadd_ru
      MATH_API_REWRITER_DEVICE(
          "__dadd_ru",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dadd_ru"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dadd_ru",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dadd_ru",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dadd_ru"),
              WARNING_FACTORY_ENTRY(
                  "__dadd_ru",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dadd_ru", BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __dadd_rz
      MATH_API_REWRITER_DEVICE(
          "__dadd_rz",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dadd_rz"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dadd_rz",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dadd_rz",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dadd_rz"),
              WARNING_FACTORY_ENTRY(
                  "__dadd_rz",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dadd_rz", BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __dmul_rd
      MATH_API_REWRITER_DEVICE(
          "__dmul_rd",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dmul_rd"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dmul_rd",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dmul_rd",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dmul_rd"),
              WARNING_FACTORY_ENTRY(
                  "__dmul_rd",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dmul_rd", BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __dmul_rn
      MATH_API_REWRITER_DEVICE(
          "__dmul_rn",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dmul_rn"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dmul_rn",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dmul_rn",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dmul_rn"),
              WARNING_FACTORY_ENTRY(
                  "__dmul_rn",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dmul_rn", BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __dmul_ru
      MATH_API_REWRITER_DEVICE(
          "__dmul_ru",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dmul_ru"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dmul_ru",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dmul_ru",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dmul_ru"),
              WARNING_FACTORY_ENTRY(
                  "__dmul_ru",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dmul_ru", BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __dmul_rz
      MATH_API_REWRITER_DEVICE(
          "__dmul_rz",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dmul_rz"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dmul_rz",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dmul_rz",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dmul_rz"),
              WARNING_FACTORY_ENTRY(
                  "__dmul_rz",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dmul_rz", BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __dsub_rd
      MATH_API_REWRITER_DEVICE(
          "__dsub_rd",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dsub_rd"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dsub_rd",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dsub_rd",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dsub_rd"),
              WARNING_FACTORY_ENTRY(
                  "__dsub_rd",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dsub_rd", BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __dsub_rn
      MATH_API_REWRITER_DEVICE(
          "__dsub_rn",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dsub_rn"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dsub_rn",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dsub_rn",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dsub_rn"),
              WARNING_FACTORY_ENTRY(
                  "__dsub_rn",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dsub_rn", BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __dsub_ru
      MATH_API_REWRITER_DEVICE(
          "__dsub_ru",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dsub_ru"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dsub_ru",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dsub_ru",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dsub_ru"),
              WARNING_FACTORY_ENTRY(
                  "__dsub_ru",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dsub_ru", BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __dsub_rz
      MATH_API_REWRITER_DEVICE(
          "__dsub_rz",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__dsub_rz"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__dsub_rz",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::dsub_rz",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__dsub_rz"),
              WARNING_FACTORY_ENTRY(
                  "__dsub_rz",
                  BINARY_OP_FACTORY_ENTRY(
                      "__dsub_rz", BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fma_rd
      MATH_API_REWRITERS_V2(
          "__fma_rd",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__fma_rd",
                  CALL(MapNames::getClNamespace() + "ext::intel::math::fma_rd",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              WARNING_FACTORY_ENTRY(
                  "__fma_rd",
                  CALL_FACTORY_ENTRY(
                      "__fma_rd",
                      CALL(MapNames::getClNamespace(false, true) + "fma",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0)),
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(1)),
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(2)))),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fma_rn
      MATH_API_REWRITERS_V2(
          "__fma_rn",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__fma_rn",
                  CALL(MapNames::getClNamespace() + "ext::intel::math::fma_rn",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              WARNING_FACTORY_ENTRY(
                  "__fma_rn",
                  CALL_FACTORY_ENTRY(
                      "__fma_rn",
                      CALL(MapNames::getClNamespace(false, true) + "fma",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0)),
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(1)),
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(2)))),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fma_ru
      MATH_API_REWRITERS_V2(
          "__fma_ru",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__fma_ru",
                  CALL(MapNames::getClNamespace() + "ext::intel::math::fma_ru",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              WARNING_FACTORY_ENTRY(
                  "__fma_ru",
                  CALL_FACTORY_ENTRY(
                      "__fma_ru",
                      CALL(MapNames::getClNamespace(false, true) + "fma",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0)),
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(1)),
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(2)))),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))
      // __fma_rz
      MATH_API_REWRITERS_V2(
          "__fma_rz",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__fma_rz",
                  CALL(MapNames::getClNamespace() + "ext::intel::math::fma_rz",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              WARNING_FACTORY_ENTRY(
                  "__fma_rz",
                  CALL_FACTORY_ENTRY(
                      "__fma_rz",
                      CALL(MapNames::getClNamespace(false, true) + "fma",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0)),
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(1)),
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(2)))),
                  Diagnostics::ROUNDING_MODE_UNSUPPORTED)))};
}
