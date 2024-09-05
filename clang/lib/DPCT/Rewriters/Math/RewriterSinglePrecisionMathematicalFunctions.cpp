//===------ RewriterSinglePrecisionMathematicalFunctions.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createSinglePrecisionMathematicalFunctionsRewriterMap() {
  return RewriterMap{
      // cyl_bessel_i0f
      MATH_API_REWRITER_DEVICE(
          "cyl_bessel_i0f",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("cyl_bessel_i0f"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "cyl_bessel_i0f",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::cyl_bessel_i0",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("cyl_bessel_i0f"),
              EMPTY_FACTORY_ENTRY("cyl_bessel_i0f")))
      // cyl_bessel_i1f
      MATH_API_REWRITER_DEVICE(
          "cyl_bessel_i1f",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("cyl_bessel_i1f"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "cyl_bessel_i1f",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::cyl_bessel_i1",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("cyl_bessel_i1f"),
              EMPTY_FACTORY_ENTRY("cyl_bessel_i1f")))
      // erfcinvf
      MATH_API_REWRITER_DEVICE(
          "erfcinvf",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("erfcinvf"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "erfcinvf",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::erfcinv",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("erfcinvf"), EMPTY_FACTORY_ENTRY("erfcinvf")))
      // erfinvf
      MATH_API_REWRITER_DEVICE(
          "erfinvf",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("erfinvf"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "erfinvf",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::erfinv",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("erfinvf"), EMPTY_FACTORY_ENTRY("erfinvf")))
      // expf
      MATH_API_REWRITER_DEVICE(
          "expf",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "expf",
                  CALL(MapNames::getClNamespace(false, true) + "native::exp",
                       CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0)))),
              EMPTY_FACTORY_ENTRY("expf"), EMPTY_FACTORY_ENTRY("expf"),
              EMPTY_FACTORY_ENTRY("expf")))
      // j0f
      MATH_API_REWRITER_DEVICE(
          "j0f",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("j0f"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "j0f",
                      CALL(MapNames::getClNamespace() + "ext::intel::math::j0",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("j0f"), EMPTY_FACTORY_ENTRY("j0f")))
      // j1f
      MATH_API_REWRITER_DEVICE(
          "j1f",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("j1f"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "j1f",
                      CALL(MapNames::getClNamespace() + "ext::intel::math::j1",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("j1f"), EMPTY_FACTORY_ENTRY("j1f")))
      // jnf
      MATH_API_REWRITERS_V2(
          "jnf",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("jnf", CALL(MapNames::getClNamespace() +
                                                 "ext::intel::math::jn",
                                             ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::unsupported_warning,
              UNSUPPORT_FACTORY_ENTRY("jnf", Diagnostics::API_NOT_MIGRATED,
                                      ARG("jnf"))))
      // ldexpf
      MATH_API_REWRITER_DEVICE(
          "ldexpf",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "ldexpf",
                  CALL(MapNames::getClNamespace(false, true) + "ldexp",
                       CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0)), ARG(1))),
              EMPTY_FACTORY_ENTRY("ldexpf"), EMPTY_FACTORY_ENTRY("ldexp"),
              EMPTY_FACTORY_ENTRY("ldexpf")))
      // normcdff
      MATH_API_REWRITER_DEVICE(
          "normcdff",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("normcdff"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "normcdff",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::cdfnorm",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("normcdff"),
              BINARY_OP_FACTORY_ENTRY(
                  "normcdff", BinaryOperatorKind::BO_Div,
                  CALL(MapNames::getClNamespace() + "erfc",
                       BO(BinaryOperatorKind::BO_Div,
                          CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0)),
                          makeLiteral("-" +
                                      MapNames::getClNamespace(false, true) +
                                      "sqrt(2.0)"))),
                  makeLiteral("2"))))
      // normcdfinvf
      MATH_API_REWRITER_DEVICE(
          "normcdfinvf",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("normcdfinvf"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "normcdfinvf",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::cdfnorminv",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("normcdfinvf"),
              EMPTY_FACTORY_ENTRY("normcdfinvf")))
      // norm3df
      MATH_API_REWRITER_DEVICE(
          "norm3df",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "norm3df",
                  CALL(MapNames::getClNamespace(false, true) + "length",
                       CALL(MapNames::getClNamespace() + "float3", ARG(0),
                            ARG(1), ARG(2)))),
              EMPTY_FACTORY_ENTRY("norm3df"), EMPTY_FACTORY_ENTRY("norm3df"),
              EMPTY_FACTORY_ENTRY("norm3df")))
      // norm4df
      MATH_API_REWRITER_DEVICE(
          "norm4df",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "norm4df",
                  CALL(MapNames::getClNamespace(false, true) + "length",
                       CALL(MapNames::getClNamespace() + "float4", ARG(0),
                            ARG(1), ARG(2), ARG(3)))),
              EMPTY_FACTORY_ENTRY("norm4df"), EMPTY_FACTORY_ENTRY("norm4df"),
              EMPTY_FACTORY_ENTRY("norm4df")))
      // normf
      MATH_API_REWRITER_DEVICE(
          "normf",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("normf"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("normf", CALL(MapNames::getClNamespace() +
                                                       "ext::intel::math::norm",
                                                   ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("normf"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  WARNING_FACTORY_ENTRY(
                      "normf",
                      CALL_FACTORY_ENTRY(
                          "normf", CALL(MapNames::getDpctNamespace() + "length",
                                        ARG(1), ARG(0))),
                      Diagnostics::MATH_EMULATION, std::string("normf"),
                      MapNames::getDpctNamespace() + "length"))))
      // rcbrtf
      MATH_API_REWRITER_DEVICE(
          "rcbrtf",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("rcbrtf"),
              EMPTY_FACTORY_ENTRY("rcbrtf"),
              EMPTY_FACTORY_ENTRY("rcbrtf"),
              WARNING_FACTORY_ENTRY(
                  "rcbrtf",
                  CALL_FACTORY_ENTRY(
                      "rcbrtf",
                      CALL(MapNames::getClNamespace(false, true) +
                               "native::recip",
                           CALL(MapNames::getDpctNamespace() + "cbrt<float>",
                                ARG(0)))),
                  Diagnostics::MATH_EMULATION, std::string("rcbrtf"),
                  MapNames::getClNamespace(false, true) + "cbrt")))
      // rnorm3df
      MATH_API_REWRITER_DEVICE(
          "rnorm3df",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("rnorm3df"),
              EMPTY_FACTORY_ENTRY("rnorm3df"),
              EMPTY_FACTORY_ENTRY("rnorm3df"),
              CALL_FACTORY_ENTRY(
                  "rnorm3df",
                  CALL(MapNames::getClNamespace(false, true) + "native::recip",
                       CALL(MapNames::getClNamespace(false, true) + "length",
                            CALL(MapNames::getClNamespace() + "float3", ARG(0),
                                 ARG(1), ARG(2)))))))
      // rnorm4df
      MATH_API_REWRITER_DEVICE(
          "rnorm4df",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("rnorm4df"),
              EMPTY_FACTORY_ENTRY("rnorm4df"),
              EMPTY_FACTORY_ENTRY("rnorm4df"),
              CALL_FACTORY_ENTRY(
                  "rnorm4df",
                  CALL(MapNames::getClNamespace(false, true) + "native::recip",
                       CALL(MapNames::getClNamespace(false, true) + "length",
                            CALL(MapNames::getClNamespace() + "float4", ARG(0),
                                 ARG(1), ARG(2), ARG(3)))))))
      // rnormf
      MATH_API_REWRITER_DEVICE(
          "rnormf",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("rnormf"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("rnormf",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::rnorm",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("rnormf"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  WARNING_FACTORY_ENTRY(
                      "rnormf",
                      CALL_FACTORY_ENTRY(
                          "rnormf",
                          CALL(MapNames::getClNamespace(false, true) +
                                   "native::recip",
                               CALL(MapNames::getDpctNamespace() + "length",
                                    ARG(1), ARG(0)))),
                      Diagnostics::MATH_EMULATION, std::string("rnormf"),
                      MapNames::getDpctNamespace() + "length"))))
      // sincospif
      WARNING_FACTORY_ENTRY(
          "sincospif",
          MULTI_STMTS_FACTORY_ENTRY(
              "sincospif", false, true, false, false,
              BO(BinaryOperatorKind::BO_Assign, DEREF(ARG_WC(1)),
                 CALL(MapNames::getClNamespace() + "sincos",
                      BO(BinaryOperatorKind::BO_Mul,
                         CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0)),
                         makeLiteral(getPiString<false>())),
                      makeArgWithAddressSpaceCast(2)))),
          Diagnostics::MATH_EMULATION, std::string("sincospif"),
          MapNames::getClNamespace() + std::string("sincos"))
      // y0f
      MATH_API_REWRITER_DEVICE(
          "y0f",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("y0f"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "y0f",
                      CALL(MapNames::getClNamespace() + "ext::intel::math::y0",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("y0f"), EMPTY_FACTORY_ENTRY("y0f")))
      // y1f
      MATH_API_REWRITER_DEVICE(
          "y1f",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("y1f"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "y1f",
                      CALL(MapNames::getClNamespace() + "ext::intel::math::y1",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("y1f"), EMPTY_FACTORY_ENTRY("y1f")))
      // ynf
      MATH_API_REWRITERS_V2(
          "ynf",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("ynf", CALL(MapNames::getClNamespace() +
                                                 "ext::intel::math::yn",
                                             ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::unsupported_warning,
              UNSUPPORT_FACTORY_ENTRY("ynf", Diagnostics::API_NOT_MIGRATED,
                                      ARG("ynf"))))};
}
