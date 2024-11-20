//===---------- RewriterDoublePrecisionMathematicalFunctions.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createDoublePrecisionMathematicalFunctionsRewriterMap() {
  return RewriterMap{
      // cospi
      CALL_FACTORY_ENTRY(
          "cospi",
          CALL(MapNames::getClNamespace(false, true) + "cospi", ARG(0)))
      // cyl_bessel_i0
      MATH_API_REWRITER_DEVICE(
          "cyl_bessel_i0",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("cyl_bessel_i0"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "cyl_bessel_i0",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::cyl_bessel_i0",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("cyl_bessel_i0"),
              EMPTY_FACTORY_ENTRY("cyl_bessel_i0")))
      // cyl_bessel_i1
      MATH_API_REWRITER_DEVICE(
          "cyl_bessel_i1",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("cyl_bessel_i1"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "cyl_bessel_i1",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::cyl_bessel_i1",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("cyl_bessel_i1"),
              EMPTY_FACTORY_ENTRY("cyl_bessel_i1")))
      // erfcinv
      MATH_API_REWRITER_DEVICE(
          "erfcinv",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("erfcinv"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "erfcinv",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::erfcinv",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("erfcinv"), EMPTY_FACTORY_ENTRY("erfcinv")))
      // erfcx
      BINARY_OP_FACTORY_ENTRY(
          "erfcx", BinaryOperatorKind::BO_Mul,
          CALL(MapNames::getClNamespace(false, true) + "exp",
               BO(BinaryOperatorKind::BO_Mul, ARG(0), ARG(0))),
          CALL(MapNames::getClNamespace(false, true) + "erfc", ARG(0)))
      // erfinv
      MATH_API_REWRITER_DEVICE(
          "erfinv",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("erfinv"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "erfinv",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::erfinv",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("erfinv"), EMPTY_FACTORY_ENTRY("erfinv")))
      // j0
      MATH_API_REWRITER_DEVICE(
          "j0",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("j0"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "j0",
                      CALL(MapNames::getClNamespace() + "ext::intel::math::j0",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("j0"), EMPTY_FACTORY_ENTRY("j0")))
      // j1
      MATH_API_REWRITER_DEVICE(
          "j1",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("j1"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "j1",
                      CALL(MapNames::getClNamespace() + "ext::intel::math::j1",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("j1"), EMPTY_FACTORY_ENTRY("j1")))
      // jn
      MATH_API_REWRITERS_V2(
          "jn",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("jn", CALL(MapNames::getClNamespace() +
                                                "ext::intel::math::jn",
                                            ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::unsupported_warning,
              UNSUPPORT_FACTORY_ENTRY("jn", Diagnostics::API_NOT_MIGRATED,
                                      ARG("jn"))))
      // ldexp
      MATH_API_REWRITER_DEVICE(
          "ldexp",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "ldexp", CALL(MapNames::getClNamespace(false, true) + "ldexp",
                                CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0)),
                                ARG(1))),
              EMPTY_FACTORY_ENTRY("ldexp"), EMPTY_FACTORY_ENTRY("ldexp"),
              EMPTY_FACTORY_ENTRY("ldexp")))
      // norm
      MATH_API_REWRITER_DEVICE(
          "norm",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("norm"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("norm", CALL(MapNames::getClNamespace() +
                                                      "ext::intel::math::norm",
                                                  ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("norm"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  WARNING_FACTORY_ENTRY(
                      "norm",
                      CALL_FACTORY_ENTRY(
                          "norm", CALL(MapNames::getDpctNamespace() + "length",
                                       ARG(1), ARG(0))),
                      Diagnostics::MATH_EMULATION, std::string("norm"),
                      MapNames::getDpctNamespace() + "length"))))
      // norm3d
      MATH_API_REWRITER_DEVICE(
          "norm3d",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "norm3d",
                  CALL(MapNames::getClNamespace(false, true) + "length",
                       CALL(MapNames::getClNamespace() + "double3", ARG(0),
                            ARG(1), ARG(2)))),
              EMPTY_FACTORY_ENTRY("norm3d"), EMPTY_FACTORY_ENTRY("norm3d"),
              EMPTY_FACTORY_ENTRY("norm3d")))
      // norm4d
      MATH_API_REWRITER_DEVICE(
          "norm4d",
          MATH_API_DEVICE_NODES(
              CALL_FACTORY_ENTRY(
                  "norm4d",
                  CALL(MapNames::getClNamespace(false, true) + "length",
                       CALL(MapNames::getClNamespace() + "double4", ARG(0),
                            ARG(1), ARG(2), ARG(3)))),
              EMPTY_FACTORY_ENTRY("norm4d"), EMPTY_FACTORY_ENTRY("norm4d"),
              EMPTY_FACTORY_ENTRY("norm4d")))
      // normcdf
      MATH_API_REWRITER_DEVICE(
          "normcdf",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("normcdf"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "normcdf",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::cdfnorm",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("normcdf"),
              BINARY_OP_FACTORY_ENTRY(
                  "normcdf", BinaryOperatorKind::BO_Div,
                  CALL(MapNames::getClNamespace() + "erfc",
                       BO(BinaryOperatorKind::BO_Div,
                          CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0)),
                          makeLiteral("-" +
                                      MapNames::getClNamespace(false, true) +
                                      "sqrt(2.0)"))),
                  makeLiteral("2"))))
      // normcdfinv
      MATH_API_REWRITER_DEVICE(
          "normcdfinv",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("normcdfinv"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "normcdfinv",
                      CALL(MapNames::getClNamespace() +
                               "ext::intel::math::cdfnorminv",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("normcdfinv"),
              EMPTY_FACTORY_ENTRY("normcdfinv")))
      // rcbrt
      MATH_API_REWRITER_DEVICE(
          "rcbrt",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("rcbrt"),
              EMPTY_FACTORY_ENTRY("rcbrt"),
              EMPTY_FACTORY_ENTRY("rcbrt"),
              WARNING_FACTORY_ENTRY(
                  "rcbrt",
                  BINARY_OP_FACTORY_ENTRY(
                      "rcbrt", BinaryOperatorKind::BO_Div, makeLiteral("1"),
                      CALL(MapNames::getDpctNamespace() + "cbrt<double>",
                           ARG(0))),
                  Diagnostics::MATH_EMULATION, std::string("rcbrt"),
                  MapNames::getClNamespace(false, true) + "cbrt")))
      // rnorm
      MATH_API_REWRITER_DEVICE(
          "rnorm",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("rnorm"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("rnorm",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::rnorm",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("rnorm"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  WARNING_FACTORY_ENTRY(
                      "rnorm",
                      BINARY_OP_FACTORY_ENTRY(
                          "rnorm", BinaryOperatorKind::BO_Div, makeLiteral("1"),
                          CALL(MapNames::getDpctNamespace() + "length", ARG(1),
                               ARG(0))),
                      Diagnostics::MATH_EMULATION, std::string("rnorm"),
                      MapNames::getDpctNamespace() + "length"))))
      // rnorm3d
      MATH_API_REWRITER_DEVICE(
          "rnorm3d",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("rnorm3d"),
              EMPTY_FACTORY_ENTRY("rnorm3d"),
              EMPTY_FACTORY_ENTRY("rnorm3d"),
              BINARY_OP_FACTORY_ENTRY(
                  "rnorm3d", BinaryOperatorKind::BO_Div, makeLiteral("1"),
                  CALL(MapNames::getClNamespace(false, true) + "length",
                       CALL(MapNames::getClNamespace() + "double3", ARG(0),
                            ARG(1), ARG(2))))))
      // rnorm4d
      MATH_API_REWRITER_DEVICE(
          "rnorm4d",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("rnorm4d"),
              EMPTY_FACTORY_ENTRY("rnorm4d"),
              EMPTY_FACTORY_ENTRY("rnorm4d"),
              BINARY_OP_FACTORY_ENTRY(
                  "rnorm4d", BinaryOperatorKind::BO_Div, makeLiteral("1"),
                  CALL(MapNames::getClNamespace(false, true) + "length",
                       CALL(MapNames::getClNamespace() + "double4", ARG(0),
                            ARG(1), ARG(2), ARG(3))))))
      // sincospi
      CONDITIONAL_FACTORY_ENTRY(
          CheckParamType(1, "double *"),
          WARNING_FACTORY_ENTRY(
              "sincospi",
              MULTI_STMTS_FACTORY_ENTRY(
                  "sincospi", false, true, false, false,
                  BO(BinaryOperatorKind::BO_Assign, DEREF(ARG_WC(1)),
                     CALL(MapNames::getClNamespace() + "sincos",
                          BO(BinaryOperatorKind::BO_Mul,
                             CAST_IF_NOT_SAME(getDerefedType(1), ARG(0)),
                             makeLiteral(getPiString<true>())),
                          makeArgWithAddressSpaceCast(2)))),
              Diagnostics::MATH_EMULATION, std::string("sincospi"),
              MapNames::getClNamespace() + std::string("sincos")),
          WARNING_FACTORY_ENTRY(
              "sincospi",
              MULTI_STMTS_FACTORY_ENTRY(
                  "sincospi", false, true, false, false,
                  BO(BinaryOperatorKind::BO_Assign, DEREF(ARG_WC(1)),
                     CALL(MapNames::getClNamespace() + "sincos",
                          BO(BinaryOperatorKind::BO_Mul,
                             CAST_IF_NOT_SAME(getDerefedType(1), ARG(0)),
                             makeLiteral(getPiString<false>())),
                          makeArgWithAddressSpaceCast(2)))),
              Diagnostics::MATH_EMULATION, std::string("sincospi"),
              MapNames::getClNamespace() + std::string("sincos")))
      // sinpi
      CALL_FACTORY_ENTRY(
          "sinpi",
          CALL(MapNames::getClNamespace(false, true) + "sinpi", ARG(0)))
      // rsqrt
      MATH_API_REWRITER_HOST_DEVICE(
          MATH_API_REWRITER_HOST(
              "rsqrt",
              CALL_FACTORY_ENTRY(
                  "rsqrt", CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                                ARG(0)))),
          MATH_API_REWRITER_DEVICE(
              "rsqrt",
              MATH_API_DEVICE_NODES(
                  CALL_FACTORY_ENTRY(
                      "rsqrt",
                      CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                           ARG(0))),
                  EMPTY_FACTORY_ENTRY("rsqrt"), EMPTY_FACTORY_ENTRY("rsqrt"),
                  EMPTY_FACTORY_ENTRY("rsqrt"))))
      // y0
      MATH_API_REWRITER_DEVICE(
          "y0",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("y0"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "y0",
                      CALL(MapNames::getClNamespace() + "ext::intel::math::y0",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("y0"), EMPTY_FACTORY_ENTRY("y0")))
      // y1
      MATH_API_REWRITER_DEVICE(
          "y1",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("y1"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY(
                      "y1",
                      CALL(MapNames::getClNamespace() + "ext::intel::math::y1",
                           CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
              EMPTY_FACTORY_ENTRY("y1"), EMPTY_FACTORY_ENTRY("y1")))
      // yn
      MATH_API_REWRITERS_V2(
          "yn",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("yn", CALL(MapNames::getClNamespace() +
                                                "ext::intel::math::yn",
                                            ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::unsupported_warning,
              UNSUPPORT_FACTORY_ENTRY("yn", Diagnostics::API_NOT_MIGRATED,
                                      ARG("yn"))))};
}
