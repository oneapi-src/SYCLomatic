//===--------------- APINamesMathRewrite.inc -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
                "rsqrt", CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                              ARG(0))),
            EMPTY_FACTORY_ENTRY("rsqrt"),
            EMPTY_FACTORY_ENTRY("rsqrt"),
            EMPTY_FACTORY_ENTRY("rsqrt")
        )
    )
)

MATH_API_REWRITER_HOST_DEVICE(
    MATH_API_REWRITER_HOST(
        "std::abs",
        HEADER_INSERT_FACTORY(
            HeaderType::HT_Stdlib,
            HEADER_INSERT_FACTORY(
                HeaderType::HT_Math,
                CALL_FACTORY_ENTRY("std::abs", CALL("std::abs", ARG(0)))))),
    MATH_API_REWRITER_DEVICE(
        "std::abs",
        MATH_API_DEVICE_NODES(
            CONDITIONAL_FACTORY_ENTRY(
                IsParameterIntegerType(0),
                CALL_FACTORY_ENTRY("std::abs",
                                   CALL(MapNames::getClNamespace(false, true) +
                                            "abs",
                                        ARG(0))),
                CALL_FACTORY_ENTRY("std::abs",
                                   CALL(MapNames::getClNamespace(false, true) +
                                            "fabs",
                                        ARG(0)))),
            EMPTY_FACTORY_ENTRY("std::abs"),
            HEADER_INSERT_FACTORY(
                HeaderType::HT_Stdlib,
                HEADER_INSERT_FACTORY(
                    HeaderType::HT_Math,
                    CALL_FACTORY_ENTRY("std::abs", CALL("std::abs", ARG(0))))),
            EMPTY_FACTORY_ENTRY("std::abs"))))

MATH_API_REWRITER_HOST_DEVICE(
    MATH_API_REWRITER_HOST(
        "abs", HEADER_INSERT_FACTORY(
                   HeaderType::HT_Stdlib,
                   HEADER_INSERT_FACTORY(
                       HeaderType::HT_Math,
                       CALL_FACTORY_ENTRY("abs", CALL("std::abs", ARG(0)))))),
    MATH_API_REWRITER_DEVICE(
        "abs",
        MATH_API_DEVICE_NODES(
            CONDITIONAL_FACTORY_ENTRY(
                IsParameterIntegerType(0),
                CALL_FACTORY_ENTRY(
                    "abs", CALL(MapNames::getClNamespace(false, true) + "abs",
                                ARG(0))),
                CALL_FACTORY_ENTRY(
                    "abs", CALL(MapNames::getClNamespace(false, true) + "fabs",
                                ARG(0)))),
            EMPTY_FACTORY_ENTRY("abs"),
            HEADER_INSERT_FACTORY(
                HeaderType::HT_Stdlib,
                HEADER_INSERT_FACTORY(HeaderType::HT_Math,
                                      CALL_FACTORY_ENTRY("abs", CALL("std::abs",
                                                                     ARG(0))))),
            EMPTY_FACTORY_ENTRY("abs"))))

MATH_API_REWRITER_DEVICE(
    "saturate",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY("saturate",
                           CALL(MapNames::getClNamespace(false, true) + "clamp<float>",
                                ARG(0), ARG("0.0f"), ARG("1.0f"))),
        EMPTY_FACTORY_ENTRY("saturate"),
        EMPTY_FACTORY_ENTRY("saturate"),
        EMPTY_FACTORY_ENTRY("saturate")
    )
)

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY("erfcinv",
                           CALL(MapNames::getClNamespace() +
                                    "ext::intel::math::erfcinv",
                                CAST_IF_NEED(makeLiteral("double"), ARG(0))))),
    ENTRY_UNSUPPORTED("erfcinv"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY("erfinv",
                           CALL(MapNames::getClNamespace() +
                                    "ext::intel::math::erfinv",
                                CAST_IF_NEED(makeLiteral("double"), ARG(0))))),
    ENTRY_UNSUPPORTED("erfinv"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY("normcdf",
                           CALL(MapNames::getClNamespace() +
                                    "ext::intel::math::cdfnorm",
                                CAST_IF_NEED(makeLiteral("double"), ARG(0))))),
    ENTRY_UNSUPPORTED("normcdf"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY("normcdfinv",
                           CALL(MapNames::getClNamespace() +
                                    "ext::intel::math::cdfnorminv",
                                CAST_IF_NEED(makeLiteral("double"), ARG(0))))),
    ENTRY_UNSUPPORTED("normcdfinv"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY(
            "norm", CALL(MapNames::getClNamespace() + "ext::intel::math::norm",
                         CAST_IF_NEED(makeLiteral("int"), ARG(0)),
                         CAST_IF_NEED(makeLiteral("const double *"), ARG(1))))),
    ENTRY_EMULATED("norm",
                   MapNames::getClNamespace(false, true) + "fast_length"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY(
            "rnorm",
            CALL(MapNames::getClNamespace() + "ext::intel::math::rnorm",
                 CAST_IF_NEED(makeLiteral("int"), ARG(0)),
                 CAST_IF_NEED(makeLiteral("const double *"), ARG(1))))),
    ENTRY_UNSUPPORTED("rnorm"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY("erfcinvf",
                           CALL(MapNames::getClNamespace() +
                                    "ext::intel::math::erfcinv",
                                CAST_IF_NEED(makeLiteral("float"), ARG(0))))),
    ENTRY_UNSUPPORTED("erfcinvf"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY("erfinvf",
                           CALL(MapNames::getClNamespace() +
                                    "ext::intel::math::erfinv",
                                CAST_IF_NEED(makeLiteral("float"), ARG(0))))),
    ENTRY_UNSUPPORTED("erfinvf"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY("normcdff",
                           CALL(MapNames::getClNamespace() +
                                    "ext::intel::math::cdfnorm",
                                CAST_IF_NEED(makeLiteral("float"), ARG(0))))),
    ENTRY_UNSUPPORTED("normcdff"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY("normcdfinvf",
                           CALL(MapNames::getClNamespace() +
                                    "ext::intel::math::cdfnorminv",
                                CAST_IF_NEED(makeLiteral("float"), ARG(0))))),
    ENTRY_UNSUPPORTED("normcdfinvf"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY(
            "normf", CALL(MapNames::getClNamespace() + "ext::intel::math::norm",
                          CAST_IF_NEED(makeLiteral("int"), ARG(0)),
                          CAST_IF_NEED(makeLiteral("const float *"), ARG(1))))),
    ENTRY_UNSUPPORTED("normf"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        CALL_FACTORY_ENTRY(
            "rnormf",
            CALL(MapNames::getClNamespace() + "ext::intel::math::rnorm",
                 CAST_IF_NEED(makeLiteral("int"), ARG(0)),
                 CAST_IF_NEED(makeLiteral("const float *"), ARG(1))))),
    ENTRY_UNSUPPORTED("rnormf"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        ENTRY_RENAMED("__hadd2_sat", MapNames::getClNamespace() +
                                         "ext::intel::math::" + "hadd2_sat")),
    ENTRY_UNSUPPORTED("__hadd2_sat"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        ENTRY_RENAMED("__hfma2_sat", MapNames::getClNamespace() +
                                         "ext::intel::math::" + "hfma2_sat")),
    ENTRY_UNSUPPORTED("__hfma2_sat"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        ENTRY_RENAMED("__hmul2_sat", MapNames::getClNamespace() +
                                         "ext::intel::math::" + "hmul2_sat")),
    ENTRY_UNSUPPORTED("__hmul2_sat"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(
        HeaderType::HT_SYCL_Math,
        ENTRY_RENAMED("__hsub2_sat", MapNames::getClNamespace() +
                                         "ext::intel::math::" + "hsub2_sat")),
    ENTRY_UNSUPPORTED("__hsub2_sat"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hbeq2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hbeq2")),
    ENTRY_UNSUPPORTED("__hbeq2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hbequ2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hbequ2")),
    ENTRY_UNSUPPORTED("__hbequ2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hbge2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hbge2")),
    ENTRY_UNSUPPORTED("__hbge2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hbgeu2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hbgeu2")),
    ENTRY_UNSUPPORTED("__hbgeu2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hbgt2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hbgt2")),
    ENTRY_UNSUPPORTED("__hbgt2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hbgtu2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hbgtu2")),
    ENTRY_UNSUPPORTED("__hbgtu2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hble2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hble2")),
    ENTRY_UNSUPPORTED("__hble2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hbleu2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hbleu2")),
    ENTRY_UNSUPPORTED("__hbleu2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hblt2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hblt2")),
    ENTRY_UNSUPPORTED("__hblt2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hbltu2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hbltu2")),
    ENTRY_UNSUPPORTED("__hbltu2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hbne2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hbne2")),
    ENTRY_UNSUPPORTED("__hbne2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hbneu2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hbneu2")),
    ENTRY_UNSUPPORTED("__hbneu2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__heq2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "heq2")),
    ENTRY_UNSUPPORTED("__heq2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hequ2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hequ2")),
    ENTRY_UNSUPPORTED("__hequ2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hge2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hge2")),
    ENTRY_UNSUPPORTED("__hge2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hgeu2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hgeu2")),
    ENTRY_UNSUPPORTED("__hgeu2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hgt2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hgt2")),
    ENTRY_UNSUPPORTED("__hgt2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hgtu2",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hgtu2")),
    ENTRY_UNSUPPORTED("__hgtu2"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hadd_sat",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hadd_sat")),
    ENTRY_UNSUPPORTED("__hadd_sat"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hfma_sat",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hfma_sat")),
    ENTRY_UNSUPPORTED("__hfma_sat"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hmul_sat",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hmul_sat")),
    ENTRY_UNSUPPORTED("__hmul_sat"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hsub_sat",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hsub_sat")),
    ENTRY_UNSUPPORTED("__hsub_sat"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hequ",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hequ")),
    ENTRY_UNSUPPORTED("__hequ"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hgeu",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hgeu")),
    ENTRY_UNSUPPORTED("__hgeu"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hgtu",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hgtu")),
    ENTRY_UNSUPPORTED("__hgtu"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hleu",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hleu")),
    ENTRY_UNSUPPORTED("__hleu"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hltu",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hltu")),
    ENTRY_UNSUPPORTED("__hltu"))

CONDITIONAL_FACTORY_ENTRY(
    math::UseIntelDeviceMath,
    HEADER_INSERT_FACTORY(HeaderType::HT_SYCL_Math,
                          ENTRY_RENAMED("__hneu",
                                        MapNames::getClNamespace() +
                                            "ext::intel::math::" + "hneu")),
    ENTRY_UNSUPPORTED("__hneu"))
