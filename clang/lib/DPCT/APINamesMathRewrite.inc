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

CONDITIONAL_FACTORY_ENTRY(
    makeCheckOr(
        makeCheckAnd(
            HasDirectCallee(),
            makeCheckAnd(
                IsDirectCalleeHasAttribute<CUDADeviceAttr>(),
                makeCheckNot(IsDirectCalleeHasAttribute<CUDAHostAttr>()))),
        makeCheckOr(IsContextCallHasAttribute<CUDADeviceAttr>(),
                    IsContextCallHasAttribute<CUDAGlobalAttr>())),
    CONDITIONAL_FACTORY_ENTRY(
        math::UseStdLibdevice,
        HEADER_INSERT_FACTORY(HeaderType::HT_Stdlib, CALL_FACTORY_ENTRY("std::abs", CALL("std::abs", ARG(0)))),
        CONDITIONAL_FACTORY_ENTRY(
            IsParameterIntegerType(0),
            CALL_FACTORY_ENTRY(
                "std::abs",
                CALL(MapNames::getClNamespace(false, true) + "abs", ARG(0))),
            CALL_FACTORY_ENTRY(
                "std::abs",
                CALL(MapNames::getClNamespace(false, true) + "fabs", ARG(0))))),
    CONDITIONAL_FACTORY_ENTRY(
        math::IsDefinedInCUDA(),
        HEADER_INSERT_FACTORY(HeaderType::HT_Stdlib, NO_REWRITE_FUNCNAME_FACTORY_ENTRY("std::abs", "std::abs")),
        NO_REWRITE_FUNCNAME_FACTORY_ENTRY("std::abs", "std::abs")
    ))

CONDITIONAL_FACTORY_ENTRY(
    makeCheckOr(
        makeCheckAnd(
            HasDirectCallee(),
            makeCheckAnd(
                IsDirectCalleeHasAttribute<CUDADeviceAttr>(),
                makeCheckNot(IsDirectCalleeHasAttribute<CUDAHostAttr>()))),
        makeCheckOr(IsContextCallHasAttribute<CUDADeviceAttr>(),
                    IsContextCallHasAttribute<CUDAGlobalAttr>())),
    CONDITIONAL_FACTORY_ENTRY(
        math::UseStdLibdevice,
        HEADER_INSERT_FACTORY(HeaderType::HT_Stdlib, CALL_FACTORY_ENTRY("abs", CALL("std::abs", ARG(0)))),
        CONDITIONAL_FACTORY_ENTRY(
            IsParameterIntegerType(0),
            CALL_FACTORY_ENTRY(
                "abs",
                CALL(MapNames::getClNamespace(false, true) + "abs", ARG(0))),
            CALL_FACTORY_ENTRY(
                "abs",
                CALL(MapNames::getClNamespace(false, true) + "fabs", ARG(0))))),
    CONDITIONAL_FACTORY_ENTRY(
        math::IsDefinedInCUDA(),
        HEADER_INSERT_FACTORY(HeaderType::HT_Stdlib, NO_REWRITE_FUNCNAME_FACTORY_ENTRY("abs", "std::abs")),
        NO_REWRITE_FUNCNAME_FACTORY_ENTRY("abs", "abs")
    ))

CONDITIONAL_FACTORY_ENTRY(
    math::IsDefinedInCUDA(),
    CALL_FACTORY_ENTRY("saturate",
                       CALL(MapNames::getClNamespace(false, true) + "clamp<float>",
                            ARG(0), ARG("0.0f"), ARG("1.0f"))),
    NO_REWRITE_FUNCNAME_FACTORY_ENTRY("saturate", "saturate"))

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

MATH_API_REWRITER_DEVICE(
    "__hadd2_sat",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hadd2_sat"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hadd2_sat",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::hadd2_sat",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hadd2_sat"), EMPTY_FACTORY_ENTRY("__hadd2_sat")))

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
        EMPTY_FACTORY_ENTRY("__hfma2_sat"), EMPTY_FACTORY_ENTRY("__hfma2_sat")))

MATH_API_REWRITER_DEVICE(
    "__hmul2_sat",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hmul2_sat"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hmul2_sat",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::hmul2_sat",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hmul2_sat"), EMPTY_FACTORY_ENTRY("__hmul2_sat")))

MATH_API_REWRITER_DEVICE(
    "__hsub2_sat",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hsub2_sat"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hsub2_sat",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::hsub2_sat",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hsub2_sat"), EMPTY_FACTORY_ENTRY("__hsub2_sat")))

MATH_API_REWRITER_DEVICE(
    "__hbeq2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbeq2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbeq2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hbeq2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbeq2"), EMPTY_FACTORY_ENTRY("__hbeq2")))

MATH_API_REWRITER_DEVICE(
    "__hbequ2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbequ2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbequ2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbequ2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbequ2"), EMPTY_FACTORY_ENTRY("__hbequ2")))

MATH_API_REWRITER_DEVICE(
    "__hbge2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbge2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbge2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hbge2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbge2"), EMPTY_FACTORY_ENTRY("__hbge2")))

MATH_API_REWRITER_DEVICE(
    "__hbgeu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbgeu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbgeu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbgeu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbgeu2"), EMPTY_FACTORY_ENTRY("__hbgeu2")))

MATH_API_REWRITER_DEVICE(
    "__hbgt2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbgt2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbgt2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hbgt2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbgt2"), EMPTY_FACTORY_ENTRY("__hbgt2")))

MATH_API_REWRITER_DEVICE(
    "__hbgtu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbgtu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbgtu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbgtu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbgtu2"), EMPTY_FACTORY_ENTRY("__hbgtu2")))

MATH_API_REWRITER_DEVICE(
    "__hble2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hble2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hble2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hble2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hble2"), EMPTY_FACTORY_ENTRY("__hble2")))

MATH_API_REWRITER_DEVICE(
    "__hbleu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbleu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbleu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbleu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbleu2"), EMPTY_FACTORY_ENTRY("__hbleu2")))

MATH_API_REWRITER_DEVICE(
    "__hblt2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hblt2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hblt2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hblt2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hblt2"), EMPTY_FACTORY_ENTRY("__hblt2")))

MATH_API_REWRITER_DEVICE(
    "__hbltu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbltu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbltu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbltu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbltu2"), EMPTY_FACTORY_ENTRY("__hbltu2")))

MATH_API_REWRITER_DEVICE(
    "__hbne2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbne2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbne2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hbne2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbne2"), EMPTY_FACTORY_ENTRY("__hbne2")))

MATH_API_REWRITER_DEVICE(
    "__hbneu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbneu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbneu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbneu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbneu2"), EMPTY_FACTORY_ENTRY("__hbneu2")))

MATH_API_REWRITER_DEVICE(
    "__heq2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__heq2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__heq2", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::heq2",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__heq2"), EMPTY_FACTORY_ENTRY("__heq2")))

MATH_API_REWRITER_DEVICE(
    "__hequ2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hequ2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hequ2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hequ2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hequ2"), EMPTY_FACTORY_ENTRY("__hequ2")))

MATH_API_REWRITER_DEVICE(
    "__hge2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hge2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hge2", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hge2",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hge2"), EMPTY_FACTORY_ENTRY("__hge2")))

MATH_API_REWRITER_DEVICE(
    "__hgeu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hgeu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hgeu2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hgeu2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hgeu2"), EMPTY_FACTORY_ENTRY("__hgeu2")))

MATH_API_REWRITER_DEVICE(
    "__hgt2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hgt2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hgt2", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgt2",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hgt2"), EMPTY_FACTORY_ENTRY("__hgt2")))

MATH_API_REWRITER_DEVICE(
    "__hgtu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hgtu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hgtu2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hgtu2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hgtu2"), EMPTY_FACTORY_ENTRY("__hgtu2")))

MATH_API_REWRITER_DEVICE(
    "__hadd_sat",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hadd_sat"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hadd_sat",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::hadd_sat",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hadd_sat"), EMPTY_FACTORY_ENTRY("__hadd_sat")))

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
        EMPTY_FACTORY_ENTRY("__hfma_sat"), EMPTY_FACTORY_ENTRY("__hfma_sat")))

MATH_API_REWRITER_DEVICE(
    "__hmul_sat",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hmul_sat"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hmul_sat",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::hmul_sat",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hmul_sat"), EMPTY_FACTORY_ENTRY("__hmul_sat")))

MATH_API_REWRITER_DEVICE(
    "__hsub_sat",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hsub_sat"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hsub_sat",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::hsub_sat",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hsub_sat"), EMPTY_FACTORY_ENTRY("__hsub_sat")))

MATH_API_REWRITER_DEVICE(
    "__hequ",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hequ"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hequ", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hequ",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hequ"), EMPTY_FACTORY_ENTRY("__hequ")))

MATH_API_REWRITER_DEVICE(
    "__hgeu",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hgeu"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hgeu", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgeu",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hgeu"), EMPTY_FACTORY_ENTRY("__hgeu")))

MATH_API_REWRITER_DEVICE(
    "__hgtu",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hgtu"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hgtu", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgtu",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hgtu"), EMPTY_FACTORY_ENTRY("__hgtu")))

MATH_API_REWRITER_DEVICE(
    "__hleu",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hleu"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hleu", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hleu",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hleu"), EMPTY_FACTORY_ENTRY("__hleu")))

MATH_API_REWRITER_DEVICE(
    "__hltu",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hltu"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hltu", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hltu",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hltu"), EMPTY_FACTORY_ENTRY("__hltu")))

MATH_API_REWRITER_DEVICE(
    "__hneu",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hneu"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hneu", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hneu",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hneu"), EMPTY_FACTORY_ENTRY("__hneu")))
