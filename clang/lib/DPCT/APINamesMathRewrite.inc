//===--------------- APINamesMathRewrite.inc -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// STD
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
            EMPTY_FACTORY_ENTRY("std::abs"), EMPTY_FACTORY_ENTRY("std::abs"),
            EMPTY_FACTORY_ENTRY("std::abs"))))

// Half Arithmetic Functions
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
        EMPTY_FACTORY_ENTRY("__hadd_sat"),
        CALL_FACTORY_ENTRY("__hadd_sat",
                           CALL(MapNames::getClNamespace() + "clamp<" +
                                    MapNames::getClNamespace() + "half>",
                                BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                                LITERAL("0.f"), LITERAL("1.0f")))))

MATH_API_REWRITER_DEVICE(
    "__hdiv",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hdiv"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hdiv", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hdiv",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hdiv"),
        BINARY_OP_FACTORY_ENTRY("__hdiv", BinaryOperatorKind::BO_Div,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

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
        CALL_FACTORY_ENTRY("__hfma_sat",
                           CALL(MapNames::getClNamespace() + "clamp<" +
                                    MapNames::getClNamespace() + "half>",
                                CALL(MapNames::getClNamespace() + "fma", ARG(0),
                                     ARG(1), ARG(2)),
                                LITERAL("0.f"), LITERAL("1.0f")))))

MATH_API_REWRITER_DEVICE(
    "__hmul",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hmul"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hmul", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmul",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hmul"),
        BINARY_OP_FACTORY_ENTRY("__hmul", BinaryOperatorKind::BO_Mul,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

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
        EMPTY_FACTORY_ENTRY("__hmul_sat"),
        CALL_FACTORY_ENTRY("__hmul_sat",
                           CALL(MapNames::getClNamespace() + "clamp<" +
                                    MapNames::getClNamespace() + "half>",
                                BO(BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                                LITERAL("0.f"), LITERAL("1.0f")))))

MATH_API_REWRITER_DEVICE(
    "__hneg",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hneg"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hneg", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hneg",
                                              ARG(0)))),
        EMPTY_FACTORY_ENTRY("__hneg"),
        UNARY_OP_FACTORY_ENTRY("__hneg", UnaryOperatorKind::UO_Minus,
                               makeCallArgCreatorWithCall(0))))

MATH_API_REWRITER_DEVICE(
    "__hsub",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hsub"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hsub", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hsub",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hsub"),
        BINARY_OP_FACTORY_ENTRY("__hsub", BinaryOperatorKind::BO_Sub,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

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
        EMPTY_FACTORY_ENTRY("__hsub_sat"),
        CALL_FACTORY_ENTRY("__hsub_sat",
                           CALL(MapNames::getClNamespace() + "clamp<" +
                                    MapNames::getClNamespace() + "half>",
                                BO(BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                                LITERAL("0.f"), LITERAL("1.0f")))))

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
                                makeCallArgCreatorWithCall(1))))

// Half2 Arithmetic Functions
MATH_API_REWRITER_DEVICE(
    "__h2div",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__h2div"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__h2div", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::h2div",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__h2div"),
        BINARY_OP_FACTORY_ENTRY("__h2div", BinaryOperatorKind::BO_Div,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

MATH_API_REWRITER_DEVICE(
    "__hadd2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hadd2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hadd2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hadd2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hadd2"),
        BINARY_OP_FACTORY_ENTRY("__hadd2", BinaryOperatorKind::BO_Add,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

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
        EMPTY_FACTORY_ENTRY("__hadd2_sat"),
        CALL_FACTORY_ENTRY("__hadd2_sat",
                           CALL(MapNames::getClNamespace() + "clamp<" +
                                    MapNames::getClNamespace() + "half2>",
                                BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                                LITERAL("{0.f, 0.f}"), LITERAL("{1.f, 1.f}")))))

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
        CALL_FACTORY_ENTRY("__hfma2_sat",
                           CALL(MapNames::getClNamespace() + "clamp<" +
                                    MapNames::getClNamespace() + "half2>",
                                CALL(MapNames::getClNamespace() + "fma", ARG(0),
                                     ARG(1), ARG(2)),
                                LITERAL("{0.f, 0.f}"), LITERAL("{1.f, 1.f}")))))

MATH_API_REWRITER_DEVICE(
    "__hmul2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hmul2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hmul2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hmul2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hmul2"),
        BINARY_OP_FACTORY_ENTRY("__hmul2", BinaryOperatorKind::BO_Mul,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

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
        EMPTY_FACTORY_ENTRY("__hmul2_sat"),
        CALL_FACTORY_ENTRY("__hmul2_sat",
                           CALL(MapNames::getClNamespace() + "clamp<" +
                                    MapNames::getClNamespace() + "half2>",
                                BO(BinaryOperatorKind::BO_Mul, ARG(0), ARG(1)),
                                LITERAL("{0.f, 0.f}"), LITERAL("{1.f, 1.f}")))))

MATH_API_REWRITER_DEVICE(
    "__hneg2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hneg2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hneg2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hneg2",
                                               ARG(0)))),
        EMPTY_FACTORY_ENTRY("__hneg2"),
        UNARY_OP_FACTORY_ENTRY("__hneg2", UnaryOperatorKind::UO_Minus,
                               makeCallArgCreatorWithCall(0))))

MATH_API_REWRITER_DEVICE(
    "__hsub2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hsub2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hsub2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hsub2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hsub2"),
        BINARY_OP_FACTORY_ENTRY("__hsub2", BinaryOperatorKind::BO_Sub,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

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
        EMPTY_FACTORY_ENTRY("__hsub2_sat"),
        CALL_FACTORY_ENTRY("__hsub2_sat",
                           CALL(MapNames::getClNamespace() + "clamp<" +
                                    MapNames::getClNamespace() + "half2>",
                                BO(BinaryOperatorKind::BO_Sub, ARG(0), ARG(1)),
                                LITERAL("{0.f, 0.f}"), LITERAL("{1.f, 1.f}")))))

MATH_API_REWRITER_DEVICE(
    "h2div",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("h2div"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("h2div", CALL(MapNames::getClNamespace() +
                                                 "ext::intel::math::h2div",
                                             ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("h2div"),
        BINARY_OP_FACTORY_ENTRY("h2div", BinaryOperatorKind::BO_Div,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

// Half Comparison Functions
MATH_API_REWRITER_DEVICE(
    "__heq",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__heq"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__heq", CALL(MapNames::getClNamespace() +
                                                 "ext::intel::math::heq",
                                             ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__heq"),
        BINARY_OP_FACTORY_ENTRY("__heq", BinaryOperatorKind::BO_EQ,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

MATH_API_REWRITER_DEVICE(
    "__hequ",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hequ"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hequ", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hequ",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hequ"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half,
            CALL_FACTORY_ENTRY("__hequ", CALL(MapNames::getDpctNamespace() +
                                                  "unordered_compare_half",
                                              ARG(0), ARG(1),
                                              LITERAL("std::equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hge",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hge"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hge", CALL(MapNames::getClNamespace() +
                                                 "ext::intel::math::hge",
                                             ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hge"),
        BINARY_OP_FACTORY_ENTRY("__hge", BinaryOperatorKind::BO_GE,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

MATH_API_REWRITER_DEVICE(
    "__hgeu",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hgeu"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hgeu", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgeu",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hgeu"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half,
            CALL_FACTORY_ENTRY(
                "__hgeu",
                CALL(MapNames::getDpctNamespace() + "unordered_compare_half",
                     ARG(0), ARG(1), LITERAL("std::greater_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hgt",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hgt"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hgt", CALL(MapNames::getClNamespace() +
                                                 "ext::intel::math::hgt",
                                             ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hgt"),
        BINARY_OP_FACTORY_ENTRY("__hgt", BinaryOperatorKind::BO_GT,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

MATH_API_REWRITER_DEVICE(
    "__hgtu",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hgtu"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hgtu", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgtu",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hgtu"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half,
            CALL_FACTORY_ENTRY("__hgtu", CALL(MapNames::getDpctNamespace() +
                                                  "unordered_compare_half",
                                              ARG(0), ARG(1),
                                              LITERAL("std::greater<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hle",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hle"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hle", CALL(MapNames::getClNamespace() +
                                                 "ext::intel::math::hle",
                                             ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hle"),
        BINARY_OP_FACTORY_ENTRY("__hle", BinaryOperatorKind::BO_LE,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

MATH_API_REWRITER_DEVICE(
    "__hleu",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hleu"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hleu", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hleu",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hleu"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half,
            CALL_FACTORY_ENTRY(
                "__hleu",
                CALL(MapNames::getDpctNamespace() + "unordered_compare_half",
                     ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hlt",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hlt"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hlt", CALL(MapNames::getClNamespace() +
                                                 "ext::intel::math::hlt",
                                             ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hlt"),
        BINARY_OP_FACTORY_ENTRY("__hlt", BinaryOperatorKind::BO_LT,
                                makeCallArgCreatorWithCall(0),
                                makeCallArgCreatorWithCall(1))))

MATH_API_REWRITER_DEVICE(
    "__hltu",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hltu"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hltu", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hltu",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hltu"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half,
            CALL_FACTORY_ENTRY("__hltu", CALL(MapNames::getDpctNamespace() +
                                                  "unordered_compare_half",
                                              ARG(0), ARG(1),
                                              LITERAL("std::less<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hne",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hne"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hne", CALL(MapNames::getClNamespace() +
                                                 "ext::intel::math::hne",
                                             ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hne"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half,
            // Notice: not equal compare need consider 'isnan'.
            CALL_FACTORY_ENTRY(
                "__hne",
                CALL(MapNames::getDpctNamespace() + "compare_half", ARG(0),
                     ARG(1), LITERAL("std::not_equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hneu",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hneu"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hneu", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hneu",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hneu"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half,
            CALL_FACTORY_ENTRY(
                "__hneu",
                CALL(MapNames::getDpctNamespace() + "unordered_compare_half",
                     ARG(0), ARG(1), LITERAL("std::not_equal_to<>()"))))))

// Half2 Comparison Functions
MATH_API_REWRITER_DEVICE(
    "__hbeq2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbeq2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbeq2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hbeq2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbeq2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_both_half2,
            CALL_FACTORY_ENTRY("__hbeq2", CALL(MapNames::getDpctNamespace() +
                                                   "compare_both_half2",
                                               ARG(0), ARG(1),
                                               LITERAL("std::equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hbequ2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbequ2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbequ2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbequ2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbequ2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_both_half2,
            CALL_FACTORY_ENTRY("__hbequ2",
                               CALL(MapNames::getDpctNamespace() +
                                        "unordered_compare_both_half2",
                                    ARG(0), ARG(1),
                                    LITERAL("std::equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hbge2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbge2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbge2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hbge2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbge2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_both_half2,
            CALL_FACTORY_ENTRY(
                "__hbge2",
                CALL(MapNames::getDpctNamespace() + "compare_both_half2",
                     ARG(0), ARG(1), LITERAL("std::greater_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hbgeu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbgeu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbgeu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbgeu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbgeu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_both_half2,
            CALL_FACTORY_ENTRY("__hbgeu2",
                               CALL(MapNames::getDpctNamespace() +
                                        "unordered_compare_both_half2",
                                    ARG(0), ARG(1),
                                    LITERAL("std::greater_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hbgt2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbgt2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbgt2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hbgt2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbgt2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_both_half2,
            CALL_FACTORY_ENTRY("__hbgt2", CALL(MapNames::getDpctNamespace() +
                                                   "compare_both_half2",
                                               ARG(0), ARG(1),
                                               LITERAL("std::greater<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hbgtu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbgtu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbgtu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbgtu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbgtu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_both_half2,
            CALL_FACTORY_ENTRY("__hbgtu2",
                               CALL(MapNames::getDpctNamespace() +
                                        "unordered_compare_both_half2",
                                    ARG(0), ARG(1),
                                    LITERAL("std::greater<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hble2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hble2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hble2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hble2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hble2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_both_half2,
            CALL_FACTORY_ENTRY(
                "__hble2",
                CALL(MapNames::getDpctNamespace() + "compare_both_half2",
                     ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hbleu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbleu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbleu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbleu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbleu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_both_half2,
            CALL_FACTORY_ENTRY("__hbleu2",
                               CALL(MapNames::getDpctNamespace() +
                                        "unordered_compare_both_half2",
                                    ARG(0), ARG(1),
                                    LITERAL("std::less_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hblt2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hblt2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hblt2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hblt2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hblt2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_both_half2,
            CALL_FACTORY_ENTRY("__hblt2", CALL(MapNames::getDpctNamespace() +
                                                   "compare_both_half2",
                                               ARG(0), ARG(1),
                                               LITERAL("std::less<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hbltu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbltu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbltu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbltu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbltu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_both_half2,
            CALL_FACTORY_ENTRY(
                "__hbltu2", CALL(MapNames::getDpctNamespace() +
                                     "unordered_compare_both_half2",
                                 ARG(0), ARG(1), LITERAL("std::less<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hbne2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbne2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbne2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hbne2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbne2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_both_half2,
            CALL_FACTORY_ENTRY(
                "__hbne2",
                CALL(MapNames::getDpctNamespace() + "compare_both_half2",
                     ARG(0), ARG(1), LITERAL("std::not_equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hbneu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hbneu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hbneu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::hbneu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hbneu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_both_half2,
            CALL_FACTORY_ENTRY("__hbneu2",
                               CALL(MapNames::getDpctNamespace() +
                                        "unordered_compare_both_half2",
                                    ARG(0), ARG(1),
                                    LITERAL("std::not_equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__heq2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__heq2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__heq2", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::heq2",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__heq2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_half2,
            CALL_FACTORY_ENTRY(
                "__heq2", CALL(MapNames::getDpctNamespace() + "compare_half2",
                               ARG(0), ARG(1), LITERAL("std::equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hequ2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hequ2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hequ2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hequ2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hequ2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half2,
            CALL_FACTORY_ENTRY("__hequ2", CALL(MapNames::getDpctNamespace() +
                                                   "unordered_compare_half2",
                                               ARG(0), ARG(1),
                                               LITERAL("std::equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hge2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hge2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hge2", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hge2",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hge2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_half2,
            CALL_FACTORY_ENTRY(
                "__hge2",
                CALL(MapNames::getDpctNamespace() + "compare_half2", ARG(0),
                     ARG(1), LITERAL("std::greater_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hgeu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hgeu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hgeu2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hgeu2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hgeu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half2,
            CALL_FACTORY_ENTRY(
                "__hgeu2",
                CALL(MapNames::getDpctNamespace() + "unordered_compare_half2",
                     ARG(0), ARG(1), LITERAL("std::greater_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hgt2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hgt2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hgt2", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgt2",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hgt2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_half2,
            CALL_FACTORY_ENTRY(
                "__hgt2", CALL(MapNames::getDpctNamespace() + "compare_half2",
                               ARG(0), ARG(1), LITERAL("std::greater<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hgtu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hgtu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hgtu2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hgtu2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hgtu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half2,
            CALL_FACTORY_ENTRY("__hgtu2", CALL(MapNames::getDpctNamespace() +
                                                   "unordered_compare_half2",
                                               ARG(0), ARG(1),
                                               LITERAL("std::greater<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hisnan2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hisnan2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hisnan2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::hisnan2",
                                    ARG(0)))),
        EMPTY_FACTORY_ENTRY("__hisnan2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_isnan,
            CALL_FACTORY_ENTRY("__hisnan2",
                               CALL(MapNames::getDpctNamespace() + "isnan",
                                    ARG(0))))))

MATH_API_REWRITER_DEVICE(
    "__hle2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hle2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hle2", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hle2",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hle2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_half2,
            CALL_FACTORY_ENTRY(
                "__hle2",
                CALL(MapNames::getDpctNamespace() + "compare_half2", ARG(0),
                     ARG(1), LITERAL("std::less_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hleu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hleu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hleu2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hleu2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hleu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half2,
            CALL_FACTORY_ENTRY(
                "__hleu2",
                CALL(MapNames::getDpctNamespace() + "unordered_compare_half2",
                     ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hlt2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hlt2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hlt2", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hlt2",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hlt2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_half2,
            CALL_FACTORY_ENTRY(
                "__hlt2", CALL(MapNames::getDpctNamespace() + "compare_half2",
                               ARG(0), ARG(1), LITERAL("std::less<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hltu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hltu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hltu2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hltu2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hltu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half2,
            CALL_FACTORY_ENTRY("__hltu2", CALL(MapNames::getDpctNamespace() +
                                                   "unordered_compare_half2",
                                               ARG(0), ARG(1),
                                               LITERAL("std::less<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hne2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hne2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hne2", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hne2",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hne2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_compare_half2,
            CALL_FACTORY_ENTRY(
                "__hne2",
                CALL(MapNames::getDpctNamespace() + "compare_half2", ARG(0),
                     ARG(1), LITERAL("std::not_equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__hneu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__hneu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__hneu2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::hneu2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__hneu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_unordered_compare_half2,
            CALL_FACTORY_ENTRY(
                "__hneu2",
                CALL(MapNames::getDpctNamespace() + "unordered_compare_half2",
                     ARG(0), ARG(1), LITERAL("std::not_equal_to<>()"))))))

// Double Precision Mathematical Functions
MATH_API_REWRITER_HOST_DEVICE(
    MATH_API_REWRITER_HOST(
        "rsqrt",
        CALL_FACTORY_ENTRY("rsqrt",
                           CALL(MapNames::getClNamespace(false, true) + "rsqrt",
                                ARG(0)))),
    MATH_API_REWRITER_DEVICE(
        "rsqrt",
        MATH_API_DEVICE_NODES(
            CALL_FACTORY_ENTRY(
                "rsqrt",
                CALL(MapNames::getClNamespace(false, true) + "rsqrt", ARG(0))),
            EMPTY_FACTORY_ENTRY("rsqrt"), EMPTY_FACTORY_ENTRY("rsqrt"),
            EMPTY_FACTORY_ENTRY("rsqrt"))))

// Integer Mathematical Functions
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
            EMPTY_FACTORY_ENTRY("abs"), EMPTY_FACTORY_ENTRY("abs"),
            EMPTY_FACTORY_ENTRY("abs"))))

MATH_API_REWRITER_HOST_DEVICE(
    MATH_API_REWRITER_HOST(
        "min",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_min_max,
            CALL_FACTORY_ENTRY("min", CALL(MapNames::getDpctNamespace() + "min",
                                           ARG(0), ARG(1))))),
    MATH_API_REWRITER_DEVICE(
        "min",
        MATH_API_DEVICE_NODES(
            FEATURE_REQUEST_FACTORY(
                HelperFeatureEnum::Math_min_max,
                CALL_FACTORY_ENTRY("min",
                                   CALL(MapNames::getDpctNamespace() + "min",
                                        ARG(0), ARG(1)))),
            EMPTY_FACTORY_ENTRY("min"), EMPTY_FACTORY_ENTRY("min"),
            EMPTY_FACTORY_ENTRY("min"))))

MATH_API_REWRITER_HOST_DEVICE(
    MATH_API_REWRITER_HOST(
        "std::min",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_min_max,
            CALL_FACTORY_ENTRY("std::min",
                               CALL(MapNames::getDpctNamespace() + "min",
                                    ARG(0), ARG(1))))),
    MATH_API_REWRITER_DEVICE(
        "std::min",
        MATH_API_DEVICE_NODES(
            FEATURE_REQUEST_FACTORY(
                HelperFeatureEnum::Math_min_max,
                CALL_FACTORY_ENTRY("std::min",
                                   CALL(MapNames::getDpctNamespace() + "min",
                                        ARG(0), ARG(1)))),
            EMPTY_FACTORY_ENTRY("std::min"), EMPTY_FACTORY_ENTRY("std::min"),
            EMPTY_FACTORY_ENTRY("std::min"))))

MATH_API_REWRITER_HOST_DEVICE(
    MATH_API_REWRITER_HOST(
        "max",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_min_max,
            CALL_FACTORY_ENTRY("max", CALL(MapNames::getDpctNamespace() + "max",
                                           ARG(0), ARG(1))))),
    MATH_API_REWRITER_DEVICE(
        "max",
        MATH_API_DEVICE_NODES(
            FEATURE_REQUEST_FACTORY(
                HelperFeatureEnum::Math_min_max,
                CALL_FACTORY_ENTRY("max",
                                   CALL(MapNames::getDpctNamespace() + "max",
                                        ARG(0), ARG(1)))),
            EMPTY_FACTORY_ENTRY("max"), EMPTY_FACTORY_ENTRY("max"),
            EMPTY_FACTORY_ENTRY("max"))))

MATH_API_REWRITER_HOST_DEVICE(
    MATH_API_REWRITER_HOST(
        "std::max",
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_min_max,
            CALL_FACTORY_ENTRY("std::max",
                               CALL(MapNames::getDpctNamespace() + "max",
                                    ARG(0), ARG(1))))),
    MATH_API_REWRITER_DEVICE(
        "std::max",
        MATH_API_DEVICE_NODES(
            FEATURE_REQUEST_FACTORY(
                HelperFeatureEnum::Math_min_max,
                CALL_FACTORY_ENTRY("std::max",
                                   CALL(MapNames::getDpctNamespace() + "max",
                                        ARG(0), ARG(1)))),
            EMPTY_FACTORY_ENTRY("std::max"), EMPTY_FACTORY_ENTRY("std::max"),
            EMPTY_FACTORY_ENTRY("std::max"))))

// C++ API Routines
MATH_API_REWRITER_DEVICE(
    "saturate",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY("saturate",
                           CALL(MapNames::getClNamespace(false, true) +
                                    "clamp<float>",
                                ARG(0), ARG("0.0f"), ARG("1.0f"))),
        EMPTY_FACTORY_ENTRY("saturate"), EMPTY_FACTORY_ENTRY("saturate"),
        EMPTY_FACTORY_ENTRY("saturate")))

// Overload (Half Arithmetic Functions and Integer Intrinsics)
MATH_API_REWRITER_DEVICE_OVERLOAD(
    CheckArgType(0, "int"),
    MATH_API_REWRITER_DEVICE(
        "__hadd",
        MATH_API_DEVICE_NODES(
            CALL_FACTORY_ENTRY(
                "__hadd", CALL(MapNames::getClNamespace(false, true) + "hadd",
                               CAST_IF_NEED(LITERAL("int"), ARG(0)),
                               CAST_IF_NEED(LITERAL("int"), ARG(1)))),
            EMPTY_FACTORY_ENTRY("__hadd"), EMPTY_FACTORY_ENTRY("__hadd"),
            EMPTY_FACTORY_ENTRY("__hadd"))),
    MATH_API_REWRITER_DEVICE(
        "__hadd",
        MATH_API_DEVICE_NODES(
            EMPTY_FACTORY_ENTRY("__hadd"),
            HEADER_INSERT_FACTORY(
                HeaderType::HT_SYCL_Math,
                CALL_FACTORY_ENTRY("__hadd", CALL(MapNames::getClNamespace() +
                                                      "ext::intel::math::hadd",
                                                  ARG(0), ARG(1)))),
            EMPTY_FACTORY_ENTRY("__hadd"),
            BINARY_OP_FACTORY_ENTRY("__hadd", BinaryOperatorKind::BO_Add,
                                    ARG(0), ARG(1)))))
