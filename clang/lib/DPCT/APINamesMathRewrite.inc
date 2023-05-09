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
            HelperFeatureEnum::Math_unordered_compare,
            CALL_FACTORY_ENTRY("__hequ", CALL(MapNames::getDpctNamespace() +
                                                  "unordered_compare",
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
            HelperFeatureEnum::Math_unordered_compare,
            CALL_FACTORY_ENTRY(
                "__hgeu",
                CALL(MapNames::getDpctNamespace() + "unordered_compare", ARG(0),
                     ARG(1), LITERAL("std::greater_equal<>()"))))))

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
            HelperFeatureEnum::Math_unordered_compare,
            CALL_FACTORY_ENTRY("__hgtu", CALL(MapNames::getDpctNamespace() +
                                                  "unordered_compare",
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
            HelperFeatureEnum::Math_unordered_compare,
            CALL_FACTORY_ENTRY(
                "__hleu",
                CALL(MapNames::getDpctNamespace() + "unordered_compare", ARG(0),
                     ARG(1), LITERAL("std::less_equal<>()"))))))

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
            HelperFeatureEnum::Math_unordered_compare,
            CALL_FACTORY_ENTRY("__hltu", CALL(MapNames::getDpctNamespace() +
                                                  "unordered_compare",
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
            HelperFeatureEnum::Math_unordered_compare,
            // Notice: not equal compare need consider 'isnan'.
            CALL_FACTORY_ENTRY(
                "__hne", CALL(MapNames::getDpctNamespace() + "compare", ARG(0),
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
            HelperFeatureEnum::Math_unordered_compare,
            CALL_FACTORY_ENTRY(
                "__hneu",
                CALL(MapNames::getDpctNamespace() + "unordered_compare", ARG(0),
                     ARG(1), LITERAL("std::not_equal_to<>()"))))))

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
            HelperFeatureEnum::Math_compare_both,
            CALL_FACTORY_ENTRY("__hbeq2", CALL(MapNames::getDpctNamespace() +
                                                   "compare_both",
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
            HelperFeatureEnum::Math_unordered_compare_both,
            CALL_FACTORY_ENTRY(
                "__hbequ2",
                CALL(MapNames::getDpctNamespace() + "unordered_compare_both",
                     ARG(0), ARG(1), LITERAL("std::equal_to<>()"))))))

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
            HelperFeatureEnum::Math_compare_both,
            CALL_FACTORY_ENTRY(
                "__hbge2",
                CALL(MapNames::getDpctNamespace() + "compare_both", ARG(0),
                     ARG(1), LITERAL("std::greater_equal<>()"))))))

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
            HelperFeatureEnum::Math_unordered_compare_both,
            CALL_FACTORY_ENTRY(
                "__hbgeu2",
                CALL(MapNames::getDpctNamespace() + "unordered_compare_both",
                     ARG(0), ARG(1), LITERAL("std::greater_equal<>()"))))))

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
            HelperFeatureEnum::Math_compare_both,
            CALL_FACTORY_ENTRY(
                "__hbgt2", CALL(MapNames::getDpctNamespace() + "compare_both",
                                ARG(0), ARG(1), LITERAL("std::greater<>()"))))))

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
            HelperFeatureEnum::Math_unordered_compare_both,
            CALL_FACTORY_ENTRY("__hbgtu2", CALL(MapNames::getDpctNamespace() +
                                                    "unordered_compare_both",
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
            HelperFeatureEnum::Math_compare_both,
            CALL_FACTORY_ENTRY(
                "__hble2",
                CALL(MapNames::getDpctNamespace() + "compare_both", ARG(0),
                     ARG(1), LITERAL("std::less_equal<>()"))))))

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
            HelperFeatureEnum::Math_unordered_compare_both,
            CALL_FACTORY_ENTRY(
                "__hbleu2",
                CALL(MapNames::getDpctNamespace() + "unordered_compare_both",
                     ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))

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
            HelperFeatureEnum::Math_compare_both,
            CALL_FACTORY_ENTRY(
                "__hblt2", CALL(MapNames::getDpctNamespace() + "compare_both",
                                ARG(0), ARG(1), LITERAL("std::less<>()"))))))

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
            HelperFeatureEnum::Math_unordered_compare_both,
            CALL_FACTORY_ENTRY("__hbltu2", CALL(MapNames::getDpctNamespace() +
                                                    "unordered_compare_both",
                                                ARG(0), ARG(1),
                                                LITERAL("std::less<>()"))))))

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
            HelperFeatureEnum::Math_compare_both,
            CALL_FACTORY_ENTRY(
                "__hbne2",
                CALL(MapNames::getDpctNamespace() + "compare_both", ARG(0),
                     ARG(1), LITERAL("std::not_equal_to<>()"))))))

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
            HelperFeatureEnum::Math_unordered_compare_both,
            CALL_FACTORY_ENTRY(
                "__hbneu2",
                CALL(MapNames::getDpctNamespace() + "unordered_compare_both",
                     ARG(0), ARG(1), LITERAL("std::not_equal_to<>()"))))))

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
            HelperFeatureEnum::Math_compare2,
            CALL_FACTORY_ENTRY(
                "__heq2", CALL(MapNames::getDpctNamespace() + "compare",
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
            HelperFeatureEnum::Math_unordered_compare2,
            CALL_FACTORY_ENTRY("__hequ2", CALL(MapNames::getDpctNamespace() +
                                                   "unordered_compare",
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
            HelperFeatureEnum::Math_compare2,
            CALL_FACTORY_ENTRY("__hge2",
                               CALL(MapNames::getDpctNamespace() + "compare",
                                    ARG(0), ARG(1),
                                    LITERAL("std::greater_equal<>()"))))))

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
            HelperFeatureEnum::Math_unordered_compare2,
            CALL_FACTORY_ENTRY(
                "__hgeu2",
                CALL(MapNames::getDpctNamespace() + "unordered_compare",
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
            HelperFeatureEnum::Math_compare2,
            CALL_FACTORY_ENTRY(
                "__hgt2", CALL(MapNames::getDpctNamespace() + "compare",
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
            HelperFeatureEnum::Math_unordered_compare2,
            CALL_FACTORY_ENTRY("__hgtu2", CALL(MapNames::getDpctNamespace() +
                                                   "unordered_compare",
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
            HelperFeatureEnum::Math_compare2,
            CALL_FACTORY_ENTRY("__hle2",
                               CALL(MapNames::getDpctNamespace() + "compare",
                                    ARG(0), ARG(1),
                                    LITERAL("std::less_equal<>()"))))))

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
            HelperFeatureEnum::Math_unordered_compare2,
            CALL_FACTORY_ENTRY(
                "__hleu2",
                CALL(MapNames::getDpctNamespace() + "unordered_compare",
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
            HelperFeatureEnum::Math_compare2,
            CALL_FACTORY_ENTRY(
                "__hlt2", CALL(MapNames::getDpctNamespace() + "compare",
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
            HelperFeatureEnum::Math_unordered_compare2,
            CALL_FACTORY_ENTRY("__hltu2", CALL(MapNames::getDpctNamespace() +
                                                   "unordered_compare",
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
            HelperFeatureEnum::Math_compare2,
            CALL_FACTORY_ENTRY("__hne2",
                               CALL(MapNames::getDpctNamespace() + "compare",
                                    ARG(0), ARG(1),
                                    LITERAL("std::not_equal_to<>()"))))))

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
            HelperFeatureEnum::Math_unordered_compare2,
            CALL_FACTORY_ENTRY(
                "__hneu2",
                CALL(MapNames::getDpctNamespace() + "unordered_compare",
                     ARG(0), ARG(1), LITERAL("std::not_equal_to<>()"))))))

// Half Precision Conversion and Data Movement
MATH_API_REWRITER_DEVICE(
    "__ldca",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__ldca"), EMPTY_FACTORY_ENTRY("__ldca"),
        EMPTY_FACTORY_ENTRY("__ldca"),
        WARNING_FACTORY_ENTRY("__ldca", DEREF_FACTORY_ENTRY("__ldca", ARG(0)),
                              Diagnostics::MATH_EMULATION_EXPRESSION,
                              std::string("__ldca"), std::string("'*'"))))

MATH_API_REWRITER_DEVICE(
    "__ldcg",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__ldcg"), EMPTY_FACTORY_ENTRY("__ldcg"),
        EMPTY_FACTORY_ENTRY("__ldcg"),
        WARNING_FACTORY_ENTRY("__ldcg", DEREF_FACTORY_ENTRY("__ldcg", ARG(0)),
                              Diagnostics::MATH_EMULATION_EXPRESSION,
                              std::string("__ldcg"), std::string("'*'"))))

MATH_API_REWRITER_DEVICE(
    "__ldcs",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__ldcs"), EMPTY_FACTORY_ENTRY("__ldcs"),
        EMPTY_FACTORY_ENTRY("__ldcs"),
        WARNING_FACTORY_ENTRY("__ldcs", DEREF_FACTORY_ENTRY("__ldcs", ARG(0)),
                              Diagnostics::MATH_EMULATION_EXPRESSION,
                              std::string("__ldcs"), std::string("'*'"))))

MATH_API_REWRITER_DEVICE(
    "__ldcv",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__ldcv"), EMPTY_FACTORY_ENTRY("__ldcv"),
        EMPTY_FACTORY_ENTRY("__ldcv"),
        WARNING_FACTORY_ENTRY("__ldcv", DEREF_FACTORY_ENTRY("__ldcv", ARG(0)),
                              Diagnostics::MATH_EMULATION_EXPRESSION,
                              std::string("__ldcv"), std::string("'*'"))))

MATH_API_REWRITER_DEVICE(
    "__ldg",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__ldg"), EMPTY_FACTORY_ENTRY("__ldg"),
        EMPTY_FACTORY_ENTRY("__ldg"),
        WARNING_FACTORY_ENTRY("__ldg", DEREF_FACTORY_ENTRY("__ldg", ARG(0)),
                              Diagnostics::MATH_EMULATION_EXPRESSION,
                              std::string("__ldg"), std::string("'*'"))))

MATH_API_REWRITER_DEVICE(
    "__ldlu",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__ldlu"), EMPTY_FACTORY_ENTRY("__ldlu"),
        EMPTY_FACTORY_ENTRY("__ldlu"),
        WARNING_FACTORY_ENTRY("__ldlu", DEREF_FACTORY_ENTRY("__ldlu", ARG(0)),
                              Diagnostics::MATH_EMULATION_EXPRESSION,
                              std::string("__ldlu"), std::string("'*'"))))

MATH_API_REWRITER_DEVICE(
    "__stcg",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__stcg"), EMPTY_FACTORY_ENTRY("__stcg"),
        EMPTY_FACTORY_ENTRY("__stcg"),
        WARNING_FACTORY_ENTRY(
            "__stcg", ASSIGN_FACTORY_ENTRY("__stcg", DEREF(ARG_WC(0)), ARG(1)),
            Diagnostics::MATH_EMULATION_EXPRESSION, std::string("__stcg"),
            std::string("'='"))))

MATH_API_REWRITER_DEVICE(
    "__stcs",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__stcs"), EMPTY_FACTORY_ENTRY("__stcs"),
        EMPTY_FACTORY_ENTRY("__stcs"),
        WARNING_FACTORY_ENTRY(
            "__stcs", ASSIGN_FACTORY_ENTRY("__stcs", DEREF(ARG_WC(0)), ARG(1)),
            Diagnostics::MATH_EMULATION_EXPRESSION, std::string("__stcs"),
            std::string("'='"))))

MATH_API_REWRITER_DEVICE(
    "__stwb",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__stwb"), EMPTY_FACTORY_ENTRY("__stwb"),
        EMPTY_FACTORY_ENTRY("__stwb"),
        WARNING_FACTORY_ENTRY(
            "__stwb", ASSIGN_FACTORY_ENTRY("__stwb", DEREF(ARG_WC(0)), ARG(1)),
            Diagnostics::MATH_EMULATION_EXPRESSION, std::string("__stwb"),
            std::string("'='"))))

MATH_API_REWRITER_DEVICE(
    "__stwt",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__stwt"), EMPTY_FACTORY_ENTRY("__stwt"),
        EMPTY_FACTORY_ENTRY("__stwt"),
        WARNING_FACTORY_ENTRY(
            "__stwt", ASSIGN_FACTORY_ENTRY("__stwt", DEREF(ARG_WC(0)), ARG(1)),
            Diagnostics::MATH_EMULATION_EXPRESSION, std::string("__stwt"),
            std::string("'='"))))

// Bfloat16 Precision Conversion and Data Movement
CALL_FACTORY_ENTRY("__bfloat1622float2",
                   CALL(MapNames::getClNamespace() + "float2",
                        ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")),
                        ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))

CALL_FACTORY_ENTRY("__bfloat162float", CALL("static_cast<float>", ARG(0)))

CALL_FACTORY_ENTRY("__float22bfloat162_rn",
                   CALL(MapNames::getClNamespace() + "marray<" +
                            MapNames::getClNamespace() +
                            "ext::oneapi::bfloat16, 2>",
                        ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")),
                        ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))

CALL_FACTORY_ENTRY("__float2bfloat16",
                   CALL(MapNames::getClNamespace() + "ext::oneapi::bfloat16",
                        ARG(0)))

// Single Precision Mathematical Functions
MATH_API_REWRITER_DEVICE(
    "erfcinvf",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("erfcinvf"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY(
                "erfcinvf",
                CALL(MapNames::getClNamespace() + "ext::intel::math::erfcinv",
                     CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
        EMPTY_FACTORY_ENTRY("erfcinvf"), EMPTY_FACTORY_ENTRY("erfcinvf")))

MATH_API_REWRITER_DEVICE(
    "erfinvf",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("erfinvf"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY(
                "erfinvf",
                CALL(MapNames::getClNamespace() + "ext::intel::math::erfinv",
                     CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
        EMPTY_FACTORY_ENTRY("erfinvf"), EMPTY_FACTORY_ENTRY("erfinvf")))

MATH_API_REWRITER_DEVICE(
    "ldexpf",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY(
            "ldexpf", CALL(MapNames::getClNamespace(false, true) + "ldexp",
                           CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0)), ARG(1))),
        EMPTY_FACTORY_ENTRY("ldexpf"), EMPTY_FACTORY_ENTRY("ldexp"),
        EMPTY_FACTORY_ENTRY("ldexpf")))

MATH_API_REWRITER_DEVICE(
    "normcdff",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("normcdff"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY(
                "normcdff",
                CALL(MapNames::getClNamespace() + "ext::intel::math::cdfnorm",
                     CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0))))),
        EMPTY_FACTORY_ENTRY("normcdff"),
        BINARY_OP_FACTORY_ENTRY(
            "normcdff", BinaryOperatorKind::BO_Div,
            CALL(MapNames::getClNamespace() + "erfc",
                 BO(BinaryOperatorKind::BO_Div,
                    CAST_IF_NOT_SAME(makeLiteral("float"), ARG(0)),
                    makeLiteral("-" + MapNames::getClNamespace(false, true) +
                                "sqrt(2.0)"))),
            makeLiteral("2"))))

MATH_API_REWRITER_DEVICE(
    "normcdfinvf",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("normcdfinvf"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("normcdfinvf",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::cdfnorminv",
                                    CAST_IF_NOT_SAME(makeLiteral("float"),
                                                 ARG(0))))),
        EMPTY_FACTORY_ENTRY("normcdfinvf"), EMPTY_FACTORY_ENTRY("normcdfinvf")))

MATH_API_REWRITER_DEVICE(
    "norm3df",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY(
            "norm3df", CALL(MapNames::getClNamespace(false, true) + "length",
                            CALL(MapNames::getClNamespace() + "float3", ARG(0),
                                 ARG(1), ARG(2)))),
        EMPTY_FACTORY_ENTRY("norm3df"), EMPTY_FACTORY_ENTRY("norm3df"),
        EMPTY_FACTORY_ENTRY("norm3df")))

MATH_API_REWRITER_DEVICE(
    "norm4df",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY(
            "norm4df", CALL(MapNames::getClNamespace(false, true) + "length",
                            CALL(MapNames::getClNamespace() + "float4", ARG(0),
                                 ARG(1), ARG(2), ARG(3)))),
        EMPTY_FACTORY_ENTRY("norm4df"), EMPTY_FACTORY_ENTRY("norm4df"),
        EMPTY_FACTORY_ENTRY("norm4df")))

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
            HelperFeatureEnum::Math_length,
            WARNING_FACTORY_ENTRY(
                "normf",
                CALL_FACTORY_ENTRY("normf",
                                   CALL(MapNames::getDpctNamespace() + "length",
                                        ARG(1), ARG(0))),
                Diagnostics::MATH_EMULATION, std::string("normf"),
                MapNames::getDpctNamespace() + "length"))))

MATH_API_REWRITER_DEVICE(
    "rcbrtf",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("rcbrtf"), EMPTY_FACTORY_ENTRY("rcbrtf"),
        EMPTY_FACTORY_ENTRY("rcbrtf"),
        WARNING_FACTORY_ENTRY(
            "rcbrtf",
            CALL_FACTORY_ENTRY(
                "rcbrtf",
                CALL(MapNames::getClNamespace(false, true) + "native::recip",
                     CALL(MapNames::getClNamespace(false, true) + "cbrt<float>",
                          ARG(0)))),
            Diagnostics::MATH_EMULATION, std::string("rcbrtf"),
            MapNames::getClNamespace(false, true) + "cbrt")))

MATH_API_REWRITER_DEVICE(
    "rnorm3df",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("rnorm3df"), EMPTY_FACTORY_ENTRY("rnorm3df"),
        EMPTY_FACTORY_ENTRY("rnorm3df"),
        CALL_FACTORY_ENTRY(
            "rnorm3df",
            CALL(MapNames::getClNamespace(false, true) + "native::recip",
                 CALL(MapNames::getClNamespace(false, true) + "length",
                      CALL(MapNames::getClNamespace() + "float3", ARG(0),
                           ARG(1), ARG(2)))))))

MATH_API_REWRITER_DEVICE(
    "rnorm4df",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("rnorm4df"), EMPTY_FACTORY_ENTRY("rnorm4df"),
        EMPTY_FACTORY_ENTRY("rnorm4df"),
        CALL_FACTORY_ENTRY(
            "rnorm4df",
            CALL(MapNames::getClNamespace(false, true) + "native::recip",
                 CALL(MapNames::getClNamespace(false, true) + "length",
                      CALL(MapNames::getClNamespace() + "float4", ARG(0),
                           ARG(1), ARG(2), ARG(3)))))))

MATH_API_REWRITER_DEVICE(
    "rnormf",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("rnormf"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("rnormf", CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::rnorm",
                                              ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("rnormf"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_length,
            WARNING_FACTORY_ENTRY(
                "rnormf",
                CALL_FACTORY_ENTRY(
                    "rnormf", CALL(MapNames::getClNamespace(false, true) +
                                       "native::recip",
                                   CALL(MapNames::getDpctNamespace() + "length",
                                        ARG(1), ARG(0)))),
                Diagnostics::MATH_EMULATION, std::string("rnormf"),
                MapNames::getDpctNamespace() + "length"))))

// Double Precision Mathematical Functions
MATH_API_REWRITER_DEVICE(
    "erfcinv",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("erfcinv"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY(
                "erfcinv",
                CALL(MapNames::getClNamespace() + "ext::intel::math::erfcinv",
                     CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
        EMPTY_FACTORY_ENTRY("erfcinv"), EMPTY_FACTORY_ENTRY("erfcinv")))

MATH_API_REWRITER_DEVICE(
    "erfinv",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("erfinv"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY(
                "erfinv",
                CALL(MapNames::getClNamespace() + "ext::intel::math::erfinv",
                     CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
        EMPTY_FACTORY_ENTRY("erfinv"), EMPTY_FACTORY_ENTRY("erfinv")))

MATH_API_REWRITER_DEVICE(
    "ldexp",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY(
            "ldexp", CALL(MapNames::getClNamespace(false, true) + "ldexp",
                          CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0)), ARG(1))),
        EMPTY_FACTORY_ENTRY("ldexp"), EMPTY_FACTORY_ENTRY("ldexp"),
        EMPTY_FACTORY_ENTRY("ldexp")))

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
            HelperFeatureEnum::Math_length,
            WARNING_FACTORY_ENTRY(
                "norm",
                CALL_FACTORY_ENTRY("norm",
                                   CALL(MapNames::getDpctNamespace() + "length",
                                        ARG(1), ARG(0))),
                Diagnostics::MATH_EMULATION, std::string("norm"),
                MapNames::getDpctNamespace() + "length"))))

MATH_API_REWRITER_DEVICE(
    "norm3d",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY(
            "norm3d", CALL(MapNames::getClNamespace(false, true) + "length",
                           CALL(MapNames::getClNamespace() + "double3", ARG(0),
                                ARG(1), ARG(2)))),
        EMPTY_FACTORY_ENTRY("norm3d"), EMPTY_FACTORY_ENTRY("norm3d"),
        EMPTY_FACTORY_ENTRY("norm3d")))

MATH_API_REWRITER_DEVICE(
    "norm4d",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY(
            "norm4d", CALL(MapNames::getClNamespace(false, true) + "length",
                           CALL(MapNames::getClNamespace() + "double4", ARG(0),
                                ARG(1), ARG(2), ARG(3)))),
        EMPTY_FACTORY_ENTRY("norm4d"), EMPTY_FACTORY_ENTRY("norm4d"),
        EMPTY_FACTORY_ENTRY("norm4d")))

MATH_API_REWRITER_DEVICE(
    "normcdf",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("normcdf"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY(
                "normcdf",
                CALL(MapNames::getClNamespace() + "ext::intel::math::cdfnorm",
                     CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0))))),
        EMPTY_FACTORY_ENTRY("normcdf"),
        BINARY_OP_FACTORY_ENTRY(
            "normcdf", BinaryOperatorKind::BO_Div,
            CALL(MapNames::getClNamespace() + "erfc",
                 BO(BinaryOperatorKind::BO_Div,
                    CAST_IF_NOT_SAME(makeLiteral("double"), ARG(0)),
                    makeLiteral("-" + MapNames::getClNamespace(false, true) +
                                "sqrt(2.0)"))),
            makeLiteral("2"))))

MATH_API_REWRITER_DEVICE(
    "normcdfinv",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("normcdfinv"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("normcdfinv",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::cdfnorminv",
                                    CAST_IF_NOT_SAME(makeLiteral("double"),
                                                 ARG(0))))),
        EMPTY_FACTORY_ENTRY("normcdfinv"), EMPTY_FACTORY_ENTRY("normcdfinv")))

MATH_API_REWRITER_DEVICE(
    "rcbrt",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("rcbrt"), EMPTY_FACTORY_ENTRY("rcbrt"),
        EMPTY_FACTORY_ENTRY("rcbrt"),
        WARNING_FACTORY_ENTRY(
            "rcbrt",
            BINARY_OP_FACTORY_ENTRY(
                "rcbrt", BinaryOperatorKind::BO_Div, makeLiteral("1"),
                CALL(MapNames::getClNamespace(false, true) + "cbrt<double>",
                     ARG(0))),
            Diagnostics::MATH_EMULATION, std::string("rcbrt"),
            MapNames::getClNamespace(false, true) + "cbrt")))

MATH_API_REWRITER_DEVICE(
    "rnorm",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("rnorm"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("rnorm", CALL(MapNames::getClNamespace() +
                                                 "ext::intel::math::rnorm",
                                             ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("rnorm"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_length,
            WARNING_FACTORY_ENTRY(
                "rnorm",
                BINARY_OP_FACTORY_ENTRY(
                    "rnorm", BinaryOperatorKind::BO_Div, makeLiteral("1"),
                    CALL(MapNames::getDpctNamespace() + "length", ARG(1),
                         ARG(0))),
                Diagnostics::MATH_EMULATION, std::string("rnorm"),
                MapNames::getDpctNamespace() + "length"))))

MATH_API_REWRITER_DEVICE(
    "rnorm3d",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("rnorm3d"), EMPTY_FACTORY_ENTRY("rnorm3d"),
        EMPTY_FACTORY_ENTRY("rnorm3d"),
        BINARY_OP_FACTORY_ENTRY(
            "rnorm3d", BinaryOperatorKind::BO_Div, makeLiteral("1"),
            CALL(MapNames::getClNamespace(false, true) + "length",
                 CALL(MapNames::getClNamespace() + "double3", ARG(0), ARG(1),
                      ARG(2))))))

MATH_API_REWRITER_DEVICE(
    "rnorm4d",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("rnorm4d"), EMPTY_FACTORY_ENTRY("rnorm4d"),
        EMPTY_FACTORY_ENTRY("rnorm4d"),
        BINARY_OP_FACTORY_ENTRY(
            "rnorm4d", BinaryOperatorKind::BO_Div, makeLiteral("1"),
            CALL(MapNames::getClNamespace(false, true) + "length",
                 CALL(MapNames::getClNamespace() + "double4", ARG(0), ARG(1),
                      ARG(2), ARG(3))))))

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
        CONDITIONAL_FACTORY_ENTRY(
            math::UsingDpctMinMax,
            FEATURE_REQUEST_FACTORY(
                HelperFeatureEnum::Math_min_max,
                CALL_FACTORY_ENTRY("min",
                                   CALL(MapNames::getDpctNamespace() + "min",
                                        CAST_IF_SPECIAL(ARG(0)),
                                        CAST_IF_SPECIAL(ARG(1))))),
            CALL_FACTORY_ENTRY("min", CALL("std::min", CAST_IF_SPECIAL(ARG(0)),
                                           CAST_IF_SPECIAL(ARG(1)))))),
    MATH_API_REWRITER_DEVICE(
        "min",
        MATH_API_DEVICE_NODES(
            CONDITIONAL_FACTORY_ENTRY(
                math::UsingDpctMinMax,
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::Math_min_max,
                    CALL_FACTORY_ENTRY(
                        "min", CALL(MapNames::getDpctNamespace() + "min",
                                    CAST_IF_SPECIAL(ARG(0)),
                                    CAST_IF_SPECIAL(ARG(1))))),
                CALL_FACTORY_ENTRY("min",
                                   CALL(MapNames::getClNamespace() + "min",
                                        CAST_IF_SPECIAL(ARG(0)),
                                        CAST_IF_SPECIAL(ARG(1))))),
            EMPTY_FACTORY_ENTRY("min"), EMPTY_FACTORY_ENTRY("min"),
            EMPTY_FACTORY_ENTRY("min"))))

MATH_API_REWRITER_HOST_DEVICE(
    MATH_API_REWRITER_HOST(
        "max",
        CONDITIONAL_FACTORY_ENTRY(
            math::UsingDpctMinMax,
            FEATURE_REQUEST_FACTORY(
                HelperFeatureEnum::Math_min_max,
                CALL_FACTORY_ENTRY("max",
                                   CALL(MapNames::getDpctNamespace() + "max",
                                        CAST_IF_SPECIAL(ARG(0)),
                                        CAST_IF_SPECIAL(ARG(1))))),
            CALL_FACTORY_ENTRY("max", CALL("std::max", CAST_IF_SPECIAL(ARG(0)),
                                           CAST_IF_SPECIAL(ARG(1)))))),
    MATH_API_REWRITER_DEVICE(
        "max",
        MATH_API_DEVICE_NODES(
            CONDITIONAL_FACTORY_ENTRY(
                math::UsingDpctMinMax,
                FEATURE_REQUEST_FACTORY(
                    HelperFeatureEnum::Math_min_max,
                    CALL_FACTORY_ENTRY(
                        "max", CALL(MapNames::getDpctNamespace() + "max",
                                    CAST_IF_SPECIAL(ARG(0)),
                                    CAST_IF_SPECIAL(ARG(1))))),
                CALL_FACTORY_ENTRY("max",
                                   CALL(MapNames::getClNamespace() + "max",
                                        CAST_IF_SPECIAL(ARG(0)),
                                        CAST_IF_SPECIAL(ARG(1))))),
            EMPTY_FACTORY_ENTRY("max"), EMPTY_FACTORY_ENTRY("max"),
            EMPTY_FACTORY_ENTRY("max"))))

MATH_API_REWRITER_DEVICE(
    "llmax",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY("llmax", CALL(MapNames::getClNamespace(false, true) +
                                             "max<long long>",
                                         ARG(0), ARG(1))),
        EMPTY_FACTORY_ENTRY("llmax"), EMPTY_FACTORY_ENTRY("llmax"),
        EMPTY_FACTORY_ENTRY("llmax")))

MATH_API_REWRITER_DEVICE(
    "llmin",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY("llmin", CALL(MapNames::getClNamespace(false, true) +
                                             "min<long long>",
                                         ARG(0), ARG(1))),
        EMPTY_FACTORY_ENTRY("llmin"), EMPTY_FACTORY_ENTRY("llmin"),
        EMPTY_FACTORY_ENTRY("llmin")))

MATH_API_REWRITER_DEVICE(
    "ullmax",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY("ullmax",
                           CALL(MapNames::getClNamespace(false, true) +
                                    "max<unsigned long long>",
                                ARG(0), ARG(1))),
        EMPTY_FACTORY_ENTRY("ullmax"), EMPTY_FACTORY_ENTRY("ullmax"),
        EMPTY_FACTORY_ENTRY("ullmax")))

MATH_API_REWRITER_DEVICE(
    "ullmin",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY("ullmin",
                           CALL(MapNames::getClNamespace(false, true) +
                                    "min<unsigned long long>",
                                ARG(0), ARG(1))),
        EMPTY_FACTORY_ENTRY("ullmin"), EMPTY_FACTORY_ENTRY("ullmin"),
        EMPTY_FACTORY_ENTRY("ullmin")))

MATH_API_REWRITER_DEVICE(
    "umax",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY("umax", CALL(MapNames::getClNamespace(false, true) +
                                            "max<unsigned int>",
                                        ARG(0), ARG(1))),
        EMPTY_FACTORY_ENTRY("umax"), EMPTY_FACTORY_ENTRY("umax"),
        EMPTY_FACTORY_ENTRY("umax")))

MATH_API_REWRITER_DEVICE(
    "umin",
    MATH_API_DEVICE_NODES(
        CALL_FACTORY_ENTRY("umin", CALL(MapNames::getClNamespace(false, true) +
                                            "min<unsigned int>",
                                        ARG(0), ARG(1))),
        EMPTY_FACTORY_ENTRY("umin"), EMPTY_FACTORY_ENTRY("umin"),
        EMPTY_FACTORY_ENTRY("umin")))

// SIMD Intrinsics
MATH_API_REWRITER_DEVICE(
    "__vabs2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vabs2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vabs2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::vabs2",
                                               ARG(0)))),
        EMPTY_FACTORY_ENTRY("__vabs2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_unary,
            CALL_FACTORY_ENTRY(
                "__vabs2",
                CALL(MapNames::getDpctNamespace() + "vectorized_unary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0),
                     LITERAL(MapNames::getDpctNamespace() + "abs()"))))))

MATH_API_REWRITER_DEVICE(
    "__vabs4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vabs4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vabs4", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::vabs4",
                                               ARG(0)))),
        EMPTY_FACTORY_ENTRY("__vabs4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_unary,
            CALL_FACTORY_ENTRY(
                "__vabs4",
                CALL(MapNames::getDpctNamespace() + "vectorized_unary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0),
                     LITERAL(MapNames::getDpctNamespace() + "abs()"))))))

MATH_API_REWRITER_DEVICE(
    "__vabsdiffs2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vabsdiffs2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vabsdiffs2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vabsdiffs2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vabsdiffs2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vabsdiffs2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "abs_diff()"))))))

MATH_API_REWRITER_DEVICE(
    "__vabsdiffs4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vabsdiffs4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vabsdiffs4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vabsdiffs4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vabsdiffs4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vabsdiffs4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "abs_diff()"))))))

MATH_API_REWRITER_DEVICE(
    "__vabsdiffu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vabsdiffu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vabsdiffu2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vabsdiffu2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vabsdiffu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vabsdiffu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "abs_diff()"))))))

MATH_API_REWRITER_DEVICE(
    "__vabsdiffu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vabsdiffu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vabsdiffu4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vabsdiffu4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vabsdiffu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vabsdiffu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "abs_diff()"))))))

MATH_API_REWRITER_DEVICE(
    "__vabsss2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vabsss2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vabsss2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vabsss2",
                                    ARG(0)))),
        EMPTY_FACTORY_ENTRY("__vabsss2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vabsss2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), LITERAL("0"),
                     LITERAL(MapNames::getDpctNamespace() + "abs_diff()"))))))

MATH_API_REWRITER_DEVICE(
    "__vabsss4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vabsss4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vabsss4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vabsss4",
                                    ARG(0)))),
        EMPTY_FACTORY_ENTRY("__vabsss4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vabsss4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), LITERAL("0"),
                     LITERAL(MapNames::getDpctNamespace() + "abs_diff()"))))))

MATH_API_REWRITER_DEVICE(
    "__vadd2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vadd2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vadd2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::vadd2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vadd2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vadd2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1), LITERAL("std::plus<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vadd4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vadd4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vadd4", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::vadd4",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vadd4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vadd4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1), LITERAL("std::plus<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vaddss2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vaddss2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vaddss2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vaddss2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vaddss2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vaddss2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "add_sat()"))))))

MATH_API_REWRITER_DEVICE(
    "__vaddss4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vaddss4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vaddss4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vaddss4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vaddss4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vaddss4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "add_sat()"))))))

MATH_API_REWRITER_DEVICE(
    "__vaddus2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vaddus2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vaddus2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vaddus2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vaddus2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vaddus2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "add_sat()"))))))

MATH_API_REWRITER_DEVICE(
    "__vaddus4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vaddus4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vaddus4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vaddus4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vaddus4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vaddus4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "add_sat()"))))))

MATH_API_REWRITER_DEVICE(
    "__vavgs2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vavgs2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vavgs2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vavgs2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vavgs2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vavgs2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "rhadd()"))))))

MATH_API_REWRITER_DEVICE(
    "__vavgs4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vavgs4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vavgs4", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vavgs4",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vavgs4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vavgs4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "rhadd()"))))))

MATH_API_REWRITER_DEVICE(
    "__vavgu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vavgu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vavgu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vavgu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vavgu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vavgu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "rhadd()"))))))

MATH_API_REWRITER_DEVICE(
    "__vavgu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vavgu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vavgu4", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vavgu4",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vavgu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vavgu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "rhadd()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpeq2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpeq2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpeq2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpeq2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpeq2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpeq2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1), LITERAL("std::equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpeq4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpeq4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpeq4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpeq4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpeq4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpeq4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1), LITERAL("std::equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpges2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpges2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpges2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpges2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpges2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpges2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1), LITERAL("std::greater_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpges4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpges4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpges4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpges4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpges4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpges4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1), LITERAL("std::greater_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpgeu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpgeu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpgeu2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpgeu2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpgeu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpgeu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1), LITERAL("std::greater_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpgeu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpgeu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpgeu4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpgeu4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpgeu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpgeu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1), LITERAL("std::greater_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpgts2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpgts2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpgts2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpgts2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpgts2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpgts2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1), LITERAL("std::greater<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpgts4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpgts4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpgts4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpgts4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpgts4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpgts4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1), LITERAL("std::greater<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpgtu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpgtu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpgtu2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpgtu2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpgtu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpgtu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1), LITERAL("std::greater<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpgtu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpgtu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpgtu4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpgtu4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpgtu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpgtu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1), LITERAL("std::greater<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmples2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmples2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmples2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmples2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmples2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmples2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmples4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmples4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmples4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmples4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmples4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmples4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpleu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpleu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpleu2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpleu2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpleu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpleu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpleu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpleu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpleu4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpleu4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpleu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpleu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmplts2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmplts2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmplts2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmplts2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmplts2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmplts2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1), LITERAL("std::less<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmplts4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmplts4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmplts4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmplts4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmplts4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmplts4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1), LITERAL("std::less<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpltu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpltu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpltu2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpltu2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpltu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpltu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1), LITERAL("std::less<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpltu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpltu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpltu4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpltu4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpltu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpltu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1), LITERAL("std::less<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpne2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpne2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpne2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpne2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpne2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpne2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1), LITERAL("std::not_equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vcmpne4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vcmpne4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vcmpne4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vcmpne4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vcmpne4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vcmpne4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1), LITERAL("std::not_equal_to<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vhaddu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vhaddu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vhaddu2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vhaddu2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vhaddu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vhaddu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "hadd()"))))))

MATH_API_REWRITER_DEVICE(
    "__vhaddu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vhaddu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vhaddu4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vhaddu4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vhaddu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vhaddu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "hadd()"))))))

MATH_API_REWRITER_DEVICE(
    "__vmaxs2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vmaxs2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vmaxs2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vmaxs2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vmaxs2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vmaxs2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "maximum()"))))))

MATH_API_REWRITER_DEVICE(
    "__vmaxs4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vmaxs4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vmaxs4", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vmaxs4",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vmaxs4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vmaxs4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "maximum()"))))))

MATH_API_REWRITER_DEVICE(
    "__vmaxu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vmaxu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vmaxu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vmaxu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vmaxu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vmaxu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "maximum()"))))))

MATH_API_REWRITER_DEVICE(
    "__vmaxu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vmaxu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vmaxu4", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vmaxu4",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vmaxu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vmaxu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "maximum()"))))))

MATH_API_REWRITER_DEVICE(
    "__vmins2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vmins2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vmins2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vmins2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vmins2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vmins2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "minimum()"))))))

MATH_API_REWRITER_DEVICE(
    "__vmins4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vmins4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vmins4", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vmins4",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vmins4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vmins4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "minimum()"))))))

MATH_API_REWRITER_DEVICE(
    "__vminu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vminu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vminu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vminu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vminu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vminu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "minimum()"))))))

MATH_API_REWRITER_DEVICE(
    "__vminu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vminu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vminu4", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vminu4",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vminu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vminu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "minimum()"))))))

MATH_API_REWRITER_DEVICE(
    "__vneg2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vneg2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vneg2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::vneg2",
                                               ARG(0)))),
        EMPTY_FACTORY_ENTRY("__vneg2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_unary,
            CALL_FACTORY_ENTRY("__vneg2",
                               CALL(MapNames::getDpctNamespace() +
                                        "vectorized_unary<" +
                                        MapNames::getClNamespace() + "short2>",
                                    ARG(0), LITERAL("std::negate<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vneg4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vneg4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vneg4", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::vneg4",
                                               ARG(0)))),
        EMPTY_FACTORY_ENTRY("__vneg4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_unary,
            CALL_FACTORY_ENTRY("__vneg4",
                               CALL(MapNames::getDpctNamespace() +
                                        "vectorized_unary<" +
                                        MapNames::getClNamespace() + "char4>",
                                    ARG(0), LITERAL("std::negate<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vnegss2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vnegss2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vnegss2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vnegss2",
                                    ARG(0)))),
        EMPTY_FACTORY_ENTRY("__vnegss2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vnegss2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     LITERAL("0"), ARG(0),
                     LITERAL(MapNames::getDpctNamespace() + "sub_sat()"))))))

MATH_API_REWRITER_DEVICE(
    "__vnegss4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vnegss4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vnegss4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vnegss4",
                                    ARG(0)))),
        EMPTY_FACTORY_ENTRY("__vnegss4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vnegss4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     LITERAL("0"), ARG(0),
                     LITERAL(MapNames::getDpctNamespace() + "sub_sat()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsads2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsads2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsads2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vsads2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsads2"),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Math_vectorized_sum_abs_diff,
                                ENTRY_RENAMED("__vsads2",
                                              MapNames::getDpctNamespace() +
                                                  "vectorized_sum_abs_diff<" +
                                                  MapNames::getClNamespace() +
                                                  "short2>"))))

MATH_API_REWRITER_DEVICE(
    "__vsads4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsads4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsads4", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vsads4",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsads4"),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Math_vectorized_sum_abs_diff,
                                ENTRY_RENAMED("__vsads4",
                                              MapNames::getDpctNamespace() +
                                                  "vectorized_sum_abs_diff<" +
                                                  MapNames::getClNamespace() +
                                                  "char4>"))))

MATH_API_REWRITER_DEVICE(
    "__vsadu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsadu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsadu2", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vsadu2",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsadu2"),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Math_vectorized_sum_abs_diff,
                                ENTRY_RENAMED("__vsadu2",
                                              MapNames::getDpctNamespace() +
                                                  "vectorized_sum_abs_diff<" +
                                                  MapNames::getClNamespace() +
                                                  "ushort2>"))))

MATH_API_REWRITER_DEVICE(
    "__vsadu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsadu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsadu4", CALL(MapNames::getClNamespace() +
                                                    "ext::intel::math::vsadu4",
                                                ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsadu4"),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Math_vectorized_sum_abs_diff,
                                ENTRY_RENAMED("__vsadu4",
                                              MapNames::getDpctNamespace() +
                                                  "vectorized_sum_abs_diff<" +
                                                  MapNames::getClNamespace() +
                                                  "uchar4>"))))

MATH_API_REWRITER_DEVICE(
    "__vseteq2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vseteq2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vseteq2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vseteq2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vseteq2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vseteq2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL("std::equal_to<unsigned short>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vseteq4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vseteq4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vseteq4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vseteq4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vseteq4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vseteq4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL("std::equal_to<unsigned char>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetges2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetges2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetges2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetges2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetges2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetges2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1), LITERAL("std::greater_equal<short>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetges4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetges4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetges4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetges4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetges4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetges4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1), LITERAL("std::greater_equal<char>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetgeu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetgeu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetgeu2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetgeu2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetgeu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetgeu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL("std::greater_equal<unsigned short>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetgeu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetgeu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetgeu4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetgeu4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetgeu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetgeu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL("std::greater_equal<unsigned char>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetgts2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetgts2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetgts2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetgts2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetgts2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetgts2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1), LITERAL("std::greater<short>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetgts4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetgts4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetgts4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetgts4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetgts4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetgts4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1), LITERAL("std::greater<char>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetgtu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetgtu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetgtu2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetgtu2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetgtu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetgtu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL("std::greater<unsigned short>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetgtu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetgtu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetgtu4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetgtu4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetgtu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetgtu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL("std::greater<unsigned char>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetles2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetles2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetles2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetles2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetles2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetles2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1), LITERAL("std::less_equal<short>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetles4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetles4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetles4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetles4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetles4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetles4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1), LITERAL("std::less_equal<char>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetleu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetleu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetleu2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetleu2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetleu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetleu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL("std::less_equal<unsigned short>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetleu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetleu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetleu4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetleu4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetleu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetleu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL("std::less_equal<unsigned char>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetlts2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetlts2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetlts2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetlts2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetlts2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetlts2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1), LITERAL("std::less<short>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetlts4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetlts4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetlts4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetlts4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetlts4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetlts4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1), LITERAL("std::less<char>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetltu2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetltu2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetltu2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetltu2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetltu2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetltu2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1), LITERAL("std::less<unsigned short>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetltu4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetltu4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetltu4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetltu4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetltu4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetltu4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1), LITERAL("std::less<unsigned char>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetne2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetne2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetne2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetne2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetne2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetne2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL("std::not_equal_to<unsigned short>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsetne4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsetne4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsetne4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsetne4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsetne4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsetne4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL("std::not_equal_to<unsigned char>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsub2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsub2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsub2", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::vsub2",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsub2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsub2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1), LITERAL("std::minus<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsub4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsub4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsub4", CALL(MapNames::getClNamespace() +
                                                   "ext::intel::math::vsub4",
                                               ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsub4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsub4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1), LITERAL("std::minus<>()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsubss2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsubss2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsubss2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsubss2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsubss2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsubss2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "short2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "sub_sat()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsubss4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsubss4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsubss4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsubss4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsubss4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsubss4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "char4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "sub_sat()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsubus2",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsubus2"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsubus2",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsubus2",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsubus2"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsubus2",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "ushort2>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "sub_sat()"))))))

MATH_API_REWRITER_DEVICE(
    "__vsubus4",
    MATH_API_DEVICE_NODES(
        EMPTY_FACTORY_ENTRY("__vsubus4"),
        HEADER_INSERT_FACTORY(
            HeaderType::HT_SYCL_Math,
            CALL_FACTORY_ENTRY("__vsubus4",
                               CALL(MapNames::getClNamespace() +
                                        "ext::intel::math::vsubus4",
                                    ARG(0), ARG(1)))),
        EMPTY_FACTORY_ENTRY("__vsubus4"),
        FEATURE_REQUEST_FACTORY(
            HelperFeatureEnum::Math_vectorized_binary,
            CALL_FACTORY_ENTRY(
                "__vsubus4",
                CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                         MapNames::getClNamespace() + "uchar4>",
                     ARG(0), ARG(1),
                     LITERAL(MapNames::getDpctNamespace() + "sub_sat()"))))))

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
                               CAST_IF_NOT_SAME(LITERAL("int"), ARG(0)),
                               CAST_IF_NOT_SAME(LITERAL("int"), ARG(1)))),
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
