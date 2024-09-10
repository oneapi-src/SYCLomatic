//===--------------- RewriterHalf2ComparisonFunctions.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createHalf2ComparisonFunctionsRewriterMap() {
  return RewriterMap{
      // __hbeq2
      MATH_API_REWRITER_DEVICE(
          "__hbeq2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hbeq2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hbeq2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hbeq2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hbeq2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hbeq2",
                      CALL(MapNames::getDpctNamespace() + "compare_both",
                           ARG(0), ARG(1), LITERAL("std::equal_to<>()"))))))
      // __hbequ2
      MATH_API_REWRITER_DEVICE(
          "__hbequ2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hbequ2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hbequ2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hbequ2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hbequ2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY("__hbequ2",
                                     CALL(MapNames::getDpctNamespace() +
                                              "unordered_compare_both",
                                          ARG(0), ARG(1),
                                          LITERAL("std::equal_to<>()"))))))
      // __hbge2
      MATH_API_REWRITER_DEVICE(
          "__hbge2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hbge2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hbge2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hbge2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hbge2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hbge2",
                      CALL(MapNames::getDpctNamespace() + "compare_both",
                           ARG(0), ARG(1),
                           LITERAL("std::greater_equal<>()"))))))
      // __hbgeu2
      MATH_API_REWRITER_DEVICE(
          "__hbgeu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hbgeu2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hbgeu2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hbgeu2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hbgeu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY("__hbgeu2",
                                     CALL(MapNames::getDpctNamespace() +
                                              "unordered_compare_both",
                                          ARG(0), ARG(1),
                                          LITERAL("std::greater_equal<>()"))))))
      // __hbgt2
      MATH_API_REWRITER_DEVICE(
          "__hbgt2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hbgt2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hbgt2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hbgt2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hbgt2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hbgt2",
                      CALL(MapNames::getDpctNamespace() + "compare_both",
                           ARG(0), ARG(1), LITERAL("std::greater<>()"))))))
      // __hbgtu2
      MATH_API_REWRITER_DEVICE(
          "__hbgtu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hbgtu2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hbgtu2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hbgtu2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hbgtu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY("__hbgtu2",
                                     CALL(MapNames::getDpctNamespace() +
                                              "unordered_compare_both",
                                          ARG(0), ARG(1),
                                          LITERAL("std::greater<>()"))))))
      // __hble2
      MATH_API_REWRITER_DEVICE(
          "__hble2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hble2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hble2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hble2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hble2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hble2",
                      CALL(MapNames::getDpctNamespace() + "compare_both",
                           ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))
      // __hbleu2
      MATH_API_REWRITER_DEVICE(
          "__hbleu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hbleu2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hbleu2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hbleu2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hbleu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY("__hbleu2",
                                     CALL(MapNames::getDpctNamespace() +
                                              "unordered_compare_both",
                                          ARG(0), ARG(1),
                                          LITERAL("std::less_equal<>()"))))))
      // __hblt2
      MATH_API_REWRITER_DEVICE(
          "__hblt2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hblt2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hblt2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hblt2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hblt2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hblt2",
                      CALL(MapNames::getDpctNamespace() + "compare_both",
                           ARG(0), ARG(1), LITERAL("std::less<>()"))))))
      // __hbltu2
      MATH_API_REWRITER_DEVICE(
          "__hbltu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hbltu2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hbltu2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hbltu2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hbltu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY("__hbltu2",
                                     CALL(MapNames::getDpctNamespace() +
                                              "unordered_compare_both",
                                          ARG(0), ARG(1),
                                          LITERAL("std::less<>()"))))))
      // __hbne2
      MATH_API_REWRITER_DEVICE(
          "__hbne2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hbne2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hbne2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hbne2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hbne2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hbne2",
                      CALL(MapNames::getDpctNamespace() + "compare_both",
                           ARG(0), ARG(1), LITERAL("std::not_equal_to<>()"))))))
      // __hbneu2
      MATH_API_REWRITER_DEVICE(
          "__hbneu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hbneu2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hbneu2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hbneu2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hbneu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY("__hbneu2",
                                     CALL(MapNames::getDpctNamespace() +
                                              "unordered_compare_both",
                                          ARG(0), ARG(1),
                                          LITERAL("std::not_equal_to<>()"))))))
      // __heq2
      MATH_API_REWRITER_DEVICE(
          "__heq2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__heq2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__heq2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::heq2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__heq2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__heq2",
                      CALL(MapNames::getDpctNamespace() + "compare", ARG(0),
                           ARG(1), LITERAL("std::equal_to<>()"))))))
      // __heq2_mask
      MATH_API_REWRITER_DEVICE(
          "__heq2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__heq2_mask"),
              EMPTY_FACTORY_ENTRY("__heq2_mask"),
              EMPTY_FACTORY_ENTRY("__heq2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__heq2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__heq2_mask")),
                  CALL_FACTORY_ENTRY(
                      "__heq2_mask",
                      CALL(MapNames::getDpctNamespace() + "compare_mask",
                           ARG(0), ARG(1), LITERAL("std::equal_to<>()"))))))
      // __hequ2
      MATH_API_REWRITER_DEVICE(
          "__hequ2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hequ2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hequ2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hequ2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hequ2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hequ2",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1), LITERAL("std::equal_to<>()"))))))
      // __hequ2_mask
      MATH_API_REWRITER_DEVICE(
          "__hequ2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hequ2_mask"),
              EMPTY_FACTORY_ENTRY("__hequ2_mask"),
              EMPTY_FACTORY_ENTRY("__hequ2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hequ2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hequ2_mask")),
                  CALL_FACTORY_ENTRY("__hequ2_mask",
                                     CALL(MapNames::getDpctNamespace() +
                                              "unordered_compare_mask",
                                          ARG(0), ARG(1),
                                          LITERAL("std::equal_to<>()"))))))
      // __hge2
      MATH_API_REWRITER_DEVICE(
          "__hge2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hge2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hge2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hge2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hge2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hge2",
                      CALL(MapNames::getDpctNamespace() + "compare", ARG(0),
                           ARG(1), LITERAL("std::greater_equal<>()"))))))
      // __hge2_mask
      MATH_API_REWRITER_DEVICE(
          "__hge2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hge2_mask"),
              EMPTY_FACTORY_ENTRY("__hge2_mask"),
              EMPTY_FACTORY_ENTRY("__hge2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hge2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hge2_mask")),
                  CALL_FACTORY_ENTRY(
                      "__hge2_mask",
                      CALL(MapNames::getDpctNamespace() + "compare_mask",
                           ARG(0), ARG(1),
                           LITERAL("std::greater_equal<>()"))))))
      // __hgeu2
      MATH_API_REWRITER_DEVICE(
          "__hgeu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hgeu2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hgeu2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgeu2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hgeu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hgeu2",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1),
                           LITERAL("std::greater_equal<>()"))))))
      // __hgeu2_mask
      MATH_API_REWRITER_DEVICE(
          "__hgeu2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hgeu2_mask"),
              EMPTY_FACTORY_ENTRY("__hgeu2_mask"),
              EMPTY_FACTORY_ENTRY("__hgeu2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hgeu2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hgeu2_mask")),
                  CALL_FACTORY_ENTRY("__hgeu2_mask",
                                     CALL(MapNames::getDpctNamespace() +
                                              "unordered_compare_mask",
                                          ARG(0), ARG(1),
                                          LITERAL("std::greater_equal<>()"))))))
      // __hgt2
      MATH_API_REWRITER_DEVICE(
          "__hgt2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hgt2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hgt2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgt2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hgt2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hgt2",
                      CALL(MapNames::getDpctNamespace() + "compare", ARG(0),
                           ARG(1), LITERAL("std::greater<>()"))))))
      // __hgt2_mask
      MATH_API_REWRITER_DEVICE(
          "__hgt2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hgt2_mask"),
              EMPTY_FACTORY_ENTRY("__hgt2_mask"),
              EMPTY_FACTORY_ENTRY("__hgt2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hgt2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hgt2_mask")),
                  CALL_FACTORY_ENTRY(
                      "__hgt2_mask",
                      CALL(MapNames::getDpctNamespace() + "compare_mask",
                           ARG(0), ARG(1), LITERAL("std::greater<>()"))))))
      // __hgtu2
      MATH_API_REWRITER_DEVICE(
          "__hgtu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hgtu2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hgtu2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hgtu2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hgtu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hgtu2",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1), LITERAL("std::greater<>()"))))))
      // __hgtu2_mask
      MATH_API_REWRITER_DEVICE(
          "__hgtu2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hgtu2_mask"),
              EMPTY_FACTORY_ENTRY("__hgtu2_mask"),
              EMPTY_FACTORY_ENTRY("__hgtu2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hgtu2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hgtu2_mask")),
                  CALL_FACTORY_ENTRY("__hgtu2_mask",
                                     CALL(MapNames::getDpctNamespace() +
                                              "unordered_compare_mask",
                                          ARG(0), ARG(1),
                                          LITERAL("std::greater<>()"))))))
      // __hisnan2
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
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
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__hisnan2",
                          CALL(MapNames::getDpctNamespace() + "isnan",
                               ARG(0)))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hisnan2",
              CALL_FACTORY_ENTRY("__hisnan2",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::isnan",
                                      ARG(0))),
              CALL_FACTORY_ENTRY(
                  "__hisnan2",
                  CALL(MapNames::getClNamespace() + "marray<" +
                           MapNames::getClNamespace() +
                           "ext::oneapi::bfloat16, 2>",
                       CALL(MapNames::getClNamespace() + "isnan",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
                       CALL(MapNames::getClNamespace() + "isnan",
                            CALL("float",
                                 ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))))))
      // __hle2
      MATH_API_REWRITER_DEVICE(
          "__hle2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hle2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hle2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hle2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hle2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hle2",
                      CALL(MapNames::getDpctNamespace() + "compare", ARG(0),
                           ARG(1), LITERAL("std::less_equal<>()"))))))
      // __hle2_mask
      MATH_API_REWRITER_DEVICE(
          "__hle2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hle2_mask"),
              EMPTY_FACTORY_ENTRY("__hle2_mask"),
              EMPTY_FACTORY_ENTRY("__hle2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hle2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hle2_mask")),
                  CALL_FACTORY_ENTRY(
                      "__hle2_mask",
                      CALL(MapNames::getDpctNamespace() + "compare_mask",
                           ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))
      // __hleu2
      MATH_API_REWRITER_DEVICE(
          "__hleu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hleu2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hleu2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hleu2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hleu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hleu2",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1), LITERAL("std::less_equal<>()"))))))
      // __hleu2_mask
      MATH_API_REWRITER_DEVICE(
          "__hleu2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hleu2_mask"),
              EMPTY_FACTORY_ENTRY("__hleu2_mask"),
              EMPTY_FACTORY_ENTRY("__hleu2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hleu2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hleu2_mask")),
                  CALL_FACTORY_ENTRY("__hleu2_mask",
                                     CALL(MapNames::getDpctNamespace() +
                                              "unordered_compare_mask",
                                          ARG(0), ARG(1),
                                          LITERAL("std::less_equal<>()"))))))
      // __hlt2
      MATH_API_REWRITER_DEVICE(
          "__hlt2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hlt2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hlt2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hlt2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hlt2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hlt2",
                      CALL(MapNames::getDpctNamespace() + "compare", ARG(0),
                           ARG(1), LITERAL("std::less<>()"))))))
      // __hlt2_mask
      MATH_API_REWRITER_DEVICE(
          "__hlt2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hlt2_mask"),
              EMPTY_FACTORY_ENTRY("__hlt2_mask"),
              EMPTY_FACTORY_ENTRY("__hlt2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hlt2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hlt2_mask")),
                  CALL_FACTORY_ENTRY(
                      "__hlt2_mask",
                      CALL(MapNames::getDpctNamespace() + "compare_mask",
                           ARG(0), ARG(1), LITERAL("std::less<>()"))))))
      // __hltu2
      MATH_API_REWRITER_DEVICE(
          "__hltu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hltu2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hltu2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hltu2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hltu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hltu2",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1), LITERAL("std::less<>()"))))))
      // __hltu2_mask
      MATH_API_REWRITER_DEVICE(
          "__hltu2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hltu2_mask"),
              EMPTY_FACTORY_ENTRY("__hltu2_mask"),
              EMPTY_FACTORY_ENTRY("__hltu2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hltu2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hltu2_mask")),
                  CALL_FACTORY_ENTRY("__hltu2_mask",
                                     CALL(MapNames::getDpctNamespace() +
                                              "unordered_compare_mask",
                                          ARG(0), ARG(1),
                                          LITERAL("std::less<>()"))))))
      // __hmax2
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "__hmax2",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hmax2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmax2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmax2",
                                              ARG(0), ARG(1)))),
                  EMPTY_FACTORY_ENTRY("__hmax2"),
                  CALL_FACTORY_ENTRY(
                      "__hmax2",
                      CALL(MapNames::getClNamespace() + "half2",
                           CALL(MapNames::getClNamespace() + "fmax",
                                ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")),
                                ARRAY_SUBSCRIPT(ARG(1), LITERAL("0"))),
                           CALL(MapNames::getClNamespace() + "fmax",
                                ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")),
                                ARRAY_SUBSCRIPT(ARG(1), LITERAL("1"))))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hmax2",
              CALL_FACTORY_ENTRY("__hmax2",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::fmax",
                                      ARG(0), ARG(1))),
              CALL_FACTORY_ENTRY(
                  "__hmax2",
                  CALL(
                      MapNames::getClNamespace() + "marray<" +
                          MapNames::getClNamespace() +
                          "ext::oneapi::bfloat16, 2>",
                      CALL(
                          MapNames::getClNamespace() + "fmax",
                          CALL("float", ARRAY_SUBSCRIPT(ARG(0), LITERAL("0"))),
                          CALL("float", ARRAY_SUBSCRIPT(ARG(1), LITERAL("0")))),
                      CALL(MapNames::getClNamespace() + "fmax",
                           CALL("float", ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))),
                           CALL("float",
                                ARRAY_SUBSCRIPT(ARG(1), LITERAL("1"))))))))
      // __hmax2_nan
      MATH_API_REWRITER_DEVICE(
          "__hmax2_nan",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hmax2_nan"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmax2_nan",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmax2_nan",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hmax2_nan"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hmax2_nan",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hmax2_nan")),
                  CALL_FACTORY_ENTRY(
                      "__hmax2_nan",
                      CALL(MapNames::getDpctNamespace() + "fmax_nan", ARG(0),
                           ARG(1))))))
      // __hmin2
      MATH_API_REWRITER_DEVICE_OVERLOAD(
          CheckArgType(0, "__half2"),
          MATH_API_REWRITER_DEVICE(
              "__hmin2",
              MATH_API_DEVICE_NODES(
                  EMPTY_FACTORY_ENTRY("__hmin2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmin2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmin2",
                                              ARG(0), ARG(1)))),
                  EMPTY_FACTORY_ENTRY("__hmin2"),
                  CALL_FACTORY_ENTRY(
                      "__hmin2",
                      CALL(MapNames::getClNamespace() + "half2",
                           CALL(MapNames::getClNamespace() + "fmin",
                                ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")),
                                ARRAY_SUBSCRIPT(ARG(1), LITERAL("0"))),
                           CALL(MapNames::getClNamespace() + "fmin",
                                ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")),
                                ARRAY_SUBSCRIPT(ARG(1), LITERAL("1"))))))),
          MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(
              "__hmin2",
              CALL_FACTORY_ENTRY("__hmin2",
                                 CALL(MapNames::getClNamespace(false, true) +
                                          "ext::oneapi::experimental::fmin",
                                      ARG(0), ARG(1))),
              CALL_FACTORY_ENTRY(
                  "__hmin2",
                  CALL(
                      MapNames::getClNamespace() + "marray<" +
                          MapNames::getClNamespace() +
                          "ext::oneapi::bfloat16, 2>",
                      CALL(
                          MapNames::getClNamespace() + "fmin",
                          CALL("float", ARRAY_SUBSCRIPT(ARG(0), LITERAL("0"))),
                          CALL("float", ARRAY_SUBSCRIPT(ARG(1), LITERAL("0")))),
                      CALL(MapNames::getClNamespace() + "fmin",
                           CALL("float", ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))),
                           CALL("float",
                                ARRAY_SUBSCRIPT(ARG(1), LITERAL("1"))))))))
      // __hmin2_nan
      MATH_API_REWRITER_DEVICE(
          "__hmin2_nan",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hmin2_nan"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hmin2_nan",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hmin2_nan",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hmin2_nan"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hmin2_nan",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hmin2_nan")),
                  CALL_FACTORY_ENTRY(
                      "__hmin2_nan",
                      CALL(MapNames::getDpctNamespace() + "fmin_nan", ARG(0),
                           ARG(1))))))
      // __hne2
      MATH_API_REWRITER_DEVICE(
          "__hne2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hne2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hne2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hne2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hne2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hne2",
                      CALL(MapNames::getDpctNamespace() + "compare", ARG(0),
                           ARG(1), LITERAL("std::not_equal_to<>()"))))))
      // __hne2_mask
      MATH_API_REWRITER_DEVICE(
          "__hne2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hne2_mask"),
              EMPTY_FACTORY_ENTRY("__hne2_mask"),
              EMPTY_FACTORY_ENTRY("__hne2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hne2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hne2_mask")),
                  CALL_FACTORY_ENTRY(
                      "__hne2_mask",
                      CALL(MapNames::getDpctNamespace() + "compare_mask",
                           ARG(0), ARG(1), LITERAL("std::not_equal_to<>()"))))))
      // __hneu2
      MATH_API_REWRITER_DEVICE(
          "__hneu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hneu2"),
              MATH_API_SPECIFIC_ELSE_EMU(
                  CheckArgType(0, "__half2"),
                  HEADER_INSERT_FACTORY(
                      HeaderType::HT_SYCL_Math,
                      CALL_FACTORY_ENTRY("__hneu2",
                                         CALL(MapNames::getClNamespace() +
                                                  "ext::intel::math::hneu2",
                                              ARG(0), ARG(1))))),
              EMPTY_FACTORY_ENTRY("__hneu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__hneu2",
                      CALL(MapNames::getDpctNamespace() + "unordered_compare",
                           ARG(0), ARG(1), LITERAL("std::not_equal_to<>()"))))))
      // __hneu2_mask
      MATH_API_REWRITER_DEVICE(
          "__hneu2_mask",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__hneu2_mask"),
              EMPTY_FACTORY_ENTRY("__hneu2_mask"),
              EMPTY_FACTORY_ENTRY("__hneu2_mask"),
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__hneu2_mask",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__hneu2_mask")),
                  CALL_FACTORY_ENTRY(
                      "__hneu2_mask",
                      CALL(MapNames::getDpctNamespace() +
                               "unordered_compare_mask",
                           ARG(0), ARG(1),
                           LITERAL("std::not_equal_to<>()"))))))};
}
