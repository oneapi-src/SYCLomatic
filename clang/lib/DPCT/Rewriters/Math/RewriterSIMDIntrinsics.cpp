//===--------------- RewriterSIMDIntrinsics.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"
#include "CommonMacroDefinition.h"

using namespace clang::dpct;

RewriterMap dpct::createSIMDIntrinsicsRewriterMap() {
  return RewriterMap{
      // __vabs2
      MATH_API_REWRITER_DEVICE(
          "__vabs2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vabs2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vabs2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vabs2",
                                          ARG(0)))),
              EMPTY_FACTORY_ENTRY("__vabs2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vabs2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_unary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0),
                           LITERAL(MapNames::getDpctNamespace() + "abs()"))))))
      // __vabs4
      MATH_API_REWRITER_DEVICE(
          "__vabs4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vabs4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vabs4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vabs4",
                                          ARG(0)))),
              EMPTY_FACTORY_ENTRY("__vabs4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vabs4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_unary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0),
                           LITERAL(MapNames::getDpctNamespace() + "abs()"))))))
      // __vabsdiffs2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vabsdiffs2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "abs_diff()"))))))
      // __vabsdiffs4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vabsdiffs4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "abs_diff()"))))))
      // __vabsdiffu2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vabsdiffu2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "abs_diff()"))))))
      // __vabsdiffu4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vabsdiffu4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "abs_diff()"))))))
      // __vabsss2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vabsss2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0), LITERAL("0"),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "abs_diff()"))))))
      // __vabsss4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vabsss4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0), LITERAL("0"),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "abs_diff()"))))))
      // __vadd2
      MATH_API_REWRITER_DEVICE(
          "__vadd2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vadd2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vadd2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vadd2",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vadd2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vadd2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1), LITERAL("std::plus<>()"))))))
      // __vadd4
      MATH_API_REWRITER_DEVICE(
          "__vadd4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vadd4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vadd4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vadd4",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vadd4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vadd4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1), LITERAL("std::plus<>()"))))))
      // __vaddss2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vaddss2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "add_sat()"))))))
      // __vaddss4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vaddss4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "add_sat()"))))))
      // __vaddus2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vaddus2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "add_sat()"))))))
      // __vaddus4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vaddus4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "add_sat()"))))))
      // __vavgs2
      MATH_API_REWRITER_DEVICE(
          "__vavgs2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vavgs2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vavgs2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vavgs2",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vavgs2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vavgs2",
                      CALL(
                          MapNames::getDpctNamespace() + "vectorized_binary<" +
                              MapNames::getClNamespace() + "short2>",
                          ARG(0), ARG(1),
                          LITERAL(MapNames::getDpctNamespace() + "rhadd()"))))))
      // __vavgs4
      MATH_API_REWRITER_DEVICE(
          "__vavgs4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vavgs4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vavgs4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vavgs4",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vavgs4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vavgs4",
                      CALL(
                          MapNames::getDpctNamespace() + "vectorized_binary<" +
                              MapNames::getClNamespace() + "char4>",
                          ARG(0), ARG(1),
                          LITERAL(MapNames::getDpctNamespace() + "rhadd()"))))))
      // __vavgu2
      MATH_API_REWRITER_DEVICE(
          "__vavgu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vavgu2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vavgu2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vavgu2",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vavgu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vavgu2",
                      CALL(
                          MapNames::getDpctNamespace() + "vectorized_binary<" +
                              MapNames::getClNamespace() + "ushort2>",
                          ARG(0), ARG(1),
                          LITERAL(MapNames::getDpctNamespace() + "rhadd()"))))))
      // __vavgu4
      MATH_API_REWRITER_DEVICE(
          "__vavgu4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vavgu4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vavgu4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vavgu4",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vavgu4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vavgu4",
                      CALL(
                          MapNames::getDpctNamespace() + "vectorized_binary<" +
                              MapNames::getClNamespace() + "uchar4>",
                          ARG(0), ARG(1),
                          LITERAL(MapNames::getDpctNamespace() + "rhadd()"))))))
      // __vcmpeq2
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpeq2",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpeq2")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY("__vcmpeq2",
                                         CALL(MapNames::getDpctNamespace() +
                                                  "vectorized_binary<" +
                                                  MapNames::getClNamespace() +
                                                  "ushort2>",
                                              ARG(0), ARG(1),
                                              LITERAL("std::equal_to<>()")))))))
      // __vcmpeq4
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpeq4",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpeq4")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY("__vcmpeq4",
                                         CALL(MapNames::getDpctNamespace() +
                                                  "vectorized_binary<" +
                                                  MapNames::getClNamespace() +
                                                  "uchar4>",
                                              ARG(0), ARG(1),
                                              LITERAL("std::equal_to<>()")))))))
      // __vcmpges2
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpges2",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpges2")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpges2",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "short2>",
                               ARG(0), ARG(1),
                               LITERAL("std::greater_equal<>()")))))))
      // __vcmpges4
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpges4",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpges4")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpges4",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "char4>",
                               ARG(0), ARG(1),
                               LITERAL("std::greater_equal<>()")))))))
      // __vcmpgeu2
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpgeu2",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpgeu2")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpgeu2",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "ushort2>",
                               ARG(0), ARG(1),
                               LITERAL("std::greater_equal<>()")))))))
      // __vcmpgeu4
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpgeu4",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpgeu4")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpgeu4",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "uchar4>",
                               ARG(0), ARG(1),
                               LITERAL("std::greater_equal<>()")))))))
      // __vcmpgts2
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpgts2",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpgts2")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpgts2",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "short2>",
                               ARG(0), ARG(1), LITERAL("std::greater<>()")))))))
      // __vcmpgts4
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpgts4",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpgts4")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpgts4",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "char4>",
                               ARG(0), ARG(1), LITERAL("std::greater<>()")))))))
      // __vcmpgtu2
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpgtu2",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpgtu2")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpgtu2",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "ushort2>",
                               ARG(0), ARG(1), LITERAL("std::greater<>()")))))))
      // __vcmpgtu4
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpgtu4",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpgtu4")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpgtu4",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "uchar4>",
                               ARG(0), ARG(1), LITERAL("std::greater<>()")))))))
      // __vcmples2
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmples2",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmples2")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmples2",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "short2>",
                               ARG(0), ARG(1),
                               LITERAL("std::less_equal<>()")))))))
      // __vcmples4
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmples4",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmples4")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmples4",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "char4>",
                               ARG(0), ARG(1),
                               LITERAL("std::less_equal<>()")))))))
      // __vcmpleu2
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpleu2",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpleu2")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpleu2",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "ushort2>",
                               ARG(0), ARG(1),
                               LITERAL("std::less_equal<>()")))))))
      // __vcmpleu4
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpleu4",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpleu4")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpleu4",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "uchar4>",
                               ARG(0), ARG(1),
                               LITERAL("std::less_equal<>()")))))))
      // __vcmplts2
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmplts2",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmplts2")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmplts2",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "short2>",
                               ARG(0), ARG(1), LITERAL("std::less<>()")))))))
      // __vcmplts4
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmplts4",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmplts4")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmplts4",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "char4>",
                               ARG(0), ARG(1), LITERAL("std::less<>()")))))))
      // __vcmpltu2
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpltu2",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpltu2")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpltu2",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "ushort2>",
                               ARG(0), ARG(1), LITERAL("std::less<>()")))))))
      // __vcmpltu4
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpltu4",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpltu4")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpltu4",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "uchar4>",
                               ARG(0), ARG(1), LITERAL("std::less<>()")))))))
      // __vcmpne2
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpne2",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpne2")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpne2",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "ushort2>",
                               ARG(0), ARG(1),
                               LITERAL("std::not_equal_to<>()")))))))
      // __vcmpne4
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
              CONDITIONAL_FACTORY_ENTRY(
                  UseSYCLCompat,
                  UNSUPPORT_FACTORY_ENTRY("__vcmpne4",
                                          Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                          LITERAL("__vcmpne4")),
                  FEATURE_REQUEST_FACTORY(
                      HelperFeatureEnum::device_ext,
                      CALL_FACTORY_ENTRY(
                          "__vcmpne4",
                          CALL(MapNames::getDpctNamespace() +
                                   "vectorized_binary<" +
                                   MapNames::getClNamespace() + "uchar4>",
                               ARG(0), ARG(1),
                               LITERAL("std::not_equal_to<>()")))))))
      // __vhaddu2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vhaddu2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() + "hadd()"))))))
      // __vhaddu4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vhaddu4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() + "hadd()"))))))
      // __vmaxs2
      MATH_API_REWRITER_DEVICE(
          "__vmaxs2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vmaxs2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vmaxs2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vmaxs2",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vmaxs2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vmaxs2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "maximum()"))))))
      // __vmaxs4
      MATH_API_REWRITER_DEVICE(
          "__vmaxs4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vmaxs4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vmaxs4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vmaxs4",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vmaxs4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vmaxs4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "maximum()"))))))
      // __vmaxu2
      MATH_API_REWRITER_DEVICE(
          "__vmaxu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vmaxu2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vmaxu2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vmaxu2",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vmaxu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vmaxu2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "maximum()"))))))
      // __vmaxu4
      MATH_API_REWRITER_DEVICE(
          "__vmaxu4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vmaxu4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vmaxu4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vmaxu4",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vmaxu4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vmaxu4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "maximum()"))))))
      // __vmins2
      MATH_API_REWRITER_DEVICE(
          "__vmins2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vmins2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vmins2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vmins2",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vmins2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vmins2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "minimum()"))))))
      // __vmins4
      MATH_API_REWRITER_DEVICE(
          "__vmins4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vmins4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vmins4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vmins4",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vmins4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vmins4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "minimum()"))))))
      // __vminu2
      MATH_API_REWRITER_DEVICE(
          "__vminu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vminu2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vminu2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vminu2",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vminu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vminu2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "minimum()"))))))
      // __vminu4
      MATH_API_REWRITER_DEVICE(
          "__vminu4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vminu4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vminu4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vminu4",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vminu4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vminu4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "minimum()"))))))
      // __vneg2
      MATH_API_REWRITER_DEVICE(
          "__vneg2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vneg2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vneg2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vneg2",
                                          ARG(0)))),
              EMPTY_FACTORY_ENTRY("__vneg2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vneg2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_unary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0), LITERAL("std::negate<>()"))))))
      // __vneg4
      MATH_API_REWRITER_DEVICE(
          "__vneg4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vneg4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vneg4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vneg4",
                                          ARG(0)))),
              EMPTY_FACTORY_ENTRY("__vneg4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vneg4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_unary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0), LITERAL("std::negate<>()"))))))
      // __vnegss2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vnegss2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "short2>",
                           LITERAL("0"), ARG(0),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "sub_sat()"))))))
      // __vnegss4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vnegss4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "char4>",
                           LITERAL("0"), ARG(0),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "sub_sat()"))))))
      // __vsads2
      MATH_API_REWRITER_DEVICE(
          "__vsads2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vsads2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vsads2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vsads2",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vsads2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  ENTRY_RENAMED("__vsads2", MapNames::getDpctNamespace() +
                                                "vectorized_sum_abs_diff<" +
                                                MapNames::getClNamespace() +
                                                "short2>"))))
      // __vsads4
      MATH_API_REWRITER_DEVICE(
          "__vsads4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vsads4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vsads4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vsads4",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vsads4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  ENTRY_RENAMED("__vsads4", MapNames::getDpctNamespace() +
                                                "vectorized_sum_abs_diff<" +
                                                MapNames::getClNamespace() +
                                                "char4>"))))
      // __vsadu2
      MATH_API_REWRITER_DEVICE(
          "__vsadu2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vsadu2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vsadu2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vsadu2",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vsadu2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  ENTRY_RENAMED("__vsadu2", MapNames::getDpctNamespace() +
                                                "vectorized_sum_abs_diff<" +
                                                MapNames::getClNamespace() +
                                                "ushort2>"))))
      // __vsadu4
      MATH_API_REWRITER_DEVICE(
          "__vsadu4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vsadu4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vsadu4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vsadu4",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vsadu4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  ENTRY_RENAMED("__vsadu4", MapNames::getDpctNamespace() +
                                                "vectorized_sum_abs_diff<" +
                                                MapNames::getClNamespace() +
                                                "uchar4>"))))
      // __vseteq2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vseteq2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL("std::equal_to<unsigned short>()"))))))
      // __vseteq4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vseteq4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL("std::equal_to<unsigned char>()"))))))
      // __vsetges2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetges2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0), ARG(1),
                           LITERAL("std::greater_equal<short>()"))))))
      // __vsetges4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetges4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0), ARG(1),
                           LITERAL("std::greater_equal<char>()"))))))
      // __vsetgeu2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetgeu2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL("std::greater_equal<unsigned short>()"))))))
      // __vsetgeu4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetgeu4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL("std::greater_equal<unsigned char>()"))))))
      // __vsetgts2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetgts2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0), ARG(1), LITERAL("std::greater<short>()"))))))
      // __vsetgts4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetgts4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0), ARG(1), LITERAL("std::greater<char>()"))))))
      // __vsetgtu2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetgtu2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL("std::greater<unsigned short>()"))))))
      // __vsetgtu4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetgtu4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL("std::greater<unsigned char>()"))))))
      // __vsetles2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetles2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0), ARG(1),
                           LITERAL("std::less_equal<short>()"))))))
      // __vsetles4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetles4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0), ARG(1),
                           LITERAL("std::less_equal<char>()"))))))
      // __vsetleu2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetleu2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL("std::less_equal<unsigned short>()"))))))
      // __vsetleu4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetleu4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL("std::less_equal<unsigned char>()"))))))
      // __vsetlts2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetlts2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0), ARG(1), LITERAL("std::less<short>()"))))))
      // __vsetlts4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetlts4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0), ARG(1), LITERAL("std::less<char>()"))))))
      // __vsetltu2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetltu2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL("std::less<unsigned short>()"))))))
      // __vsetltu4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetltu4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL("std::less<unsigned char>()"))))))
      // __vsetne2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetne2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL("std::not_equal_to<unsigned short>()"))))))
      // __vsetne4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsetne4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL("std::not_equal_to<unsigned char>()"))))))
      // __vsub2
      MATH_API_REWRITER_DEVICE(
          "__vsub2",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vsub2"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vsub2",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vsub2",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vsub2"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsub2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1), LITERAL("std::minus<>()"))))))
      // __vsub4
      MATH_API_REWRITER_DEVICE(
          "__vsub4",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__vsub4"),
              HEADER_INSERT_FACTORY(
                  HeaderType::HT_SYCL_Math,
                  CALL_FACTORY_ENTRY("__vsub4",
                                     CALL(MapNames::getClNamespace() +
                                              "ext::intel::math::vsub4",
                                          ARG(0), ARG(1)))),
              EMPTY_FACTORY_ENTRY("__vsub4"),
              FEATURE_REQUEST_FACTORY(
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsub4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1), LITERAL("std::minus<>()"))))))
      // __vsubss2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsubss2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "short2>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "sub_sat()"))))))
      // __vsubss4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsubss4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "char4>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "sub_sat()"))))))
      // __vsubus2
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsubus2",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "ushort2>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "sub_sat()"))))))
      // __vsubus4
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
                  HelperFeatureEnum::device_ext,
                  CALL_FACTORY_ENTRY(
                      "__vsubus4",
                      CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                               MapNames::getClNamespace() + "uchar4>",
                           ARG(0), ARG(1),
                           LITERAL(MapNames::getDpctNamespace() +
                                   "sub_sat()"))))))};
}
