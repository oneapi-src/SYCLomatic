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
      MATH_API_REWRITERS_V2(
          "__vabs2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vabs2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vabs2<unsigned>",
                                      ARG(0)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vabs2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_unary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0),
                       LITERAL(MapNames::getDpctNamespace() + "abs()")))))
      // __vabs4
      MATH_API_REWRITERS_V2(
          "__vabs4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vabs4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vabs4<unsigned>",
                                      ARG(0)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vabs4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_unary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0),
                       LITERAL(MapNames::getDpctNamespace() + "abs()")))))
      // __vabsdiffs2
      MATH_API_REWRITERS_V2(
          "__vabsdiffs2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vabsdiffs2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vabsdiffs2<unsigned>",
                       ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vabsdiffs2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "abs_diff()")))))
      // __vabsdiffs4
      MATH_API_REWRITERS_V2(
          "__vabsdiffs4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vabsdiffs4",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vabsdiffs4<unsigned>",
                       ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vabsdiffs4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "abs_diff()")))))
      // __vabsdiffu2
      MATH_API_REWRITERS_V2(
          "__vabsdiffu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vabsdiffu2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vabsdiffu2<unsigned>",
                       ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vabsdiffu2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "abs_diff()")))))
      // __vabsdiffu4
      MATH_API_REWRITERS_V2(
          "__vabsdiffu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vabsdiffu4",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vabsdiffu4<unsigned>",
                       ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vabsdiffu4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "abs_diff()")))))
      // __vabsss2
      MATH_API_REWRITERS_V2(
          "__vabsss2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vabsss2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vabsss2<unsigned>",
                                      ARG(0)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vabsss2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), LITERAL("0"),
                       LITERAL(MapNames::getDpctNamespace() + "abs_diff()")))))
      // __vabsss4
      MATH_API_REWRITERS_V2(
          "__vabsss4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vabsss4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vabsss4<unsigned>",
                                      ARG(0)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vabsss4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0), LITERAL("0"),
                       LITERAL(MapNames::getDpctNamespace() + "abs_diff()")))))
      // __vadd2
      MATH_API_REWRITERS_V2(
          "__vadd2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vadd2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vadd2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vadd2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1), LITERAL("std::plus<>()")))))
      // __vadd4
      MATH_API_REWRITERS_V2(
          "__vadd4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vadd4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vadd4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vadd4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1), LITERAL("std::plus<>()")))))
      // __vaddss2
      MATH_API_REWRITERS_V2(
          "__vaddss2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vaddss2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vaddss2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vaddss2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "add_sat()")))))
      // __vaddss4
      MATH_API_REWRITERS_V2(
          "__vaddss4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vaddss4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vaddss4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vaddss4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "add_sat()")))))
      // __vaddus2
      MATH_API_REWRITERS_V2(
          "__vaddus2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vaddus2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vaddus2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vaddus2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "add_sat()")))))
      // __vaddus4
      MATH_API_REWRITERS_V2(
          "__vaddus4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vaddus4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vaddus4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vaddus4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "add_sat()")))))
      // __vavgs2
      MATH_API_REWRITERS_V2(
          "__vavgs2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vavgs2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vavgs2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vavgs2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "rhadd()")))))
      // __vavgs4
      MATH_API_REWRITERS_V2(
          "__vavgs4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vavgs4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vavgs4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vavgs4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "rhadd()")))))
      // __vavgu2
      MATH_API_REWRITERS_V2(
          "__vavgu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vavgu2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vavgu2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vavgu2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "rhadd()")))))
      // __vavgu4
      MATH_API_REWRITERS_V2(
          "__vavgu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vavgu4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vavgu4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vavgu4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "rhadd()")))))
      // __vcmpeq2
      MATH_API_REWRITERS_V2(
          "__vcmpeq2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vcmpeq2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vcmpeq2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpeq4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vcmpeq4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vcmpeq4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpges2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpges2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpges2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpges4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpges4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpges4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpgeu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpgeu2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpgeu2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpgeu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpgeu4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpgeu4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpgts2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpgts2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpgts2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpgts4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpgts4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpgts4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpgtu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpgtu2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpgtu2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpgtu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpgtu4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpgtu4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmples2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmples2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmples2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmples4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmples4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmples4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpleu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpleu2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpleu2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpleu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpleu4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpleu4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmplts2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmplts2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmplts2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmplts4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmplts4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmplts4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpltu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpltu2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpltu2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpltu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vcmpltu4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vcmpltu4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpne2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vcmpne2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vcmpne2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vcmpne4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vcmpne4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vcmpne4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
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
      MATH_API_REWRITERS_V2(
          "__vhaddu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vhaddu2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vhaddu2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vhaddu2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "hadd()")))))
      // __vhaddu4
      MATH_API_REWRITERS_V2(
          "__vhaddu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vhaddu4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vhaddu4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vhaddu4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "hadd()")))))
      // __viaddmax_s16x2
      MATH_API_REWRITERS_V2(
          "__viaddmax_s16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__viaddmax_s16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::viaddmax_s16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmax_s16x2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1), ARG(2), LITERAL("std::plus<>()"),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()")))))
      // __viaddmax_s16x2_relu
      MATH_API_REWRITERS_V2(
          "__viaddmax_s16x2_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__viaddmax_s16x2_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::viaddmax_s16x2_relu<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmax_s16x2_relu",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1), ARG(2), LITERAL("std::plus<>()"),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()"),
                       LITERAL("true")))))
      // __viaddmax_s32
      MATH_API_REWRITERS_V2(
          "__viaddmax_s32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__viaddmax_s32",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::viaddmax_s32<int>",
                                      ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmax_s32",
                  CALL(MapNames::getClNamespace() + "max<int>",
                       BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                       ARG(2)))))
      // __viaddmax_s32_relu
      MATH_API_REWRITERS_V2(
          "__viaddmax_s32_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__viaddmax_s32_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::viaddmax_s32_relu<int>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmax_s32_relu",
                  CALL(MapNames::getDpctNamespace() + "relu<int>",
                       CALL(MapNames::getClNamespace() + "max<int>",
                            BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                            ARG(2))))))
      // __viaddmax_u16x2
      MATH_API_REWRITERS_V2(
          "__viaddmax_u16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__viaddmax_u16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::viaddmax_u16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmax_u16x2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1), ARG(2), LITERAL("std::plus<>()"),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()")))))
      // __viaddmax_u32
      MATH_API_REWRITERS_V2(
          "__viaddmax_u32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__viaddmax_u32",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::viaddmax_u32<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmax_u32",
                  CALL(MapNames::getClNamespace() + "max<unsigned>",
                       BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                       ARG(2)))))
      // __viaddmin_s16x2
      MATH_API_REWRITERS_V2(
          "__viaddmin_s16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__viaddmin_s16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::viaddmin_s16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmin_s16x2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1), ARG(2), LITERAL("std::plus<>()"),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()")))))
      // __viaddmin_s16x2_relu
      MATH_API_REWRITERS_V2(
          "__viaddmin_s16x2_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__viaddmin_s16x2_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::viaddmin_s16x2_relu<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmin_s16x2_relu",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1), ARG(2), LITERAL("std::plus<>()"),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()"),
                       LITERAL("true")))))
      // __viaddmin_s32
      MATH_API_REWRITERS_V2(
          "__viaddmin_s32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__viaddmin_s32",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::viaddmin_s32<int>",
                                      ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmin_s32",
                  CALL(MapNames::getClNamespace() + "min<int>",
                       BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                       ARG(2)))))
      // __viaddmin_s32_relu
      MATH_API_REWRITERS_V2(
          "__viaddmin_s32_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__viaddmin_s32_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::viaddmin_s32_relu<int>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmin_s32_relu",
                  CALL(MapNames::getDpctNamespace() + "relu<int>",
                       CALL(MapNames::getClNamespace() + "min<int>",
                            BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                            ARG(2))))))
      // __viaddmin_u16x2
      MATH_API_REWRITERS_V2(
          "__viaddmin_u16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__viaddmin_u16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::viaddmin_u16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmin_u16x2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1), ARG(2), LITERAL("std::plus<>()"),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()")))))
      // __viaddmin_u32
      MATH_API_REWRITERS_V2(
          "__viaddmin_u32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__viaddmin_u32",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::viaddmin_u32<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__viaddmin_u32",
                  CALL(MapNames::getClNamespace() + "min<unsigned>",
                       BO(BinaryOperatorKind::BO_Add, ARG(0), ARG(1)),
                       ARG(2)))))
      // __vibmax_s16x2
      MATH_API_REWRITERS_V2(
          "__vibmax_s16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vibmax_s16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vibmax_s16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2), ARG(3)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vibmax_s16x2",
                  CALL(MapNames::getDpctNamespace() +
                           "vectorized_with_pred<short>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()"),
                       ARG(2), ARG(3)))))
      // __vibmax_s32
      MATH_API_REWRITERS_V2(
          "__vibmax_s32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vibmax_s32",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vibmax_s32<int>",
                                      ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vibmax_s32",
                  CALL(MapNames::getDpctNamespace() + "maximum()", ARG(0),
                       ARG(1), ARG(2)))))
      // __vibmax_u16x2
      MATH_API_REWRITERS_V2(
          "__vibmax_u16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vibmax_u16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vibmax_u16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2), ARG(3)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vibmax_u16x2",
                  CALL(MapNames::getDpctNamespace() +
                           "vectorized_with_pred<unsigned short>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()"),
                       ARG(2), ARG(3)))))
      // __vibmax_u32
      MATH_API_REWRITERS_V2(
          "__vibmax_u32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vibmax_u32",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vibmax_u32<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vibmax_u32",
                  CALL(MapNames::getDpctNamespace() + "maximum()", ARG(0),
                       ARG(1), ARG(2)))))
      // __vibmin_s16x2
      MATH_API_REWRITERS_V2(
          "__vibmin_s16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vibmin_s16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vibmin_s16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2), ARG(3)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vibmin_s16x2",
                  CALL(MapNames::getDpctNamespace() +
                           "vectorized_with_pred<short>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()"),
                       ARG(2), ARG(3)))))
      // __vibmin_s32
      MATH_API_REWRITERS_V2(
          "__vibmin_s32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vibmin_s32",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vibmin_s32<int>",
                                      ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vibmin_s32",
                  CALL(MapNames::getDpctNamespace() + "minimum()", ARG(0),
                       ARG(1), ARG(2)))))
      // __vibmin_u16x2
      MATH_API_REWRITERS_V2(
          "__vibmin_u16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vibmin_u16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vibmin_u16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2), ARG(3)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vibmin_u16x2",
                  CALL(MapNames::getDpctNamespace() +
                           "vectorized_with_pred<unsigned short>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()"),
                       ARG(2), ARG(3)))))
      // __vibmin_u32
      MATH_API_REWRITERS_V2(
          "__vibmin_u32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vibmin_u32",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vibmin_u32<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vibmin_u32",
                  CALL(MapNames::getDpctNamespace() + "minimum()", ARG(0),
                       ARG(1), ARG(2)))))
      // __vimax3_s16x2
      MATH_API_REWRITERS_V2(
          "__vimax3_s16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimax3_s16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimax3_s16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimax3_s16x2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1), ARG(2),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()"),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()")))))
      // __vimax3_s16x2_relu
      MATH_API_REWRITERS_V2(
          "__vimax3_s16x2_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimax3_s16x2_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimax3_s16x2_relu<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimax3_s16x2_relu",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1), ARG(2),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()"),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()"),
                       LITERAL("true")))))
      // __vimax3_s32
      MATH_API_REWRITERS_V2(
          "__vimax3_s32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vimax3_s32",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vimax3_s32<int>",
                                      ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimax3_s32",
                  CALL(MapNames::getClNamespace() + "max<int>",
                       CALL(MapNames::getClNamespace() + "max<int>", ARG(0),
                            ARG(1)),
                       ARG(2)))))
      // __vimax3_s32_relu
      MATH_API_REWRITERS_V2(
          "__vimax3_s32_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimax3_s32_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimax3_s32_relu<int>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimax3_s32_relu",
                  CALL(MapNames::getDpctNamespace() + "relu<int>",
                       CALL(MapNames::getClNamespace() + "max<int>",
                            CALL(MapNames::getClNamespace() + "max<int>",
                                 ARG(0), ARG(1)),
                            ARG(2))))))
      // __vimax3_u16x2
      MATH_API_REWRITERS_V2(
          "__vimax3_u16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimax3_u16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimax3_u16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimax3_u16x2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1), ARG(2),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()"),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()")))))
      // __vimax3_u32
      MATH_API_REWRITERS_V2(
          "__vimax3_u32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimax3_u32",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimax3_u32<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimax3_u32",
                  CALL(MapNames::getClNamespace() + "max<unsigned>",
                       CALL(MapNames::getClNamespace() + "max<unsigned>",
                            ARG(0), ARG(1)),
                       ARG(2)))))
      // __vimax_s16x2_relu
      MATH_API_REWRITERS_V2(
          "__vimax_s16x2_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimax_s16x2_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimax_s16x2_relu<unsigned>",
                       ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimax_s16x2_relu",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()"),
                       LITERAL("true")))))
      // __vimax_s32_relu
      MATH_API_REWRITERS_V2(
          "__vimax_s32_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimax_s32_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimax_s32_relu<int>",
                       ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimax_s32_relu",
                  CALL(MapNames::getDpctNamespace() + "relu<int>",
                       CALL(MapNames::getClNamespace() + "max<int>", ARG(0),
                            ARG(1))))))
      // __vimin3_s16x2
      MATH_API_REWRITERS_V2(
          "__vimin3_s16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimin3_s16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimin3_s16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimin3_s16x2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1), ARG(2),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()"),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()")))))
      // __vimin3_s16x2_relu
      MATH_API_REWRITERS_V2(
          "__vimin3_s16x2_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimin3_s16x2_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimin3_s16x2_relu<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimin3_s16x2_relu",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1), ARG(2),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()"),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()"),
                       LITERAL("true")))))
      // __vimin3_s32
      MATH_API_REWRITERS_V2(
          "__vimin3_s32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vimin3_s32",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vimin3_s32<int>",
                                      ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimin3_s32",
                  CALL(MapNames::getClNamespace() + "min<int>",
                       CALL(MapNames::getClNamespace() + "min<int>", ARG(0),
                            ARG(1)),
                       ARG(2)))))
      // __vimin3_s32_relu
      MATH_API_REWRITERS_V2(
          "__vimin3_s32_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimin3_s32_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimin3_s32_relu<int>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimin3_s32_relu",
                  CALL(MapNames::getDpctNamespace() + "relu<int>",
                       CALL(MapNames::getClNamespace() + "min<int>",
                            CALL(MapNames::getClNamespace() + "min<int>",
                                 ARG(0), ARG(1)),
                            ARG(2))))))
      // __vimin3_u16x2
      MATH_API_REWRITERS_V2(
          "__vimin3_u16x2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimin3_u16x2",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimin3_u16x2<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimin3_u16x2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_ternary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1), ARG(2),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()"),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()")))))
      // __vimin3_u32
      MATH_API_REWRITERS_V2(
          "__vimin3_u32",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimin3_u32",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimin3_u32<unsigned>",
                       ARG(0), ARG(1), ARG(2)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimin3_u32",
                  CALL(MapNames::getClNamespace() + "min<unsigned>",
                       CALL(MapNames::getClNamespace() + "min<unsigned>",
                            ARG(0), ARG(1)),
                       ARG(2)))))
      // __vimin_s16x2_relu
      MATH_API_REWRITERS_V2(
          "__vimin_s16x2_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimin_s16x2_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimin_s16x2_relu<unsigned>",
                       ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimin_s16x2_relu",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()"),
                       LITERAL("true")))))
      // __vimin_s32_relu
      MATH_API_REWRITERS_V2(
          "__vimin_s32_relu",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vimin_s32_relu",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::vimin_s32_relu<int>",
                       ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vimin_s32_relu",
                  CALL(MapNames::getDpctNamespace() + "relu<int>",
                       CALL(MapNames::getClNamespace() + "min<int>", ARG(0),
                            ARG(1))))))
      // __vmaxs2
      MATH_API_REWRITERS_V2(
          "__vmaxs2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vmaxs2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vmaxs2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vmaxs2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()")))))
      // __vmaxs4
      MATH_API_REWRITERS_V2(
          "__vmaxs4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vmaxs4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vmaxs4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vmaxs4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()")))))
      // __vmaxu2
      MATH_API_REWRITERS_V2(
          "__vmaxu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vmaxu2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vmaxu2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vmaxu2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()")))))
      // __vmaxu4
      MATH_API_REWRITERS_V2(
          "__vmaxu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vmaxu4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vmaxu4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vmaxu4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "maximum()")))))
      // __vmins2
      MATH_API_REWRITERS_V2(
          "__vmins2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vmins2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vmins2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vmins2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()")))))
      // __vmins4
      MATH_API_REWRITERS_V2(
          "__vmins4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vmins4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vmins4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vmins4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()")))))
      // __vminu2
      MATH_API_REWRITERS_V2(
          "__vminu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vminu2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vminu2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vminu2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()")))))
      // __vminu4
      MATH_API_REWRITERS_V2(
          "__vminu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vminu4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vminu4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vminu4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "minimum()")))))
      // __vneg2
      MATH_API_REWRITERS_V2(
          "__vneg2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vneg2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vneg2<unsigned>",
                                      ARG(0)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vneg2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_unary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), LITERAL("std::negate<>()")))))
      // __vneg4
      MATH_API_REWRITERS_V2(
          "__vneg4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vneg4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vneg4<unsigned>",
                                      ARG(0)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY("__vneg4",
                                 CALL(MapNames::getDpctNamespace() +
                                          "vectorized_unary<" +
                                          MapNames::getClNamespace() + "char4>",
                                      ARG(0), LITERAL("std::negate<>()")))))
      // __vnegss2
      MATH_API_REWRITERS_V2(
          "__vnegss2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vnegss2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vnegss2<unsigned>",
                                      ARG(0)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vnegss2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       LITERAL("0"), ARG(0),
                       LITERAL(MapNames::getDpctNamespace() + "sub_sat()")))))
      // __vnegss4
      MATH_API_REWRITERS_V2(
          "__vnegss4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vnegss4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vnegss4<unsigned>",
                                      ARG(0)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vnegss4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       LITERAL("0"), ARG(0),
                       LITERAL(MapNames::getDpctNamespace() + "sub_sat()")))))
      // __vsads2
      MATH_API_REWRITERS_V2(
          "__vsads2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsads2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsads2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsads2", CALL(MapNames::getDpctNamespace() +
                                       "vectorized_sum_abs_diff<" +
                                       MapNames::getClNamespace() + "short2>",
                                   ARG(0), ARG(1)))))
      // __vsads4
      MATH_API_REWRITERS_V2(
          "__vsads4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsads4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsads4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY("__vsads4",
                                 CALL(MapNames::getDpctNamespace() +
                                          "vectorized_sum_abs_diff<" +
                                          MapNames::getClNamespace() + "char4>",
                                      ARG(0), ARG(1)))))
      // __vsadu2
      MATH_API_REWRITERS_V2(
          "__vsadu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsadu2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsadu2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsadu2", CALL(MapNames::getDpctNamespace() +
                                       "vectorized_sum_abs_diff<" +
                                       MapNames::getClNamespace() + "ushort2>",
                                   ARG(0), ARG(1)))))
      // __vsadu4
      MATH_API_REWRITERS_V2(
          "__vsadu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsadu4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsadu4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsadu4", CALL(MapNames::getDpctNamespace() +
                                       "vectorized_sum_abs_diff<" +
                                       MapNames::getClNamespace() + "uchar4>",
                                   ARG(0), ARG(1)))))
      // __vseteq2
      MATH_API_REWRITERS_V2(
          "__vseteq2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vseteq2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vseteq2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vseteq2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL("std::equal_to<unsigned short>()")))))
      // __vseteq4
      MATH_API_REWRITERS_V2(
          "__vseteq4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vseteq4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vseteq4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vseteq4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL("std::equal_to<unsigned char>()")))))
      // __vsetges2
      MATH_API_REWRITERS_V2(
          "__vsetges2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetges2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetges2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetges2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1),
                       LITERAL("std::greater_equal<short>()")))))
      // __vsetges4
      MATH_API_REWRITERS_V2(
          "__vsetges4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetges4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetges4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetges4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0), ARG(1), LITERAL("std::greater_equal<char>()")))))
      // __vsetgeu2
      MATH_API_REWRITERS_V2(
          "__vsetgeu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetgeu2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetgeu2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetgeu2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL("std::greater_equal<unsigned short>()")))))
      // __vsetgeu4
      MATH_API_REWRITERS_V2(
          "__vsetgeu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetgeu4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetgeu4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetgeu4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL("std::greater_equal<unsigned char>()")))))
      // __vsetgts2
      MATH_API_REWRITERS_V2(
          "__vsetgts2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetgts2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetgts2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetgts2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1), LITERAL("std::greater<short>()")))))
      // __vsetgts4
      MATH_API_REWRITERS_V2(
          "__vsetgts4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetgts4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetgts4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetgts4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0), ARG(1), LITERAL("std::greater<char>()")))))
      // __vsetgtu2
      MATH_API_REWRITERS_V2(
          "__vsetgtu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetgtu2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetgtu2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetgtu2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL("std::greater<unsigned short>()")))))
      // __vsetgtu4
      MATH_API_REWRITERS_V2(
          "__vsetgtu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetgtu4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetgtu4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetgtu4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL("std::greater<unsigned char>()")))))
      // __vsetles2
      MATH_API_REWRITERS_V2(
          "__vsetles2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetles2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetles2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetles2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1), LITERAL("std::less_equal<short>()")))))
      // __vsetles4
      MATH_API_REWRITERS_V2(
          "__vsetles4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetles4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetles4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetles4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0), ARG(1), LITERAL("std::less_equal<char>()")))))
      // __vsetleu2
      MATH_API_REWRITERS_V2(
          "__vsetleu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetleu2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetleu2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetleu2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL("std::less_equal<unsigned short>()")))))
      // __vsetleu4
      MATH_API_REWRITERS_V2(
          "__vsetleu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetleu4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetleu4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetleu4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL("std::less_equal<unsigned char>()")))))
      // __vsetlts2
      MATH_API_REWRITERS_V2(
          "__vsetlts2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetlts2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetlts2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetlts2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1), LITERAL("std::less<short>()")))))
      // __vsetlts4
      MATH_API_REWRITERS_V2(
          "__vsetlts4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetlts4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetlts4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetlts4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0), ARG(1), LITERAL("std::less<char>()")))))
      // __vsetltu2
      MATH_API_REWRITERS_V2(
          "__vsetltu2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetltu2", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetltu2<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetltu2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL("std::less<unsigned short>()")))))
      // __vsetltu4
      MATH_API_REWRITERS_V2(
          "__vsetltu4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY(
                  "__vsetltu4", CALL(MapNames::getClNamespace() +
                                         "ext::intel::math::vsetltu4<unsigned>",
                                     ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetltu4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1), LITERAL("std::less<unsigned char>()")))))
      // __vsetne2
      MATH_API_REWRITERS_V2(
          "__vsetne2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsetne2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsetne2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetne2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL("std::not_equal_to<unsigned short>()")))))
      // __vsetne4
      MATH_API_REWRITERS_V2(
          "__vsetne4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsetne4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsetne4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsetne4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL("std::not_equal_to<unsigned char>()")))))
      // __vsub2
      MATH_API_REWRITERS_V2(
          "__vsub2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsub2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsub2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsub2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1), LITERAL("std::minus<>()")))))
      // __vsub4
      MATH_API_REWRITERS_V2(
          "__vsub4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsub4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsub4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsub4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1), LITERAL("std::minus<>()")))))
      // __vsubss2
      MATH_API_REWRITERS_V2(
          "__vsubss2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsubss2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsubss2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsubss2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "short2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "sub_sat()")))))
      // __vsubss4
      MATH_API_REWRITERS_V2(
          "__vsubss4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsubss4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsubss4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsubss4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "char4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "sub_sat()")))))
      // __vsubus2
      MATH_API_REWRITERS_V2(
          "__vsubus2",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsubus2",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsubus2<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsubus2",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "ushort2>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "sub_sat()")))))
      // __vsubus4
      MATH_API_REWRITERS_V2(
          "__vsubus4",
          MATH_API_REWRITER_PAIR(
              math::Tag::math_libdevice,
              CALL_FACTORY_ENTRY("__vsubus4",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::vsubus4<unsigned>",
                                      ARG(0), ARG(1)))),
          MATH_API_REWRITER_PAIR(
              math::Tag::emulation,
              CALL_FACTORY_ENTRY(
                  "__vsubus4",
                  CALL(MapNames::getDpctNamespace() + "vectorized_binary<" +
                           MapNames::getClNamespace() + "uchar4>",
                       ARG(0), ARG(1),
                       LITERAL(MapNames::getDpctNamespace() + "sub_sat()")))))};
}
