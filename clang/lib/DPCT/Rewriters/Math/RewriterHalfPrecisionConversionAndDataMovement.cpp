//===------- RewriterHalfPrecisionConversionAndDataMovement.cpp -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"
#include "CommonMacroDefinition.h"

using namespace clang::dpct;

RewriterMap dpct::createHalfPrecisionConversionAndDataMovementRewriterMap() {
  return RewriterMap{
      // __double2half
      CALL_FACTORY_ENTRY("__double2half",
                         CALL(MapNames::getClNamespace() + "half", ARG(0)))
      // __float22half2_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__float22half2_rn",
                  CALL(MapNames::getClNamespace() + "half2",
                       CALL(MapNames::getClNamespace() +
                                "ext::intel::math::float2half_rn",
                            ARRAY_SUBSCRIPT(ARG(0), LITERAL("0"))),
                       CALL(MapNames::getClNamespace() +
                                "ext::intel::math::float2half_rn",
                            ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")))))),
          MEMBER_CALL_HAS_EXPLICIT_TEMP_ARG_FACTORY_ENTRY(
              "__float22half2_rn", ARG(0), false,
              "convert<" + MapNames::getClNamespace() + "half, " +
                  MapNames::getClNamespace() + "rounding_mode::rte>"))
      // __float2half
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__float2half",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::float2half_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__float2half",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::automatic>"),
              LITERAL("0")))
      // __float2half2_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__float2half2_rn",
                  CALL(MapNames::getClNamespace() + "half2",
                       CALL(MapNames::getClNamespace() +
                                "ext::intel::math::float2half_rn",
                            ARG(0))))),
          MEMBER_CALL_FACTORY_ENTRY(
              "__float2half2_rn",
              CALL(MapNames::getClNamespace() + "float2", ARG(0)), false,
              "convert<" + MapNames::getClNamespace() + "half, " +
                  MapNames::getClNamespace() + "rounding_mode::rte>"))
      // __float2half_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__float2half_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::float2half_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__float2half_rd",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtn>"),
              LITERAL("0")))
      // __float2half_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__float2half_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::float2half_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__float2half_rn",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rte>"),
              LITERAL("0")))
      // __float2half_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__float2half_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::float2half_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__float2half_ru",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtp>"),
              LITERAL("0")))
      // __float2half_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__float2half_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::float2half_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__float2half_rz",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtz>"),
              LITERAL("0")))
      // __floats2half2_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__floats2half2_rn",
                  CALL(MapNames::getClNamespace() + "half2",
                       CALL(MapNames::getClNamespace() +
                                "ext::intel::math::float2half_rn",
                            ARG(0)),
                       CALL(MapNames::getClNamespace() +
                                "ext::intel::math::float2half_rn",
                            ARG(1))))),
          MEMBER_CALL_FACTORY_ENTRY(
              "__floats2half2_rn",
              CALL(MapNames::getClNamespace() + "float2", ARG(0), ARG(1)),
              false,
              "convert<" + MapNames::getClNamespace() + "half, " +
                  MapNames::getClNamespace() + "rounding_mode::rte>"))
      // __half22float2
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__half22float2",
                  CALL(MapNames::getClNamespace() + "float2",
                       CALL(MapNames::getClNamespace() +
                                "ext::intel::math::half2float",
                            ARRAY_SUBSCRIPT(ARG(0), LITERAL("0"))),
                       CALL(MapNames::getClNamespace() +
                                "ext::intel::math::half2float",
                            ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")))))),
          MEMBER_CALL_HAS_EXPLICIT_TEMP_ARG_FACTORY_ENTRY(
              "__half22float2", ARG(0), false,
              "convert<float, " + MapNames::getClNamespace() +
                  "rounding_mode::automatic>"))
      // __half2float
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2float",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2float",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2float",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<float, " + MapNames::getClNamespace() +
                              "rounding_mode::automatic>"),
              LITERAL("0")))
      // __half2half2
      CALL_FACTORY_ENTRY("__half2half2",
                         CALL(MapNames::getClNamespace() + "half2", ARG(0)))
      // __half2int_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2int_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2int_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2int_rd",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<int, " + MapNames::getClNamespace() +
                              "rounding_mode::rtn>"),
              LITERAL("0")))
      // __half2int_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2int_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2int_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2int_rn",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<int, " + MapNames::getClNamespace() +
                              "rounding_mode::rte>"),
              LITERAL("0")))
      // __half2int_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2int_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2int_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2int_ru",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<int, " + MapNames::getClNamespace() +
                              "rounding_mode::rtp>"),
              LITERAL("0")))
      // __half2int_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2int_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2int_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2int_rz",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<int, " + MapNames::getClNamespace() +
                              "rounding_mode::rtz>"),
              LITERAL("0")))
      // __half2ll_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ll_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ll_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ll_rd",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<long long, " + MapNames::getClNamespace() +
                              "rounding_mode::rtn>"),
              LITERAL("0")))
      // __half2ll_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ll_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ll_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ll_rn",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<long long, " + MapNames::getClNamespace() +
                              "rounding_mode::rte>"),
              LITERAL("0")))
      // __half2ll_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ll_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ll_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ll_ru",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<long long, " + MapNames::getClNamespace() +
                              "rounding_mode::rtp>"),
              LITERAL("0")))
      // __half2ll_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ll_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ll_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ll_rz",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<long long, " + MapNames::getClNamespace() +
                              "rounding_mode::rtz>"),
              LITERAL("0")))
      // __half2short_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2short_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2short_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2short_rd",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<short, " + MapNames::getClNamespace() +
                              "rounding_mode::rtn>"),
              LITERAL("0")))
      // __half2short_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2short_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2short_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2short_rn",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<short, " + MapNames::getClNamespace() +
                              "rounding_mode::rte>"),
              LITERAL("0")))
      // __half2short_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2short_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2short_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2short_ru",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<short, " + MapNames::getClNamespace() +
                              "rounding_mode::rtp>"),
              LITERAL("0")))
      // __half2short_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2short_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2short_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2short_rz",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<short, " + MapNames::getClNamespace() +
                              "rounding_mode::rtz>"),
              LITERAL("0")))
      // __half2uint_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2uint_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2uint_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2uint_rd",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned, " + MapNames::getClNamespace() +
                              "rounding_mode::rtn>"),
              LITERAL("0")))
      // __half2uint_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2uint_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2uint_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2uint_rn",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned, " + MapNames::getClNamespace() +
                              "rounding_mode::rte>"),
              LITERAL("0")))
      // __half2uint_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2uint_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2uint_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2uint_ru",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned, " + MapNames::getClNamespace() +
                              "rounding_mode::rtp>"),
              LITERAL("0")))
      // __half2uint_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2uint_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2uint_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2uint_rz",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned, " + MapNames::getClNamespace() +
                              "rounding_mode::rtz>"),
              LITERAL("0")))
      // __half2ull_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ull_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ull_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ull_rd",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned long long, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtn>"),
              LITERAL("0")))
      // __half2ull_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ull_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ull_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ull_rn",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned long long, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rte>"),
              LITERAL("0")))
      // __half2ull_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ull_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ull_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ull_ru",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned long long, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtp>"),
              LITERAL("0")))
      // __half2ull_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ull_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ull_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ull_rz",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned long long, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtz>"),
              LITERAL("0")))
      // __half2ushort_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ushort_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ushort_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ushort_rd",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned short, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtn>"),
              LITERAL("0")))
      // __half2ushort_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ushort_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ushort_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ushort_rn",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned short, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rte>"),
              LITERAL("0")))
      // __half2ushort_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ushort_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ushort_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ushort_ru",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned short, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtp>"),
              LITERAL("0")))
      // __half2ushort_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__half2ushort_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::half2ushort_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__half2ushort_rz",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<" +
                                   MapNames::getClNamespace() + "half, 1>",
                               ARG(0)),
                          false,
                          "convert<unsigned short, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtz>"),
              LITERAL("0")))
      // __half_as_short
      CALL_FACTORY_ENTRY("__half_as_short",
                         CALL(MapNames::getClNamespace() + "bit_cast<short, " +
                                  MapNames::getClNamespace() + "half>",
                              ARG(0)))
      // __half_as_ushort
      CALL_FACTORY_ENTRY("__half_as_ushort",
                         CALL(MapNames::getClNamespace() +
                                  "bit_cast<unsigned short, " +
                                  MapNames::getClNamespace() + "half>",
                              ARG(0)))
      // __halves2half2
      CALL_FACTORY_ENTRY(
          "__halves2half2",
          CALL(MapNames::getClNamespace() + "half2", ARG(0), ARG(1)))
      // __high2float
      ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY("__high2float", ARG(0), LITERAL("1"))
      // __high2half
      ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY("__high2half", ARG(0), LITERAL("1"))
      // __high2half2
      CALL_FACTORY_ENTRY("__high2half2",
                         CALL(MapNames::getClNamespace() + "half2",
                              ARRAY_SUBSCRIPT(ARG(0), LITERAL("1"))))
      // __highs2half2
      CALL_FACTORY_ENTRY("__highs2half2",
                         CALL(MapNames::getClNamespace() + "half2",
                              ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")),
                              ARRAY_SUBSCRIPT(ARG(1), LITERAL("1"))))
      // __int2half_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__int2half_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::int2half_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__int2half_rd",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<int, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtn>"),
              LITERAL("0")))
      // __int2half_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__int2half_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::int2half_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__int2half_rn",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<int, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rte>"),
              LITERAL("0")))
      // __int2half_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__int2half_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::int2half_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__int2half_ru",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<int, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtp>"),
              LITERAL("0")))
      // __int2half_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__int2half_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::int2half_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__int2half_rz",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<int, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtz>"),
              LITERAL("0")))
      // __funnelshift_l
      MATH_API_REWRITER_DEVICE(
          "__funnelshift_l",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__funnelshift_l"),
              EMPTY_FACTORY_ENTRY("__funnelshift_l"),
              EMPTY_FACTORY_ENTRY("__funnelshift_l"),
              WARNING_FACTORY_ENTRY(
                  "__funnelshift_l",
                  CALL_FACTORY_ENTRY(
                      "__funnelshift_l",
                      CALL(MapNames::getDpctNamespace() + "funnelshift_l",
                           ARG(0), ARG(1), ARG(2))),
                  Diagnostics::MATH_EMULATION, std::string("__funnelshift_l"),
                  MapNames::getDpctNamespace() + "funnelshift_l")))
      // __funnelshift_lc
      MATH_API_REWRITER_DEVICE(
          "__funnelshift_lc",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__funnelshift_lc"),
              EMPTY_FACTORY_ENTRY("__funnelshift_lc"),
              EMPTY_FACTORY_ENTRY("__funnelshift_lc"),
              WARNING_FACTORY_ENTRY(
                  "__funnelshift_lc",
                  CALL_FACTORY_ENTRY(
                      "__funnelshift_lc",
                      CALL(MapNames::getDpctNamespace() + "funnelshift_lc",
                           ARG(0), ARG(1), ARG(2))),
                  Diagnostics::MATH_EMULATION, std::string("__funnelshift_lc"),
                  MapNames::getDpctNamespace() + "funnelshift_lc")))
      // __funnelshift_r
      MATH_API_REWRITER_DEVICE(
          "__funnelshift_r",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__funnelshift_r"),
              EMPTY_FACTORY_ENTRY("__funnelshift_r"),
              EMPTY_FACTORY_ENTRY("__funnelshift_r"),
              WARNING_FACTORY_ENTRY(
                  "__funnelshift_r",
                  CALL_FACTORY_ENTRY(
                      "__funnelshift_r",
                      CALL(MapNames::getDpctNamespace() + "funnelshift_r",
                           ARG(0), ARG(1), ARG(2))),
                  Diagnostics::MATH_EMULATION, std::string("__funnelshift_r"),
                  MapNames::getDpctNamespace() + "funnelshift_r")))
      // __funnelshift_rc
      MATH_API_REWRITER_DEVICE(
          "__funnelshift_rc",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__funnelshift_rc"),
              EMPTY_FACTORY_ENTRY("__funnelshift_rc"),
              EMPTY_FACTORY_ENTRY("__funnelshift_rc"),
              WARNING_FACTORY_ENTRY(
                  "__funnelshift_rc",
                  CALL_FACTORY_ENTRY(
                      "__funnelshift_rc",
                      CALL(MapNames::getDpctNamespace() + "funnelshift_rc",
                           ARG(0), ARG(1), ARG(2))),
                  Diagnostics::MATH_EMULATION, std::string("__funnelshift_rc"),
                  MapNames::getDpctNamespace() + "funnelshift_rc")))
      // __ldca
      MATH_API_REWRITER_DEVICE(
          "__ldca", MATH_API_DEVICE_NODES(
                        EMPTY_FACTORY_ENTRY("__ldca"),
                        EMPTY_FACTORY_ENTRY("__ldca"),
                        EMPTY_FACTORY_ENTRY("__ldca"),
                        WARNING_FACTORY_ENTRY(
                            "__ldca", DEREF_FACTORY_ENTRY("__ldca", ARG(0)),
                            Diagnostics::MATH_EMULATION_EXPRESSION,
                            std::string("__ldca"), std::string("'*'"))))
      // __ldcg
      MATH_API_REWRITER_DEVICE(
          "__ldcg", MATH_API_DEVICE_NODES(
                        EMPTY_FACTORY_ENTRY("__ldcg"),
                        EMPTY_FACTORY_ENTRY("__ldcg"),
                        EMPTY_FACTORY_ENTRY("__ldcg"),
                        WARNING_FACTORY_ENTRY(
                            "__ldcg", DEREF_FACTORY_ENTRY("__ldcg", ARG(0)),
                            Diagnostics::MATH_EMULATION_EXPRESSION,
                            std::string("__ldcg"), std::string("'*'"))))
      // __ldcs
      MATH_API_REWRITER_DEVICE(
          "__ldcs", MATH_API_DEVICE_NODES(
                        EMPTY_FACTORY_ENTRY("__ldcs"),
                        EMPTY_FACTORY_ENTRY("__ldcs"),
                        EMPTY_FACTORY_ENTRY("__ldcs"),
                        WARNING_FACTORY_ENTRY(
                            "__ldcs", DEREF_FACTORY_ENTRY("__ldcs", ARG(0)),
                            Diagnostics::MATH_EMULATION_EXPRESSION,
                            std::string("__ldcs"), std::string("'*'"))))
      // __ldcv
      MATH_API_REWRITER_DEVICE(
          "__ldcv", MATH_API_DEVICE_NODES(
                        EMPTY_FACTORY_ENTRY("__ldcv"),
                        EMPTY_FACTORY_ENTRY("__ldcv"),
                        EMPTY_FACTORY_ENTRY("__ldcv"),
                        WARNING_FACTORY_ENTRY(
                            "__ldcv", DEREF_FACTORY_ENTRY("__ldcv", ARG(0)),
                            Diagnostics::MATH_EMULATION_EXPRESSION,
                            std::string("__ldcv"), std::string("'*'"))))
      // __ldg
      MATH_API_REWRITER_DEVICE(
          "__ldg", MATH_API_DEVICE_NODES(
                       EMPTY_FACTORY_ENTRY("__ldg"),
                       EMPTY_FACTORY_ENTRY("__ldg"),
                       EMPTY_FACTORY_ENTRY("__ldg"),
                       WARNING_FACTORY_ENTRY(
                           "__ldg", DEREF_FACTORY_ENTRY("__ldg", ARG(0)),
                           Diagnostics::MATH_EMULATION_EXPRESSION,
                           std::string("__ldg"), std::string("'*'"))))
      // __ldlu
      MATH_API_REWRITER_DEVICE(
          "__ldlu", MATH_API_DEVICE_NODES(
                        EMPTY_FACTORY_ENTRY("__ldlu"),
                        EMPTY_FACTORY_ENTRY("__ldlu"),
                        EMPTY_FACTORY_ENTRY("__ldlu"),
                        WARNING_FACTORY_ENTRY(
                            "__ldlu", DEREF_FACTORY_ENTRY("__ldlu", ARG(0)),
                            Diagnostics::MATH_EMULATION_EXPRESSION,
                            std::string("__ldlu"), std::string("'*'"))))
      // __ll2half_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ll2half_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ll2half_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ll2half_rd",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<long long, 1>",
                               ARG(0)),
                          false,
                          "convert<" + MapNames::getClNamespace() + "half, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtn>"),
              LITERAL("0")))
      // __ll2half_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ll2half_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ll2half_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ll2half_rn",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<long long, 1>",
                               ARG(0)),
                          false,
                          "convert<" + MapNames::getClNamespace() + "half, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rte>"),
              LITERAL("0")))
      // __ll2half_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ll2half_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ll2half_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ll2half_ru",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<long long, 1>",
                               ARG(0)),
                          false,
                          "convert<" + MapNames::getClNamespace() + "half, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtp>"),
              LITERAL("0")))
      // __ll2half_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ll2half_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ll2half_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ll2half_rz",
              MEMBER_CALL(CALL(MapNames::getClNamespace() + "vec<long long, 1>",
                               ARG(0)),
                          false,
                          "convert<" + MapNames::getClNamespace() + "half, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtz>"),
              LITERAL("0")))
      // __low2float
      ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY("__low2float", ARG(0), LITERAL("0"))
      // __low2half
      ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY("__low2half", ARG(0), LITERAL("0"))
      // __low2half2
      CALL_FACTORY_ENTRY("__low2half2",
                         CALL(MapNames::getClNamespace() + "half2",
                              ARRAY_SUBSCRIPT(ARG(0), LITERAL("0"))))
      // __lowhigh2highlow
      CALL_FACTORY_ENTRY("__lowhigh2highlow",
                         CALL(MapNames::getClNamespace() + "half2",
                              ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")),
                              ARRAY_SUBSCRIPT(ARG(0), LITERAL("0"))))
      // __lows2half2
      CALL_FACTORY_ENTRY("__lows2half2",
                         CALL(MapNames::getClNamespace() + "half2",
                              ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")),
                              ARRAY_SUBSCRIPT(ARG(1), LITERAL("0"))))
      // __short2half_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__short2half_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::short2half_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__short2half_rd",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<short, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtn>"),
              LITERAL("0")))
      // __short2half_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__short2half_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::short2half_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__short2half_rn",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<short, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rte>"),
              LITERAL("0")))
      // __short2half_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__short2half_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::short2half_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__short2half_ru",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<short, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtp>"),
              LITERAL("0")))
      // __short2half_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__short2half_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::short2half_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__short2half_rz",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<short, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtz>"),
              LITERAL("0")))
      // __short_as_half
      CALL_FACTORY_ENTRY("__short_as_half",
                         CALL(MapNames::getClNamespace() + "bit_cast<" +
                                  MapNames::getClNamespace() + "half, short>",
                              ARG(0)))
      // __stcg
      MATH_API_REWRITER_DEVICE(
          "__stcg",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__stcg"),
              EMPTY_FACTORY_ENTRY("__stcg"),
              EMPTY_FACTORY_ENTRY("__stcg"),
              WARNING_FACTORY_ENTRY(
                  "__stcg",
                  ASSIGN_FACTORY_ENTRY("__stcg", DEREF(ARG_WC(0)), ARG(1)),
                  Diagnostics::MATH_EMULATION_EXPRESSION, std::string("__stcg"),
                  std::string("'='"))))
      // __stcs
      MATH_API_REWRITER_DEVICE(
          "__stcs",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__stcs"),
              EMPTY_FACTORY_ENTRY("__stcs"),
              EMPTY_FACTORY_ENTRY("__stcs"),
              WARNING_FACTORY_ENTRY(
                  "__stcs",
                  ASSIGN_FACTORY_ENTRY("__stcs", DEREF(ARG_WC(0)), ARG(1)),
                  Diagnostics::MATH_EMULATION_EXPRESSION, std::string("__stcs"),
                  std::string("'='"))))
      // __stwb
      MATH_API_REWRITER_DEVICE(
          "__stwb",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__stwb"),
              EMPTY_FACTORY_ENTRY("__stwb"),
              EMPTY_FACTORY_ENTRY("__stwb"),
              WARNING_FACTORY_ENTRY(
                  "__stwb",
                  ASSIGN_FACTORY_ENTRY("__stwb", DEREF(ARG_WC(0)), ARG(1)),
                  Diagnostics::MATH_EMULATION_EXPRESSION, std::string("__stwb"),
                  std::string("'='"))))
      // __stwt
      MATH_API_REWRITER_DEVICE(
          "__stwt",
          MATH_API_DEVICE_NODES(
              EMPTY_FACTORY_ENTRY("__stwt"),
              EMPTY_FACTORY_ENTRY("__stwt"),
              EMPTY_FACTORY_ENTRY("__stwt"),
              WARNING_FACTORY_ENTRY(
                  "__stwt",
                  ASSIGN_FACTORY_ENTRY("__stwt", DEREF(ARG_WC(0)), ARG(1)),
                  Diagnostics::MATH_EMULATION_EXPRESSION, std::string("__stwt"),
                  std::string("'='"))))
      // __uint2half_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__uint2half_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::uint2half_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__uint2half_rd",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<unsigned, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtn>"),
              LITERAL("0")))
      // __uint2half_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__uint2half_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::uint2half_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__uint2half_rn",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<unsigned, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rte>"),
              LITERAL("0")))
      // __uint2half_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__uint2half_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::uint2half_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__uint2half_ru",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<unsigned, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtp>"),
              LITERAL("0")))
      // __uint2half_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__uint2half_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::uint2half_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__uint2half_rz",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<unsigned, 1>", ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtz>"),
              LITERAL("0")))
      // __ull2half_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ull2half_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ull2half_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ull2half_rd",
              MEMBER_CALL(CALL(MapNames::getClNamespace() +
                                   "vec<unsigned long long, 1>",
                               ARG(0)),
                          false,
                          "convert<" + MapNames::getClNamespace() + "half, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtn>"),
              LITERAL("0")))
      // __ull2half_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ull2half_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ull2half_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ull2half_rn",
              MEMBER_CALL(CALL(MapNames::getClNamespace() +
                                   "vec<unsigned long long, 1>",
                               ARG(0)),
                          false,
                          "convert<" + MapNames::getClNamespace() + "half, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rte>"),
              LITERAL("0")))
      // __ull2half_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ull2half_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ull2half_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ull2half_ru",
              MEMBER_CALL(CALL(MapNames::getClNamespace() +
                                   "vec<unsigned long long, 1>",
                               ARG(0)),
                          false,
                          "convert<" + MapNames::getClNamespace() + "half, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtp>"),
              LITERAL("0")))
      // __ull2half_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ull2half_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ull2half_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ull2half_rz",
              MEMBER_CALL(CALL(MapNames::getClNamespace() +
                                   "vec<unsigned long long, 1>",
                               ARG(0)),
                          false,
                          "convert<" + MapNames::getClNamespace() + "half, " +
                              MapNames::getClNamespace() +
                              "rounding_mode::rtz>"),
              LITERAL("0")))
      // __ushort2half_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ushort2half_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ushort2half_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ushort2half_rd",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<unsigned short, 1>",
                       ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtn>"),
              LITERAL("0")))
      // __ushort2half_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ushort2half_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ushort2half_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ushort2half_rn",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<unsigned short, 1>",
                       ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rte>"),
              LITERAL("0")))
      // __ushort2half_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ushort2half_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ushort2half_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ushort2half_ru",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<unsigned short, 1>",
                       ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtp>"),
              LITERAL("0")))
      // __ushort2half_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ushort2half_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ushort2half_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__ushort2half_rz",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<unsigned short, 1>",
                       ARG(0)),
                  false,
                  "convert<" + MapNames::getClNamespace() + "half, " +
                      MapNames::getClNamespace() + "rounding_mode::rtz>"),
              LITERAL("0")))
      // __ushort_as_half
      CALL_FACTORY_ENTRY("__ushort_as_half",
                         CALL(MapNames::getClNamespace() + "bit_cast<" +
                                  MapNames::getClNamespace() +
                                  "half, unsigned short>",
                              ARG(0)))
      // make_half2
      ENTRY_RENAMED("make_half2", MapNames::getClNamespace() + "half2")};
}
