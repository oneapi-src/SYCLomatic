//===----- RewriterBfloat16PrecisionConversionAndDataMovement.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"

using namespace clang::dpct;

RewriterMap dpct::createBfloat16PrecisionConversionAndDataMovementRewriterMap() {
  return RewriterMap{
      // __bfloat1622float2
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__bfloat1622float2",
                  CALL(MapNames::getClNamespace() + "float2",
                       CALL(MapNames::getClNamespace() +
                                "ext::intel::math::bfloat162float",
                            ARRAY_SUBSCRIPT(ARG(0), LITERAL("0"))),
                       CALL(MapNames::getClNamespace() +
                                "ext::intel::math::bfloat162float",
                            ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")))))),
          CALL_FACTORY_ENTRY("__bfloat1622float2",
                             CALL(MapNames::getClNamespace() + "float2",
                                  ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")),
                                  ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")))))
      // __bfloat162bfloat162
      CONDITIONAL_FACTORY_ENTRY(
          math::UseBFloat16,
          CALL_FACTORY_ENTRY("__bfloat162bfloat162",
                             CALL(MapNames::getClNamespace() + "marray<" +
                                      MapNames::getClNamespace() +
                                      "ext::oneapi::bfloat16, 2>",
                                  ARG(0), ARG(0))),
          UNSUPPORT_FACTORY_ENTRY("__bfloat162bfloat162",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__bfloat162bfloat162")))
      // __bfloat162float
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162float",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162float",
                                      ARG(0)))),
          CALL_FACTORY_ENTRY("__bfloat162float",
                             CALL("static_cast<float>", ARG(0))))
      // __bfloat162int_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162int_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162int_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162int_rd",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<int, " + MapNames::getClNamespace() +
                      "rounding_mode::rtn>"),
              LITERAL("0")))
      // __bfloat162int_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162int_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162int_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162int_rn",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<int, " + MapNames::getClNamespace() +
                      "rounding_mode::rte>"),
              LITERAL("0")))
      // __bfloat162int_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162int_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162int_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162int_ru",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<int, " + MapNames::getClNamespace() +
                      "rounding_mode::rtp>"),
              LITERAL("0")))
      // __bfloat162int_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162int_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162int_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162int_rz",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<int, " + MapNames::getClNamespace() +
                      "rounding_mode::rtz>"),
              LITERAL("0")))
      // __bfloat162ll_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162ll_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162ll_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ll_rd",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<long long, " + MapNames::getClNamespace() +
                      "rounding_mode::rtn>"),
              LITERAL("0")))
      // __bfloat162ll_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162ll_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162ll_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ll_rn",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<long long, " + MapNames::getClNamespace() +
                      "rounding_mode::rte>"),
              LITERAL("0")))
      // __bfloat162ll_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162ll_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162ll_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ll_ru",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<long long, " + MapNames::getClNamespace() +
                      "rounding_mode::rtp>"),
              LITERAL("0")))
      // __bfloat162ll_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162ll_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162ll_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ll_rz",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<long long, " + MapNames::getClNamespace() +
                      "rounding_mode::rtz>"),
              LITERAL("0")))
      // __bfloat162short_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162short_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162short_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162short_rd",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<short, " + MapNames::getClNamespace() +
                      "rounding_mode::rtn>"),
              LITERAL("0")))
      // __bfloat162short_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162short_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162short_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162short_rn",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<short, " + MapNames::getClNamespace() +
                      "rounding_mode::rte>"),
              LITERAL("0")))
      // __bfloat162short_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162short_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162short_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162short_ru",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<short, " + MapNames::getClNamespace() +
                      "rounding_mode::rtp>"),
              LITERAL("0")))
      // __bfloat162short_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162short_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162short_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162short_rz",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<short, " + MapNames::getClNamespace() +
                      "rounding_mode::rtz>"),
              LITERAL("0")))
      // __bfloat162uint_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162uint_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162uint_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162uint_rd",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned, " + MapNames::getClNamespace() +
                      "rounding_mode::rtn>"),
              LITERAL("0")))
      // __bfloat162uint_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162uint_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162uint_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162uint_rn",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned, " + MapNames::getClNamespace() +
                      "rounding_mode::rte>"),
              LITERAL("0")))
      // __bfloat162uint_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162uint_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162uint_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162uint_ru",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned, " + MapNames::getClNamespace() +
                      "rounding_mode::rtp>"),
              LITERAL("0")))
      // __bfloat162uint_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162uint_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162uint_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162uint_rz",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned, " + MapNames::getClNamespace() +
                      "rounding_mode::rtz>"),
              LITERAL("0")))
      // __bfloat162ull_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162ull_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162ull_rd",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ull_rd",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned long long, " + MapNames::getClNamespace() +
                      "rounding_mode::rtn>"),
              LITERAL("0")))
      // __bfloat162ull_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162ull_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162ull_rn",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ull_rn",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned long long, " + MapNames::getClNamespace() +
                      "rounding_mode::rte>"),
              LITERAL("0")))
      // __bfloat162ull_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162ull_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162ull_ru",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ull_ru",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned long long, " + MapNames::getClNamespace() +
                      "rounding_mode::rtp>"),
              LITERAL("0")))
      // __bfloat162ull_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat162ull_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat162ull_rz",
                                      ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ull_rz",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned long long, " + MapNames::getClNamespace() +
                      "rounding_mode::rtz>"),
              LITERAL("0")))
      // __bfloat162ushort_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__bfloat162ushort_rd",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::bfloat162ushort_rd",
                       ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ushort_rd",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned short, " + MapNames::getClNamespace() +
                      "rounding_mode::rtn>"),
              LITERAL("0")))
      // __bfloat162ushort_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__bfloat162ushort_rn",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::bfloat162ushort_rn",
                       ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ushort_rn",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned short, " + MapNames::getClNamespace() +
                      "rounding_mode::rte>"),
              LITERAL("0")))
      // __bfloat162ushort_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__bfloat162ushort_ru",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::bfloat162ushort_ru",
                       ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ushort_ru",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned short, " + MapNames::getClNamespace() +
                      "rounding_mode::rtp>"),
              LITERAL("0")))
      // __bfloat162ushort_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__bfloat162ushort_rz",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::bfloat162ushort_rz",
                       ARG(0)))),
          ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(
              "__bfloat162ushort_rz",
              MEMBER_CALL(
                  CALL(MapNames::getClNamespace() + "vec<float, 1>", ARG(0)),
                  false,
                  "convert<unsigned short, " + MapNames::getClNamespace() +
                      "rounding_mode::rtz>"),
              LITERAL("0")))
      // __bfloat16_as_short
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__bfloat16_as_short",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::bfloat16_as_short",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__bfloat16_as_short",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__bfloat16_as_short")))
      // __bfloat16_as_ushort
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__bfloat16_as_ushort",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::bfloat16_as_ushort",
                       ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__bfloat16_as_ushort",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__bfloat16_as_ushort")))
      // __double2bfloat16
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__double2bfloat16",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::double2bfloat16",
                                      ARG(0)))),
          CONDITIONAL_FACTORY_ENTRY(
              math::UseBFloat16,
              CALL_FACTORY_ENTRY(
                  "__double2bfloat16",
                  CALL(MapNames::getClNamespace() + "ext::oneapi::bfloat16",
                       ARG(0))),
              UNSUPPORT_FACTORY_ENTRY("__double2bfloat16",
                                      Diagnostics::API_NOT_MIGRATED,
                                      ARG("__double2bfloat16"))))
      // __float22bfloat162_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseBFloat16,
          CALL_FACTORY_ENTRY("__float22bfloat162_rn",
                             CALL(MapNames::getClNamespace() + "marray<" +
                                      MapNames::getClNamespace() +
                                      "ext::oneapi::bfloat16, 2>",
                                  ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")),
                                  ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")))),
          UNSUPPORT_FACTORY_ENTRY("__float22bfloat162_rn",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__float22bfloat162_rn")))
      // __float2bfloat16
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__float2bfloat16",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::float2bfloat16",
                                      ARG(0)))),
          CONDITIONAL_FACTORY_ENTRY(
              math::UseBFloat16,
              CALL_FACTORY_ENTRY(
                  "__float2bfloat16",
                  CALL(MapNames::getClNamespace() + "ext::oneapi::bfloat16",
                       ARG(0))),
              UNSUPPORT_FACTORY_ENTRY("__float2bfloat16",
                                      Diagnostics::API_NOT_MIGRATED,
                                      ARG("__float2bfloat16"))))
      // __float2bfloat162_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseBFloat16,
          CALL_FACTORY_ENTRY("__float2bfloat162_rn",
                             CALL(MapNames::getClNamespace() + "marray<" +
                                      MapNames::getClNamespace() +
                                      "ext::oneapi::bfloat16, 2>",
                                  ARG(0), ARG(0))),
          UNSUPPORT_FACTORY_ENTRY("__float2bfloat162_rn",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__float2bfloat162_rn")))
      // __float2bfloat16_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__float2bfloat16_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::float2bfloat16_rd",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__float2bfloat16_rd",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__float2bfloat16_rd")))
      // __float2bfloat16_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__float2bfloat16_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::float2bfloat16_rn",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__float2bfloat16_rn",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__float2bfloat16_rn")))
      // __float2bfloat16_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__float2bfloat16_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::float2bfloat16_ru",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__float2bfloat16_ru",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__float2bfloat16_ru")))
      // __float2bfloat16_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__float2bfloat16_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::float2bfloat16_rz",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__float2bfloat16_rz",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__float2bfloat16_rz")))
      // __floats2bfloat162_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseBFloat16,
          CALL_FACTORY_ENTRY("__floats2bfloat162_rn",
                             CALL(MapNames::getClNamespace() + "marray<" +
                                      MapNames::getClNamespace() +
                                      "ext::oneapi::bfloat16, 2>",
                                  ARG(0), ARG(1))),
          UNSUPPORT_FACTORY_ENTRY("__floats2bfloat162_rn",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__floats2bfloat162_rn")))
      // __halves2bfloat162
      CONDITIONAL_FACTORY_ENTRY(
          math::UseBFloat16,
          CALL_FACTORY_ENTRY("__halves2bfloat162",
                             CALL(MapNames::getClNamespace() + "marray<" +
                                      MapNames::getClNamespace() +
                                      "ext::oneapi::bfloat16, 2>",
                                  ARG(0), ARG(1))),
          UNSUPPORT_FACTORY_ENTRY("__halves2bfloat162",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__halves2bfloat162")))
      // __high2bfloat16
      CONDITIONAL_FACTORY_ENTRY(
          math::UseBFloat16,
          CALL_FACTORY_ENTRY(
              "__high2bfloat16",
              CALL(MapNames::getClNamespace() + "ext::oneapi::bfloat16",
                   ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")))),
          UNSUPPORT_FACTORY_ENTRY("__high2bfloat16",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__high2bfloat16")))
      // __high2bfloat162
      CONDITIONAL_FACTORY_ENTRY(
          math::UseBFloat16,
          CALL_FACTORY_ENTRY("__high2bfloat162",
                             CALL(MapNames::getClNamespace() + "marray<" +
                                      MapNames::getClNamespace() +
                                      "ext::oneapi::bfloat16, 2>",
                                  ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")),
                                  ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")))),
          UNSUPPORT_FACTORY_ENTRY("__high2bfloat162",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__high2bfloat162")))
      // __highs2bfloat162
      CONDITIONAL_FACTORY_ENTRY(
          math::UseBFloat16,
          CALL_FACTORY_ENTRY("__highs2bfloat162",
                             CALL(MapNames::getClNamespace() + "marray<" +
                                      MapNames::getClNamespace() +
                                      "ext::oneapi::bfloat16, 2>",
                                  ARRAY_SUBSCRIPT(ARG(0), LITERAL("1")),
                                  ARRAY_SUBSCRIPT(ARG(1), LITERAL("1")))),
          UNSUPPORT_FACTORY_ENTRY("__highs2bfloat162",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__highs2bfloat162")))
      // __int2bfloat16_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__int2bfloat16_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::int2bfloat16_rd",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__int2bfloat16_rd",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__int2bfloat16_rd")))
      // __int2bfloat16_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__int2bfloat16_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::int2bfloat16_rn",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__int2bfloat16_rn",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__int2bfloat16_rn")))
      // __int2bfloat16_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__int2bfloat16_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::int2bfloat16_ru",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__int2bfloat16_ru",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__int2bfloat16_ru")))
      // __int2bfloat16_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__int2bfloat16_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::int2bfloat16_rz",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__int2bfloat16_rz",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__int2bfloat16_rz")))
      // __ll2bfloat16_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ll2bfloat16_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ll2bfloat16_rd",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ll2bfloat16_rd",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ll2bfloat16_rd")))
      // __ll2bfloat16_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ll2bfloat16_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ll2bfloat16_rn",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ll2bfloat16_rn",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ll2bfloat16_rn")))
      // __ll2bfloat16_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ll2bfloat16_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ll2bfloat16_ru",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ll2bfloat16_ru",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ll2bfloat16_ru")))
      // __ll2bfloat16_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ll2bfloat16_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ll2bfloat16_rz",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ll2bfloat16_rz",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ll2bfloat16_rz")))
      // __low2bfloat16
      CONDITIONAL_FACTORY_ENTRY(
          math::UseBFloat16,
          CALL_FACTORY_ENTRY(
              "__low2bfloat16",
              CALL(MapNames::getClNamespace() + "ext::oneapi::bfloat16",
                   ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
          UNSUPPORT_FACTORY_ENTRY("__low2bfloat16",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__low2bfloat16")))
      // __low2bfloat162
      CONDITIONAL_FACTORY_ENTRY(
          math::UseBFloat16,
          CALL_FACTORY_ENTRY("__low2bfloat162",
                             CALL(MapNames::getClNamespace() + "marray<" +
                                      MapNames::getClNamespace() +
                                      "ext::oneapi::bfloat16, 2>",
                                  ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")),
                                  ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")))),
          UNSUPPORT_FACTORY_ENTRY("__low2bfloat162",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__low2bfloat162")))
      // __lows2bfloat162
      CONDITIONAL_FACTORY_ENTRY(
          math::UseBFloat16,
          CALL_FACTORY_ENTRY("__lows2bfloat162",
                             CALL(MapNames::getClNamespace() + "marray<" +
                                      MapNames::getClNamespace() +
                                      "ext::oneapi::bfloat16, 2>",
                                  ARRAY_SUBSCRIPT(ARG(0), LITERAL("0")),
                                  ARRAY_SUBSCRIPT(ARG(1), LITERAL("0")))),
          UNSUPPORT_FACTORY_ENTRY("__lows2bfloat162",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__lows2bfloat162")))
      // __short2bfloat16_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__short2bfloat16_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::short2bfloat16_rd",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__short2bfloat16_rd",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__short2bfloat16_rd")))
      // __short2bfloat16_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__short2bfloat16_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::short2bfloat16_rn",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__short2bfloat16_rn",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__short2bfloat16_rn")))
      // __short2bfloat16_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__short2bfloat16_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::short2bfloat16_ru",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__short2bfloat16_ru",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__short2bfloat16_ru")))
      // __short2bfloat16_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__short2bfloat16_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::short2bfloat16_rz",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__short2bfloat16_rz",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__short2bfloat16_rz")))
      // __short_as_bfloat16
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__short_as_bfloat16",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::short_as_bfloat16",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__short_as_bfloat16",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__short_as_bfloat16")))
      // __uint2bfloat16_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__uint2bfloat16_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::uint2bfloat16_rd",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__uint2bfloat16_rd",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__uint2bfloat16_rd")))
      // __uint2bfloat16_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__uint2bfloat16_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::uint2bfloat16_rn",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__uint2bfloat16_rn",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__uint2bfloat16_rn")))
      // __uint2bfloat16_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__uint2bfloat16_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::uint2bfloat16_ru",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__uint2bfloat16_ru",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__uint2bfloat16_ru")))
      // __uint2bfloat16_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__uint2bfloat16_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::uint2bfloat16_rz",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__uint2bfloat16_rz",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__uint2bfloat16_rz")))
      // __ull2bfloat16_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ull2bfloat16_rd",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ull2bfloat16_rd",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ull2bfloat16_rd",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ull2bfloat16_rd")))
      // __ull2bfloat16_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ull2bfloat16_rn",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ull2bfloat16_rn",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ull2bfloat16_rn",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ull2bfloat16_rn")))
      // __ull2bfloat16_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ull2bfloat16_ru",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ull2bfloat16_ru",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ull2bfloat16_ru",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ull2bfloat16_ru")))
      // __ull2bfloat16_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY("__ull2bfloat16_rz",
                                 CALL(MapNames::getClNamespace() +
                                          "ext::intel::math::ull2bfloat16_rz",
                                      ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ull2bfloat16_rz",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ull2bfloat16_rz")))
      // __ushort2bfloat16_rd
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__ushort2bfloat16_rd",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::ushort2bfloat16_rd",
                       ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ushort2bfloat16_rd",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ushort2bfloat16_rd")))
      // __ushort2bfloat16_rn
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__ushort2bfloat16_rn",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::ushort2bfloat16_rn",
                       ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ushort2bfloat16_rn",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ushort2bfloat16_rn")))
      // __ushort2bfloat16_ru
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__ushort2bfloat16_ru",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::ushort2bfloat16_ru",
                       ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ushort2bfloat16_ru",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ushort2bfloat16_ru")))
      // __ushort2bfloat16_rz
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__ushort2bfloat16_rz",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::ushort2bfloat16_rz",
                       ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ushort2bfloat16_rz",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ushort2bfloat16_rz")))
      // __ushort_as_bfloat16
      CONDITIONAL_FACTORY_ENTRY(
          math::UseIntelDeviceMath,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_SYCL_Math,
              CALL_FACTORY_ENTRY(
                  "__ushort_as_bfloat16",
                  CALL(MapNames::getClNamespace() +
                           "ext::intel::math::ushort_as_bfloat16",
                       ARG(0)))),
          UNSUPPORT_FACTORY_ENTRY("__ushort_as_bfloat16",
                                  Diagnostics::API_NOT_MIGRATED,
                                  ARG("__ushort_as_bfloat16")))};
}
