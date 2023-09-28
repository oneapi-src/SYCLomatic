//===--------------- RewriterUtilityFunctions.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createUtilityFunctionsRewriterMap() {
  return RewriterMap{
      // cub::IADD3
      CALL_FACTORY_ENTRY(
          "cub::IADD3",
          CALL("", BO(BinaryOperatorKind::BO_Add,
                      BO(BinaryOperatorKind::BO_Add,
                         CAST_IF_NOT_SAME(LITERAL("unsigned int"), ARG(0)),
                         CAST_IF_NOT_SAME(LITERAL("unsigned int"), ARG(1))),
                      CAST_IF_NOT_SAME(LITERAL("unsigned int"), ARG(2)))))

      // cub::LaneId
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::LaneId", MEMBER_CALL(NDITEM, false, LITERAL("get_sub_group")),
          false, "get_local_linear_id")

      // cub::WarpId
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::WarpId", MEMBER_CALL(NDITEM, false, LITERAL("get_sub_group")),
          false, "get_group_linear_id")

      // cub::SyncStream
      CONDITIONAL_FACTORY_ENTRY(
          CheckArgIsDefaultCudaStream(0),
          MEMBER_CALL_FACTORY_ENTRY("cub::SyncStream", QUEUESTR, false, "wait"),
          MEMBER_CALL_FACTORY_ENTRY("cub::SyncStream", ARG(0), true, "wait"))
      // cub::DeviceCount
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::DeviceCount",
          CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"), false,
          "device_count")
      // cub::DeviceCountUncached
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::DeviceCountUncached",
          CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"), false,
          "device_count")
      // cub::DeviceCountCachedValue
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::DeviceCountCachedValue",
          CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"), false,
          "device_count")
      // cub::CurrentDevice
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::CurrentDevice",
          CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"), false,
          "current_device_id")

  };
}
