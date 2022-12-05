//===--------------- CallExprRewriterNccl.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"

namespace clang {
namespace dpct {
void CallExprRewriterFactoryBase::initRewriterMapNccl() {
  RewriterMap->merge(
  std::unordered_map<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
  {

  FEATURE_REQUEST_FACTORY(
  HelperFeatureEnum::CclUtils_get_version,
  ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
  "ncclGetVersion", DEREF(0),
  CALL(MapNames::getDpctNamespace() + "ccl::get_version"))))

  FEATURE_REQUEST_FACTORY(
  HelperFeatureEnum::CclUtils_create_kvs_address,
  ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
  "ncclGetUniqueId", DEREF(0),
  CALL(MapNames::getDpctNamespace() + "ccl::create_kvs_address"))))

  FEATURE_REQUEST_FACTORY(
  HelperFeatureEnum::CclUtils_create_kvs,
  ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
  "ncclCommInitRank", DEREF(0),
  NEW(
  "oneapi::ccl::communicator",
  CALL("oneapi::ccl::create_communicator", ARG(1), ARG(3),
       CALL(MapNames::getDpctNamespace() + "ccl::create_kvs", ARG(2)))))))}));
}

} // namespace dpct
} // namespace clang
