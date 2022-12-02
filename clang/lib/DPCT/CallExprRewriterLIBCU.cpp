//===--------------- CallExprRewriterLIBCU.cpp -----------------------------===//
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

// clang-format off
void CallExprRewriterFactoryBase::initRewriterMapLIBCU() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
CALL_FACTORY_ENTRY("cuda::std::atomic_thread_fence",
                   CALL(MapNames::getClNamespace() + "atomic_fence", ARG(0),
                        LITERAL(MapNames::getClNamespace() +
                                "memory_scope::system")))

CALL_FACTORY_ENTRY("cuda::atomic_thread_fence",
                   CALL(MapNames::getClNamespace() + "atomic_fence", ARG(0),
                        ARG(1)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Atomic_atomic_class_store,
                        MEMBER_CALL_FACTORY_ENTRY("store", ARG(0), false,
                                                  "store", ARG(0), ARG(1)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Atomic_atomic_class_load,
                        MEMBER_CALL_FACTORY_ENTRY("load", ARG(0), false, "load",
                                                  ARG(0)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Atomic_atomic_class_exchange,
                        MEMBER_CALL_FACTORY_ENTRY("exchange", ARG(0), false,
                                                  "exchange", ARG(0), ARG(1)))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Atomic_atomic_class_compare_exchange_weak,
    MEMBER_CALL_FACTORY_ENTRY("compare_exchange_weak", ARG(0), false,
                              "compare_exchange_weak", ARG(0), ARG(1), ARG(2),
                              ARG(3), ARG(4)))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Atomic_atomic_class_compare_exchange_strong,
    MEMBER_CALL_FACTORY_ENTRY("compare_exchange_strong", ARG(0), false,
                              "compare_exchange_strong", ARG(0), ARG(1), ARG(2),
                              ARG(3), ARG(4)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Atomic_atomic_class_fetch_add,
                        MEMBER_CALL_FACTORY_ENTRY("fetch_add", ARG(0), false,
                                                  "fetch_add", ARG(0), ARG(1)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Atomic_atomic_class_fetch_sub,
                        MEMBER_CALL_FACTORY_ENTRY("fetch_sub", ARG(0), false,
                                                  "fetch_sub", ARG(0), ARG(1)))
      }));
}
// clang-format on

} // namespace dpct
} // namespace clang
