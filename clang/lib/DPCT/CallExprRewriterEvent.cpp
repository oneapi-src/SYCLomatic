//===--------------- CallExprRewriterEvent.cpp ----------------------------===//
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
void CallExprRewriterFactoryBase::initRewriterMapEvent() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_typedef_event_ptr,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cudaEventCreate",
                            DEREF(makeDerefArgCreatorWithCall(0)),
                            NEW("sycl::event"))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_typedef_event_ptr,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cuEventCreate",
                            DEREF(makeDerefArgCreatorWithCall(0)),
                            NEW("sycl::event"))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_typedef_event_ptr,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cudaEventCreateWithFlags",
                            DEREF(makeDerefArgCreatorWithCall(0)),
                            NEW("sycl::event"))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_destroy_event,
                        ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
                            "cudaEventDestroy",
                            CALL(MapNames::getDpctNamespace() + "destroy_event",
                                 ARG(0)))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_destroy_event,
                        ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
                            "cuEventDestroy_v2",
                            CALL(MapNames::getDpctNamespace() + "destroy_event",
                                 ARG(0)))))

      }));
}
// clang-format on

} // namespace dpct
} // namespace clang
