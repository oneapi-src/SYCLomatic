//===--------------- CallExprRewriterDriver.cpp ---------------------------===//
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

template <class T>
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createContextFactory(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        NoQueueDeviceFactory,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        LegacyFactory,
    T) {
  assert(NoQueueDeviceFactory.first == LegacyFactory.first);
  if (DpctGlobalInfo::useNoQueueDevice()) {
    return std::make_pair(
        LegacyFactory.first,
        createReportWarningRewriterFactory(
            createAssignableFactory(std::move(NoQueueDeviceFactory)),
            LegacyFactory.first, Diagnostics::FUNC_CALL_REMOVED,
            LegacyFactory.first,
            std::string(
                "it is redundant if it is migrated with option "
                "--helper-function-preference=no-queue-device which declares a "
                "global SYCL device and queue.")));
  }
  if (DpctGlobalInfo::useSYCLCompat()) {
    return std::make_pair(
        LegacyFactory.first,
        createUnsupportRewriterFactory(std::string(LegacyFactory.first),
                                       Diagnostics::UNSUPPORT_SYCLCOMPAT,
                                       std::string(LegacyFactory.first)));
  }
  return createFeatureRequestFactory(
      HelperFeatureEnum::device_ext,
      createAssignableFactory(std::move(LegacyFactory)));
}

#define CONTEXT_ENTRY(NOQUEUEDEVICE, LEGACY)                                   \
  createContextFactory(NOQUEUEDEVICE LEGACY 0),

void CallExprRewriterFactoryBase::initRewriterMapDriver() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesDriver.inc"
      }));
}

} // namespace dpct
} // namespace clang
