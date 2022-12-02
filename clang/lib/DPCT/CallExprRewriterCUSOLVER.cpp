//===--------------- CallExprRewriterCUSOLVER.cpp -------------------------===//
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
void CallExprRewriterFactoryBase::initRewriterMapCUSOLVER() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_sygvd,
    CALL_FACTORY_ENTRY("cusolverDnSsygvd",
                       CALL(MapNames::getDpctNamespace() + "lapack::sygvd",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                            ARG(12))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnSsygvd_bufferSize", DEREF(10),
    CALL("oneapi::mkl::lapack::sygvd_scratchpad_size<float>", DEREF(0), ARG(1),
         ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))
      }));
}
// clang-format on

} // namespace dpct
} // namespace clang
