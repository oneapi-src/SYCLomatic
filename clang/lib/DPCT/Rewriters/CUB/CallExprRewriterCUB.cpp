//===--------------- CallExprRewriterCUB.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

namespace clang {
namespace dpct {
void CallExprRewriterFactoryBase::initRewriterMapCUB() {
  RewriterMap->merge(createDeviceReduceRewriterMap());
  RewriterMap->merge(createDeviceScanRewriterMap());
  RewriterMap->merge(createDeviceSelectRewriterMap());
  RewriterMap->merge(createDeviceRunLengthEncodeRewriterMap());
  RewriterMap->merge(createDeviceSegmentedReduceRewriterMap());
  RewriterMap->merge(createDeviceRadixSortRewriterMap());
  RewriterMap->merge(createDeviceSegmentedRadixSortRewriterMap());
  RewriterMap->merge(createDeviceSegmentedSortRewriterMap());
  RewriterMap->merge(createDeviceHistgramRewriterMap());
  RewriterMap->merge(createUtilityFunctionsRewriterMap());
}

void CallExprRewriterFactoryBase::initMethodRewriterMapCUB() {
  MethodRewriterMap->merge(createClassMethodsRewriterMap());
}

} // namespace dpct
} // namespace clang
