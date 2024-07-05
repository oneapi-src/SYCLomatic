//===--------------- CallExprRewriterGraph.cpp ----------------------------===//
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

void CallExprRewriterFactoryBase::initRewriterMapGraph() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesGraph.inc"
      }));
}

} // namespace dpct
} // namespace clang
