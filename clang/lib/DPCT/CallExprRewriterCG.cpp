//===--------------- CallExprRewriterCG.cpp -------------------------------===//
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

void CallExprRewriterFactoryBase::initRewriterMapCooperativeGroups() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#define FUNCTION_CALL
#define CLASS_METHOD_CALL
#include "APINamesCooperativeGroups.inc"
#undef FUNCTION_CALL
#undef CLASS_METHOD_CALL
      }));
}

void CallExprRewriterFactoryBase::initMethodRewriterMap() {
  MethodRewriterMap = std::make_unique<std::unordered_map<
      std::string, std::shared_ptr<CallExprRewriterFactoryBase>>>(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#define CLASS_METHOD_CALL
#include "APINamesCooperativeGroups.inc"
#undef CLASS_METHOD_CALL
      }));
}

} // namespace dpct
} // namespace clang
