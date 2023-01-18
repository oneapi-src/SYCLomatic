//===--------------- CallExprRewriterLIBCU.cpp-----------------------------===//
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

#define BIND_TEXTURE_FACTORY_ENTRY(FuncName, ...)                              \
  {FuncName, createBindTextureRewriterFactory<__VA_ARGS__>(FuncName)},

#define REWRITER_FACTORY_ENTRY(FuncName, RewriterFactory, ...)                 \
  {FuncName, std::make_shared<RewriterFactory>(FuncName, __VA_ARGS__)},
#define FUNC_NAME_FACTORY_ENTRY(FuncName, RewriterName)                        \
  REWRITER_FACTORY_ENTRY(FuncName, FuncCallExprRewriterFactory, RewriterName)

#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)                            \
  FUNC_NAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)


void CallExprRewriterFactoryBase::initRewriterMapLIBCU() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesLIBCU.inc"
      }));
}

void CallExprRewriterFactoryBase::initMethodRewriterMapLIBCU() {
  MethodRewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesLIBCU.inc"
      }));
}

#undef ENTRY_RENAMED

} // namespace dpct
} // namespace clang
