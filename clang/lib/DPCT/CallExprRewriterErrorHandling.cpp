//===--------------- CallExprRewriterErrorHandling.cpp --------------------===//
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

#define REWRITER_FACTORY_ENTRY(FuncName, RewriterFactory, ...)                 \
  {FuncName, std::make_shared<RewriterFactory>(FuncName, __VA_ARGS__)},

using NoRewriteFuncNameRewriterFactory =
    clang::dpct::CallExprRewriterFactory<clang::dpct::NoRewriteFuncNameRewriter,
                                         std::string>;
#define FUNC_NAME_REWRITER_FACTORY_ENTRY(FuncName, RewriterName)               \
  REWRITER_FACTORY_ENTRY(FuncName, NoRewriteFuncNameRewriterFactory,           \
                         RewriterName)
#define NO_REWRITER_FUNC_NAME_REWRITER_FACTORY_ENTRY(SOURCEAPINAME,            \
                                                     TARGETAPINAME)            \
  FUNC_NAME_REWRITER_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)

void CallExprRewriterFactoryBase::initRewriterMapErrorHandling() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesErrorHandling.inc"
      }));
}

} // namespace dpct
} // namespace clang
