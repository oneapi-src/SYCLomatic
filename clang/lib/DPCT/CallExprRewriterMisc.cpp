//===--------------- CallExprRewriterMisc.cpp -----------------------------===//
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

auto UseOccupancyCalculation = [](const CallExpr *C) -> bool {
  return DpctGlobalInfo::useOccupancyCalculation();
};

#define REWRITER_FACTORY_ENTRY(FuncName, RewriterFactory, ...)                 \
  {FuncName, std::make_shared<RewriterFactory>(FuncName, __VA_ARGS__)},
#define UNSUPPORTED_FACTORY_ENTRY(FuncName, MsgID)                             \
  REWRITER_FACTORY_ENTRY(FuncName,                                             \
                         UnsupportFunctionRewriterFactory<std::string>, MsgID, \
                         FuncName)
#define ENTRY_UNSUPPORTED(SOURCEAPINAME, MSGID)                                \
  UNSUPPORTED_FACTORY_ENTRY(SOURCEAPINAME, MSGID)

void CallExprRewriterFactoryBase::initRewriterMapMisc() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesMisc.inc"
      }));
}
#undef ENTRY_UNSUPPORTED
#undef UNSUPPORTED_FACTORY_ENTRY

} // namespace dpct
} // namespace clang
