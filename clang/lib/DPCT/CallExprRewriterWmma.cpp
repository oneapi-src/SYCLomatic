//===--------------- CallExprRewriterWmma.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTTraversal.h"
#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"
#include "MapNames.h"

namespace clang {
namespace dpct {

std::function<bool(const CallExpr *)> checkEnableJointMatrix() {
  return [=](const CallExpr *) -> bool {
    return DpctGlobalInfo::useExtJointMatrix();
  };
}

void CallExprRewriterFactoryBase::initRewriterMapWmma() {
  // Load this migration mapping on-demand based on options at runtime.
  // Using SYCL experimental type as the the mapping of
  // nvcuda::wmma::mem_row_major, this mapping must be protected by option
  // "--use-experimental-features=matrix".
  if (DpctGlobalInfo::useExtJointMatrix() && !DpctGlobalInfo::useSYCLCompat()) {
    EnumConstantRule::EnumNamesMap.insert(
        {"nvcuda::wmma::mem_row_major",
         std::make_shared<EnumNameRule>(
             MapNames::getClNamespace() +
             "ext::oneapi::experimental::matrix::layout::row_major")});
    EnumConstantRule::EnumNamesMap.insert(
        {"nvcuda::wmma::mem_col_major",
         std::make_shared<EnumNameRule>(
             MapNames::getClNamespace() +
             "ext::oneapi::experimental::matrix::layout::col_major")});
  }
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#define REWRITER_FACTORY_ENTRY(FuncName, RewriterFactory, ...)                 \
  {FuncName, std::make_shared<RewriterFactory>(FuncName, __VA_ARGS__)},
#define UNSUPPORTED_FACTORY_ENTRY(FuncName, MsgID)                             \
  REWRITER_FACTORY_ENTRY(FuncName,                                             \
                         UnsupportFunctionRewriterFactory<std::string>, MsgID, \
                         FuncName)
#define ENTRY_UNSUPPORTED(SOURCEAPINAME, MSGID)                                \
  UNSUPPORTED_FACTORY_ENTRY(SOURCEAPINAME, MSGID)
#include "APINamesWmma.inc"
#undef ENTRY_UNSUPPORTED
#undef UNSUPPORTED_FACTORY_ENTRY

      }));
}

} // namespace dpct
} // namespace clang
