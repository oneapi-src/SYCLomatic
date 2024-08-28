//===--------------- RewriterSYCLcompat.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../CallExprRewriter.h"

namespace clang {
namespace dpct {

void initRewriterMapSYCLcompatUnsupport(
    std::unordered_map<std::string,
                       std::shared_ptr<CallExprRewriterFactoryBase>>
        &RewriterMap) {

#define ARG(x) makeCallArgCreator(x)
#define UNSUPPORT_FACTORY_ENTRY(FuncName, ...)                                \
  std::make_pair(                                                              \
      FuncName, createUnsupportRewriterFactory(FuncName, __VA_ARGS__)),
#define SYCLCOMPAT_UNSUPPORT(NAME)                                        \
  UNSUPPORT_FACTORY_ENTRY(NAME, Diagnostics::UNSUPPORT_SYCLCOMPAT, ARG(NAME))
#define ENTRY_UNSUPPORTED(...) UNSUPPORT_FACTORY_ENTRY(__VA_ARGS__)
#define CONDITIONAL_FACTORY_ENTRY(COND, A, B) A
#define FEATURE_REQUEST_FACTORY(FEATURE, SUB) SUB
#define ASSIGNABLE_FACTORY(SUB) SUB
#define ENTRY_RENAMED(NAME, ...) SYCLCOMPAT_UNSUPPORT(NAME)
#define MEMBER_CALL_FACTORY_ENTRY(NAME, ...) SYCLCOMPAT_UNSUPPORT(NAME)
#define CALL_FACTORY_ENTRY(NAME, ...) SYCLCOMPAT_UNSUPPORT(NAME)
#define ASSIGN_FACTORY_ENTRY(NAME, ...) SYCLCOMPAT_UNSUPPORT(NAME)
#define ENTRY_TEXTURE(NAME, ...) SYCLCOMPAT_UNSUPPORT(NAME)
#define ENTRY_BIND(NAME, ...) SYCLCOMPAT_UNSUPPORT(NAME)
#define DELETE_FACTORY_ENTRY(NAME, ...) SYCLCOMPAT_UNSUPPORT(NAME)
#define DELETER_FACTORY_ENTRY(NAME, ...) SYCLCOMPAT_UNSUPPORT(NAME)
#define MULTI_STMTS_FACTORY_ENTRY(NAME, ...) SYCLCOMPAT_UNSUPPORT(NAME)
#define WARNING_FACTORY_ENTRY(NAME, ...) SYCLCOMPAT_UNSUPPORT(NAME)

RewriterMap.insert({
#include "../APINamesGraph.inc"
#include "../APINamesTexture.inc"
        });
}

void CallExprRewriterFactoryBase::initRewriterMapSYCLcompat(
    std::unordered_map<std::string,
                       std::shared_ptr<CallExprRewriterFactoryBase>>
        &RewriterMap) {
  initRewriterMapSYCLcompatUnsupport(RewriterMap);
}

} // namespace dpct
} // namespace clang