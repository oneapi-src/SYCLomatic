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
SYCLCOMPAT_UNSUPPORT("cudaMemcpy2DArrayToArray")
SYCLCOMPAT_UNSUPPORT("cudaMemcpy2DFromArray")
SYCLCOMPAT_UNSUPPORT("cudaMemcpy2DFromArrayAsync")
SYCLCOMPAT_UNSUPPORT("cudaMemcpy2DToArray")
SYCLCOMPAT_UNSUPPORT("cudaMemcpy2DToArrayAsync")
SYCLCOMPAT_UNSUPPORT("cudaMemcpyArrayToArray")
SYCLCOMPAT_UNSUPPORT("cudaMemcpyToArray")
SYCLCOMPAT_UNSUPPORT("cudaMemcpyToArrayAsync")
SYCLCOMPAT_UNSUPPORT("cudaMemcpyFromArray")
SYCLCOMPAT_UNSUPPORT("cudaMemcpyFromArrayAsync")
SYCLCOMPAT_UNSUPPORT("cuMemcpyAtoH_v2")
SYCLCOMPAT_UNSUPPORT("cuMemcpyHtoA_v2")
SYCLCOMPAT_UNSUPPORT("cuMemcpyAtoHAsync_v2")
SYCLCOMPAT_UNSUPPORT("cuMemcpyHtoAAsync_v2")
SYCLCOMPAT_UNSUPPORT("cuMemcpyAtoD_v2")
SYCLCOMPAT_UNSUPPORT("cuMemcpyDtoA_v2")
SYCLCOMPAT_UNSUPPORT("cuMemcpyAtoA_v2")
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