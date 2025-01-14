//===--------------- RewriterSYCLcompat.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RuleInfra/CallExprRewriter.h"

namespace clang {
namespace dpct {

void initRewriterMethodMapCooperativeGroupsSYCLcompat(
    std::unordered_map<std::string,
                       std::shared_ptr<CallExprRewriterFactoryBase>>
        &MethodRewriterMap);

#define ARG(x) makeCallArgCreator(x)
#define UNSUPPORT_FACTORY_ENTRY(FuncName, ...)                                 \
  std::make_pair(FuncName,                                                     \
                 createUnsupportRewriterFactory(FuncName, __VA_ARGS__)),
#define SYCLCOMPAT_UNSUPPORT(NAME)                                             \
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

void CallExprRewriterFactoryBase::initRewriterMapSYCLcompat(
    std::unordered_map<std::string,
                       std::shared_ptr<CallExprRewriterFactoryBase>>
        &RewriterMap) {
  // clang-format off
  RewriterMap.insert({
#include "../RulesLang/APINamesGraph.inc"
#include "../RulesLang/APINamesTexture.inc"
#include "../RulesLang/APINamesGraphicsInterop.inc"
#include "../RulesLang/APINamesWmma.inc"
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
SYCLCOMPAT_UNSUPPORT("cuMemcpyPeer")
SYCLCOMPAT_UNSUPPORT("cuMemcpyPeerAsync")
SYCLCOMPAT_UNSUPPORT("cudaMemcpyPeer")
SYCLCOMPAT_UNSUPPORT("cudaMemcpyPeerAsync")
SYCLCOMPAT_UNSUPPORT("cub::LoadDirectBlocked")
SYCLCOMPAT_UNSUPPORT("cub::LoadDirectStriped")
SYCLCOMPAT_UNSUPPORT("cub::StoreDirectBlocked")
SYCLCOMPAT_UNSUPPORT("cub::StoreDirectStriped")
SYCLCOMPAT_UNSUPPORT("cub::ShuffleDown")
SYCLCOMPAT_UNSUPPORT("cub::ShuffleUp")
SYCLCOMPAT_UNSUPPORT("cuPointerGetAttributes")
  });
  // clang-format on
  initRewriterMethodMapCooperativeGroupsSYCLcompat(RewriterMap);
}

void CallExprRewriterFactoryBase::initRewriterMethodMapSYCLcompat(
    std::unordered_map<std::string,
                       std::shared_ptr<CallExprRewriterFactoryBase>>
        &MethodRewriterMap) {
  // clang-format off
  MethodRewriterMap.insert({
SYCLCOMPAT_UNSUPPORT("cub::BlockRadixSort.Sort")
SYCLCOMPAT_UNSUPPORT("cub::BlockRadixSort.SortDescending")
SYCLCOMPAT_UNSUPPORT("cub::BlockRadixSort.SortBlockedToStriped")
SYCLCOMPAT_UNSUPPORT("cub::BlockRadixSort.SortDescendingBlockedToStriped")
SYCLCOMPAT_UNSUPPORT("cub::BlockExchange.BlockedToStriped")
SYCLCOMPAT_UNSUPPORT("cub::BlockExchange.StripedToBlocked")
SYCLCOMPAT_UNSUPPORT("cub::BlockExchange.ScatterToBlocked")
SYCLCOMPAT_UNSUPPORT("cub::BlockExchange.ScatterToStriped")
SYCLCOMPAT_UNSUPPORT("cub::BlockShuffle.Offset")
SYCLCOMPAT_UNSUPPORT("cub::BlockShuffle.Rotate")
SYCLCOMPAT_UNSUPPORT("cub::BlockShuffle.Up")
SYCLCOMPAT_UNSUPPORT("cub::BlockShuffle.Down")
SYCLCOMPAT_UNSUPPORT("cub::BlockLoad.Load")
SYCLCOMPAT_UNSUPPORT("cub::BlockStore.Store")
  });
  // clang-format on
  initRewriterMethodMapCooperativeGroupsSYCLcompat(MethodRewriterMap);
}
} // namespace dpct
} // namespace clang

#undef ARG
#undef UNSUPPORT_FACTORY_ENTRY
#undef SYCLCOMPAT_UNSUPPORT
#undef ENTRY_UNSUPPORTED
#undef CONDITIONAL_FACTORY_ENTRY
#undef FEATURE_REQUEST_FACTORY
#undef ASSIGNABLE_FACTORY
#undef ENTRY_RENAMED
#undef MEMBER_CALL_FACTORY_ENTRY
#undef CALL_FACTORY_ENTRY
#undef ASSIGN_FACTORY_ENTRY
#undef ENTRY_TEXTURE
#undef ENTRY_BIND
#undef DELETE_FACTORY_ENTRY
#undef DELETER_FACTORY_ENTRY
#undef MULTI_STMTS_FACTORY_ENTRY
#undef WARNING_FACTORY_ENTRY

#include "RuleInfra/CallExprRewriterCommon.h"

void dpct::initRewriterMethodMapCooperativeGroupsSYCLcompat(
    std::unordered_map<std::string,
                       std::shared_ptr<CallExprRewriterFactoryBase>>
        &MethodRewriterMap) {
  auto SyclId2Dim3 = [](const std::string &SourceMember,
                        const std::string &MemberCallName) {
    auto GetID = [Member = MemberCallName](std::string Dimension) {
      return makeMemberCallCreator<false>(MemberExprBase(), false,
                                          std::move(Member),
                                          makeLiteral(std::move(Dimension)));
    };
    auto Name = "cooperative_groups::__v1::thread_block." + SourceMember;
    return std::make_pair(
        Name,
        createCallExprRewriterFactory(
            Name, makeCallExprCreator(MapNames::getDpctNamespace() + "dim3",
                                      GetID("2"), GetID("1"), GetID("0"))));
  };
  MethodRewriterMap.insert(SyclId2Dim3("group_index", "get_group_id"));
  MethodRewriterMap.insert(SyclId2Dim3("thread_index", "get_local_id"));
}