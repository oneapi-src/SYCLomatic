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

/*
   Note that tiled_partition, the function for constructing thread_block_tiles,
   has the signature

     template <unsigned int Size, typename ParentT=void>
     thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)

   so tiled_partition<32>(tb) has type tiled_partition<32, thread_block>.
   However, you can assign tiled_partition<32, thread_block> to
   tiled_partition<32> ( = tiled_partition<32, void> ).
   Thus, the declarations

     tiled_partition<32> tp1 = tiled_partition<32>(tb);
     auto tp2 = tiled_partition<32>(tb);

   have different types.
*/
static inline bool hasThreadBlockTileType(const Expr *E, unsigned nTile) {
  const auto typeName =
    DpctGlobalInfo::getUnqualifiedTypeName(E->getType().getCanonicalType());

  // search for substr starting at index i,
  // return empty optional if not string found at that index,
  // otherwise return optional with index of the end of the substr
  const auto find = [&](const std::string &s, unsigned i = 0)
    -> std::optional<decltype(s.size())> {
    const auto res = typeName.find(s, i);
    if (res != i)
      return {};
    else
      return res + s.size();
  };

  // search for "cooperative_groups::__v1::thread_block_tile<{i}",
  // then check for a comma or right angle bracket
  // (so an input like thread_block_tile<320, ...> returns false
  const auto x1 = find("cooperative_groups::__v1::thread_block_tile<");
  if (!x1) return false;
  const auto x2 = find(std::to_string(nTile), *x1);
  if (!x2) return false;
  return find(",", *x2) || find(">", *x2);
}

static inline std::function<bool(const CallExpr*)>
argHasThreadBlockTileType(unsigned argIdx, unsigned nTile) {
  return [=](const CallExpr *CE) {
    if (argIdx < CE->getNumArgs())
      return hasThreadBlockTileType(CE->getArg(argIdx), nTile);
    else
      return false;
  };
}

static inline std::function<bool(const CallExpr*)>
baseHasThreadBlockTileType(unsigned nTile) {
  return [=](const CallExpr *CE) {
    if (const auto ME = dyn_cast<MemberExpr>(CE->getCallee()->IgnoreImplicit())) {
      return hasThreadBlockTileType(ME->getBase()->IgnoreImplicit(), nTile);
    } else
      return false;
  };
}

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

void CallExprRewriterFactoryBase::initMethodRewriterMapCooperativeGroups() {
  MethodRewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#define CLASS_METHOD_CALL
#include "APINamesCooperativeGroups.inc"
#undef CLASS_METHOD_CALL
      }));
}

} // namespace dpct
} // namespace clang
