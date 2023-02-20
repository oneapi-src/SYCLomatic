//===--------------- MemberExprRewriter.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemberExprRewriter.h"

namespace clang {
namespace dpct {

template <class BaseT, class MemberArgsT>
std::shared_ptr<MemberExprRewriterFactoryBase> createMemberRewriterFactory(
    std::function<BaseT(const MemberExpr *)> BaseCreator,
    std::function<bool(const MemberExpr *)> IsArrow,
    std::function<MemberArgsT(const MemberExpr *)> MemCreator)
{
  return std::make_shared<MemberExprRewriterFactory<
                          MemberExprFieldRewriter<BaseT, MemberArgsT>,
                          std::function<BaseT(const MemberExpr *)>,
                          std::function<bool(const MemberExpr *)>,
                          std::function<MemberArgsT(const MemberExpr *)>>
                          >(BaseCreator, IsArrow, MemCreator);
}

template <class... MsgArgs>
class ReportMemWarningRewriterFactory
    : public MemberExprRewriterFactory<UnsupportExprRewriter<MsgArgs...>,
                                     Diagnostics, MsgArgs...> {
  using BaseT = MemberExprRewriterFactory<UnsupportExprRewriter<MsgArgs...>,
                                        Diagnostics, MsgArgs...>;
  std::shared_ptr<MemberExprRewriterFactoryBase> First;

public:
  ReportMemWarningRewriterFactory(
      std::shared_ptr<MemberExprRewriterFactoryBase> FirstFactory,
      std::string FuncName, Diagnostics MsgID, MsgArgs... Args)
      : BaseT(MsgID, Args...), First(FirstFactory) {}
  std::shared_ptr<MemberExprBaseRewriter> create(const MemberExpr *M) const override {
    auto R = BaseT::create(M);
    if (First)
      return First->create(M);
    return R;
  }
};


std::function<std::string(const MemberExpr *)> makeMemberBase() {
  return [=] (const MemberExpr *ME) -> std::string {
    auto Base = ME->getBase()->IgnoreImpCasts();
    if (!Base)
      return "";
    return getStmtSpelling(Base);
  };
}

std::function<bool(const MemberExpr *)> isArrow() {
  return [=] (const MemberExpr *ME) -> bool {
    auto Base = ME->getBase()->IgnoreImpCasts();
    if (Base->getType()->isPointerType())
      return true;
    return false;
  };
}

std::function<std::string(const MemberExpr *)>
                makeMemberGetCall(std::string FuncName) {
  return [=] (const MemberExpr * ME) -> std::string {
    return FuncName + "()";
  };
}

std::function<std::string(const MemberExpr *)>
makeLiteral(std::string literal) {
  return [=](const MemberExpr *ME) {
    return literal;
  };
}

std::unique_ptr<std::unordered_map<
        std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>>
        MemberExprRewriterFactoryBase::MemberExprRewriterMap;

template <class... ArgsT>
inline std::shared_ptr<MemberExprRewriterFactoryBase> createReportMemWarningRewriterFactory(
    std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
        Factory, const std::string &FuncName, Diagnostics MsgId,
        ArgsT... ArgsCreator) {
  return std::make_shared<ReportMemWarningRewriterFactory<ArgsT...>> (
      Factory.second, FuncName, MsgId, ArgsCreator...);
}

void MemberExprRewriterFactoryBase::initMemberExprRewriterMap() {
    MemberExprRewriterMap = std::make_unique<std::unordered_map<
      std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>>(
        std::unordered_map<std::string,
                         std::shared_ptr<MemberExprRewriterFactoryBase>>({

#define CALL(...) makeCallExprCreator(__VA_ARGS__)
#define MEMBER_REWRITE_ENTRY(Name, Factory) {Name, Factory},
#define WARNING_FACTORY_ENTRY(Name, Factory, ...) {Name,                \
        createReportMemWarningRewriterFactory(Factory Name, __VA_ARGS__)},
#define MEMBER_FACTORY(...) createMemberRewriterFactory(__VA_ARGS__)
#define MEM_BASE makeMemberBase()
#define MEM_CALL(x) makeMemberGetCall(x)
#define LITERAL(x) makeLiteral(x)
#define IS_ARROW isArrow()
#include "APINamesMemberExpr.inc"
#undef MEMBER_REWRITE_ENTRY
#undef WARNING_FACTORY_ENTRY
#undef MEMBER_FACTORY
#undef MEM_BASE
#undef MEM_ATTR
#undef LITERAL
      }));
}
} // namespace dpct
} // namespace clang

