//===--------------- MemberExprRewriter.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemberExprRewriter.h"

namespace clang {
namespace dpct {
namespace member_expr {

template <class Printer>
class PrinterRewriter : Printer, public CallExprRewriter {
public:
  template <class... ArgsT>
  PrinterRewriter(const MemberExpr *C, StringRef Source, ArgsT &&...Args)
      : Printer(std::forward<ArgsT>(Args)...), CallExprRewriter(C, Source) {}
  template <class... ArgsT>
  PrinterRewriter(
      const MemberExpr *C, StringRef Source,
      const std::function<ArgsT(const MemberExpr *)> &...ArgCreators)
      : PrinterRewriter(C, Source, ArgCreators(C)...) {}
  std::optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Printer::print(OS);
    return OS.str();
  }
};

template <class BaseT, class MemberT, class... CallArgsT>
class MemberCallPrinter
    : public CallExprPrinter<MemberExprPrinter<BaseT, MemberT>, CallArgsT...> {
public:
  MemberCallPrinter(const BaseT &Base, bool IsArrow, MemberT MemberName,
                    CallArgsT &&...Args)
      : CallExprPrinter<MemberExprPrinter<BaseT, MemberT>, CallArgsT...>(
            MemberExprPrinter<BaseT, MemberT>(std::move(Base), IsArrow,
                                              std::move(MemberName)),
            std::forward<CallArgsT>(Args)...) {}
};

template <class BaseT, class... ArgsT>
class MemberCallExprRewriter
    : public PrinterRewriter<MemberCallPrinter<BaseT, StringRef, ArgsT...>> {
public:
  MemberCallExprRewriter(
      const MemberExpr *C, StringRef Source,
      const std::function<BaseT(const MemberExpr *)> &BaseCreator, bool IsArrow,
      StringRef Member,
      const std::function<ArgsT(const MemberExpr *)> &...ArgsCreator)
      : PrinterRewriter<MemberCallPrinter<BaseT, StringRef, ArgsT...>>(
            C, Source, BaseCreator(C), IsArrow, Member, ArgsCreator(C)...) {}
  MemberCallExprRewriter(
      const MemberExpr *C, StringRef Source, const BaseT &BaseCreator,
      bool IsArrow, StringRef Member,
      const std::function<ArgsT(const MemberExpr *)> &...ArgsCreator)
      : PrinterRewriter<MemberCallPrinter<BaseT, StringRef, ArgsT...>>(
            C, Source, BaseCreator, IsArrow, Member, ArgsCreator(C)...) {}
};

template <class BaseT, class... ArgsT>
inline std::shared_ptr<MemberExprRewriterFactoryBase>
createMemberCallExprRewriterFactory(
    const std::string &SourceName,
    std::function<BaseT(const MemberExpr *)> BaseCreator, bool IsArrow,
    std::string MemberName,
    std::function<ArgsT(const MemberExpr *)>... ArgsCreator) {
  return std::make_shared<CallExprRewriterFactory<
      MemberCallExprRewriter<BaseT, ArgsT...>,
      std::function<BaseT(const MemberExpr *)>, bool, std::string,
      std::function<ArgsT(const MemberExpr *)>...>>(
      SourceName,
      std::forward<std::function<BaseT(const MemberExpr *)>>(BaseCreator),
      IsArrow, MemberName,
      std::forward<std::function<ArgsT(const MemberExpr *)>>(ArgsCreator)...);
}

template <class BaseT, class... ArgsT>
inline std::shared_ptr<MemberExprRewriterFactoryBase>
createMemberCallExprRewriterFactory(
    const std::string &SourceName, BaseT BaseCreator, bool IsArrow,
    std::string MemberName,
    std::function<ArgsT(const MemberExpr *)>... ArgsCreator) {
  return std::make_shared<CallExprRewriterFactory<
      MemberCallExprRewriter<BaseT, ArgsT...>, BaseT, bool, std::string,
      std::function<ArgsT(const MemberExpr *)>...>>(
      SourceName, BaseCreator, IsArrow, MemberName,
      std::forward<std::function<ArgsT(const MemberExpr *)>>(ArgsCreator)...);
}

inline std::function<const std::string(const MemberExpr *)> makeMemberBase() {
  return [=](const MemberExpr *ME) -> const std::string {
    auto *Base = ME->getBase()->IgnoreImpCasts();
    if (!Base)
      return "";
    return getStmtSpelling(Base);
  };
}

std::unique_ptr<std::unordered_map<
    std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>>
    MemberExprRewriterFactoryBase::MemberExprRewriterMap;

void MemberExprRewriterFactoryBase::initMemberExprRewriterMap() {
  MemberExprRewriterMap = std::make_unique<std::unordered_map<
      std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>>(
      std::unordered_map<std::string,
                         std::shared_ptr<MemberExprRewriterFactoryBase>>({
#define MEM_BASE makeMemberBase()
#define MEMBER_CALL_FACTORY_ENTRY(FuncName, ...)                               \
  {FuncName, createMemberCallExprRewriterFactory(FuncName, __VA_ARGS__)},
#include "APINamesMemberExpr.inc"
#undef MEMBER_CALL_FACTORY_ENTRY
#undef MEM_BASE
      }));
}
} // namespace member_expr
} // namespace dpct
} // namespace clang
