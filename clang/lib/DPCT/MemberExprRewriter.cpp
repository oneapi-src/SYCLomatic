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

template <class BaseT, class MemberT> class MemberExprPrinter {
  BaseT Base;
  bool IsArrow;
  MemberT MemberName;

public:
  MemberExprPrinter(const BaseT &Base, bool IsArrow, MemberT MemberName)
      : Base(Base), IsArrow(IsArrow), MemberName(MemberName) {}

  template <class StreamT> void print(StreamT &Stream) const {
    printBase(Stream, Base, IsArrow);
    dpct::print(Stream, MemberName);
  }
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

template <class BaseT, class MemberT>
class MemberExprRewriter
    : public PrinterRewriter<MemberExprPrinter<BaseT, MemberT>> {
public:
  MemberExprRewriter(
      const MemberExpr *C, StringRef Source,
      const std::function<BaseT(const MemberExpr *)> &BaseCreator, bool IsArrow,
      const std::function<MemberT(const MemberExpr *)> &MemberCreator)
      : PrinterRewriter<MemberExprPrinter<BaseT, MemberT>>(
            C, Source, BaseCreator(C), IsArrow, MemberCreator(C)) {}
};

class RewriterFactoryWithFeatureRequest : public MemberExprRewriterFactoryBase {
  std::shared_ptr<MemberExprRewriterFactoryBase> Inner;
  HelperFeatureEnum Feature;

public:
  RewriterFactoryWithFeatureRequest(
      HelperFeatureEnum Feature,
      std::shared_ptr<MemberExprRewriterFactoryBase> InnerFactory)
      : Inner(InnerFactory), Feature(Feature) {}
  std::shared_ptr<CallExprRewriter> create(const MemberExpr *C) const override {
    requestFeature(Feature);
    return Inner->create(C);
  }
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

template <class BaseT, class MemberT>
inline std::shared_ptr<MemberExprRewriterFactoryBase>
createMemberExprRewriterFactory(
    const std::string &SourceName,
    std::function<BaseT(const MemberExpr *)> &&BaseCreator, bool IsArrow,
    std::function<MemberT(const MemberExpr *)> &&MemberCreator) {
  return std::make_shared<
      CallExprRewriterFactory<MemberExprRewriter<BaseT, MemberT>,
                              std::function<BaseT(const MemberExpr *)>, bool,
                              std::function<MemberT(const MemberExpr *)>>>(
      SourceName,
      std::forward<std::function<BaseT(const MemberExpr *)>>(BaseCreator),
      IsArrow,
      std::forward<std::function<MemberT(const MemberExpr *)>>(MemberCreator));
}

inline std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<RewriterFactoryWithFeatureRequest>(Feature,
                                                          Input.second));
}

template <class T>
inline std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
        &&Input,
    T) {
  return createFeatureRequestFactory(Feature, std::move(Input));
}

inline std::function<const std::string(const MemberExpr *)> makeMemberBase() {
  return [=](const MemberExpr *ME) -> const std::string {
    auto *Base = ME->getBase()->IgnoreImpCasts();
    if (!Base)
      return "";
    return getStmtSpelling(Base);
  };
}

inline std::function<std::string(const MemberExpr *)>
makeLiteral(std::string Str) {
  return [=](const MemberExpr *) { return Str; };
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
#define LITERAL(x) makeLiteral(x)
#define MEMBER_CALL_FACTORY_ENTRY(FuncName, ...)                               \
  {FuncName, createMemberCallExprRewriterFactory(FuncName, __VA_ARGS__)},
#define MEM_EXPR_ENTRY(FuncName, B, IsArrow, M)                                \
  {FuncName, createMemberExprRewriterFactory(FuncName, B, IsArrow, M)},
#define FEATURE_REQUEST_FACTORY(FEATURE, x)                                    \
  createFeatureRequestFactory(FEATURE, x 0),
#include "APINamesMemberExpr.inc"
#undef MEMBER_CALL_FACTORY_ENTRY
#undef MEM_BASE
      }));
}
} // namespace member_expr
} // namespace dpct
} // namespace clang
