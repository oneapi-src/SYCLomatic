//===--------------- MemberExprRewriter.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RuleInfra/MemberExprRewriter.h"
#include "AnalysisInfo.h"
#include "RuleInfra/MapNames.h"

namespace clang {
namespace dpct {
template <class Printer> class MEPrinterRewriter : Printer, public MERewriter {
public:
  template <class... ArgsT>
  MEPrinterRewriter(const MemberExpr *ME, StringRef Source, ArgsT &&...Args)
      : Printer(std::forward<ArgsT>(Args)...), MERewriter(ME, Source) {}
  template <class... ArgsT>
  MEPrinterRewriter(
      const MemberExpr *ME, StringRef Source,
      const std::function<ArgsT(const MemberExpr *)> &...ArgCreators)
      : MEPrinterRewriter(ME, Source, ArgCreators(ME)...) {}
  std::optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Printer::print(OS);
    return OS.str();
  }
};

template <class BaseT, class MemberT> class MEMemberExprPrinter {
  BaseT Base;
  bool IsArrow;
  MemberT MemberName;

public:
  MEMemberExprPrinter(const BaseT &Base, bool IsArrow, MemberT MemberName)
      : Base(Base), IsArrow(IsArrow), MemberName(MemberName) {}

  template <class StreamT> void print(StreamT &Stream) const {
    printBase(Stream, Base, IsArrow, false);
    dpct::print(Stream, MemberName);
  }
};

template <class BaseT, class MemberT, class... CallArgsT>
class MEMemberCallPrinter
    : public CallExprPrinter<MEMemberExprPrinter<BaseT, MemberT>,
                             CallArgsT...> {
public:
  MEMemberCallPrinter(const BaseT &Base, bool IsArrow, MemberT MemberName,
                      CallArgsT &&...Args)
      : CallExprPrinter<MEMemberExprPrinter<BaseT, MemberT>, CallArgsT...>(
            MEMemberExprPrinter<BaseT, MemberT>(std::move(Base), IsArrow,
                                                std::move(MemberName)),
            std::forward<CallArgsT>(Args)...) {}
};

bool isArrow(const MemberExpr *ME) {
  auto *Base = ME->getBase()->IgnoreImpCasts();
  if (Base->getType()->isPointerType())
    return true;
  return false;
}

template <class BaseT, class... ArgsT>
class MEMemberCallExprRewriter
    : public MEPrinterRewriter<
          MEMemberCallPrinter<BaseT, StringRef, ArgsT...>> {
public:
  MEMemberCallExprRewriter(
      const MemberExpr *ME, StringRef Source,
      const std::function<BaseT(const MemberExpr *)> &BaseCreator,
      StringRef Member,
      const std::function<ArgsT(const MemberExpr *)> &...ArgsCreator)
      : MEPrinterRewriter<MEMemberCallPrinter<BaseT, StringRef, ArgsT...>>(
            ME, Source, BaseCreator(ME), isArrow(ME), Member,
            ArgsCreator(ME)...) {}
  MEMemberCallExprRewriter(
      const MemberExpr *ME, StringRef Source, const BaseT &BaseCreator,
      StringRef Member,
      const std::function<ArgsT(const MemberExpr *)> &...ArgsCreator)
      : MEPrinterRewriter<MEMemberCallPrinter<BaseT, StringRef, ArgsT...>>(
            ME, Source, BaseCreator, isArrow(ME), Member, ArgsCreator(ME)...) {}
};

template <class BaseT, class MemberT>
class MEMemberExprRewriter
    : public MEPrinterRewriter<MEMemberExprPrinter<BaseT, MemberT>> {
public:
  MEMemberExprRewriter(
      const MemberExpr *ME, StringRef Source,
      const std::function<BaseT(const MemberExpr *)> &BaseCreator,
      const std::function<MemberT(const MemberExpr *)> &MemberCreator)
      : MEPrinterRewriter<MEMemberExprPrinter<BaseT, MemberT>>(
            ME, Source, BaseCreator(ME), isArrow(ME), MemberCreator(ME)) {}
};

class MERewriterFactoryWithFeatureRequest
    : public MemberExprRewriterFactoryBase {
  std::shared_ptr<MemberExprRewriterFactoryBase> Inner;
  HelperFeatureEnum Feature;

public:
  MERewriterFactoryWithFeatureRequest(
      HelperFeatureEnum Feature,
      std::shared_ptr<MemberExprRewriterFactoryBase> InnerFactory)
      : Inner(InnerFactory), Feature(Feature) {}
  std::shared_ptr<MERewriter> create(const MemberExpr *ME) const override {
    requestFeature(Feature);
    return Inner->create(ME);
  }
};

class CudaMemoryTypeLiteralNumberMigration
    : public MemberExprRewriterFactoryBase {
  std::shared_ptr<MemberExprRewriterFactoryBase> Inner;

  const IntegerLiteral *
  isComparingWithIntegerLiteral(const MemberExpr *ME) const {
    DynTypedNode Pre;
    const Stmt *Target = DpctGlobalInfo::findAncestor<Stmt>(
        ME, [&](const DynTypedNode &Cur) -> bool {
          if (Cur.get<ImplicitCastExpr>()) {
            Pre = Cur;
            return false;
          }
          return true;
        });
    const BinaryOperator *BO = dyn_cast<BinaryOperator>(Target);
    if (!BO)
      return nullptr;
    const Expr *Another = BO->getRHS();
    if (Pre.get<Stmt>() && (Pre.get<Stmt>() == BO->getRHS()))
      Another = BO->getLHS();
    if (const IntegerLiteral *IL = dyn_cast<IntegerLiteral>(Another))
      return IL;
    return nullptr;
  }

public:
  CudaMemoryTypeLiteralNumberMigration(
      std::shared_ptr<MemberExprRewriterFactoryBase> InnerFactory)
      : Inner(InnerFactory) {}
  std::shared_ptr<MERewriter> create(const MemberExpr *ME) const override {
    if (const IntegerLiteral *IL = isComparingWithIntegerLiteral(ME)) {
      Expr::EvalResult Result{};
      IL->EvaluateAsInt(Result, DpctGlobalInfo::getContext());
      int64_t Value = Result.Val.getInt().getExtValue();
      std::string Repl;
      switch (Value) {
      case 0:
        Repl = MapNames::getClNamespace() + "usm::alloc::unknown";
        break;
      case 1:
        Repl = MapNames::getClNamespace() + "usm::alloc::host";
        break;
      case 2:
        Repl = MapNames::getClNamespace() + "usm::alloc::device";
        break;
      case 3:
        Repl = MapNames::getClNamespace() + "usm::alloc::shared";
        break;
      }
      if (!Repl.empty())
        DpctGlobalInfo::getInstance().addReplacement(
            std::make_shared<ExtReplacement>(
                DpctGlobalInfo::getSourceManager(), IL->getBeginLoc(),
                Lexer::MeasureTokenLength(
                    IL->getBeginLoc(), DpctGlobalInfo::getSourceManager(),
                    DpctGlobalInfo::getContext().getLangOpts()),
                Repl, nullptr));
    }
    return Inner->create(ME);
  }
};

template <class... MsgArgs>
class MEUnsupportFunctionRewriter : public MERewriter {
  template <class T>
  std::string getMsgArg(const std::function<T(const MemberExpr *)> &Func,
                        const MemberExpr *ME) {
    return getMsgArg(Func(ME), ME);
  }
  template <class T>
  static std::string getMsgArg(const T &InputArg, const MemberExpr *) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    print(OS, InputArg);
    return OS.str();
  }

public:
  MEUnsupportFunctionRewriter(const MemberExpr *CE, StringRef CalleeName,
                              Diagnostics MsgID, const MsgArgs &...Args)
      : MERewriter(CE, CalleeName) {
    report(MsgID, false, getMsgArg(Args, CE)...);
  }

  std::optional<std::string> rewrite() override { return std::nullopt; }
};

template <class... MsgArgs>
class MEReportWarningRewriterFactory
    : public MERewriterFactory<MEUnsupportFunctionRewriter<MsgArgs...>,
                               Diagnostics, MsgArgs...> {
  using BaseT = MERewriterFactory<MEUnsupportFunctionRewriter<MsgArgs...>,
                                  Diagnostics, MsgArgs...>;
  std::shared_ptr<MemberExprRewriterFactoryBase> First;

public:
  MEReportWarningRewriterFactory(
      std::shared_ptr<MemberExprRewriterFactoryBase> FirstFactory,
      std::string FuncName, Diagnostics MsgID, MsgArgs... Args)
      : BaseT(FuncName, MsgID, Args...), First(FirstFactory) {}
  std::shared_ptr<MERewriter> create(const MemberExpr *ME) const override {
    auto R = BaseT::create(ME);
    if (First)
      return First->create(ME);
    return R;
  }
};

template <class BaseT, class... ArgsT>
inline std::shared_ptr<MemberExprRewriterFactoryBase>
createMemberMERewriterFactory(
    const std::string &SourceName,
    std::function<BaseT(const MemberExpr *)> BaseCreator,
    std::string MemberName,
    std::function<ArgsT(const MemberExpr *)>... ArgsCreator) {
  return std::make_shared<
      MERewriterFactory<MEMemberCallExprRewriter<BaseT, ArgsT...>,
                        std::function<BaseT(const MemberExpr *)>, std::string,
                        std::function<ArgsT(const MemberExpr *)>...>>(
      SourceName,
      std::forward<std::function<BaseT(const MemberExpr *)>>(BaseCreator),
      MemberName,
      std::forward<std::function<ArgsT(const MemberExpr *)>>(ArgsCreator)...);
}

template <class BaseT, class... ArgsT>
inline std::shared_ptr<MemberExprRewriterFactoryBase>
createMemberMERewriterFactory(
    const std::string &SourceName, BaseT BaseCreator, std::string MemberName,
    std::function<ArgsT(const MemberExpr *)>... ArgsCreator) {
  return std::make_shared<MERewriterFactory<
      MEMemberCallExprRewriter<BaseT, ArgsT...>, BaseT, std::string,
      std::function<ArgsT(const MemberExpr *)>...>>(
      SourceName, BaseCreator, MemberName,
      std::forward<std::function<ArgsT(const MemberExpr *)>>(ArgsCreator)...);
}

template <class BaseT, class MemberT>
inline std::shared_ptr<MemberExprRewriterFactoryBase>
createMemberExprRewriterFactory(
    const std::string &SourceName,
    std::function<BaseT(const MemberExpr *)> &&BaseCreator,
    std::function<MemberT(const MemberExpr *)> &&MemberCreator) {
  return std::make_shared<
      MERewriterFactory<MEMemberExprRewriter<BaseT, MemberT>,
                        std::function<BaseT(const MemberExpr *)>,
                        std::function<MemberT(const MemberExpr *)>>>(
      SourceName,
      std::forward<std::function<BaseT(const MemberExpr *)>>(BaseCreator),
      std::forward<std::function<MemberT(const MemberExpr *)>>(MemberCreator));
}

inline std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<MERewriterFactoryWithFeatureRequest>(Feature,
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

inline std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
createCudaMemoryTypeLiteralNumberMigrationFactory(
    std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<CudaMemoryTypeLiteralNumberMigration>(Input.second));
}

template <class T>
inline std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
createCudaMemoryTypeLiteralNumberMigrationFactory(
    std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
        &&Input,
    T) {
  return createCudaMemoryTypeLiteralNumberMigrationFactory(std::move(Input));
}

template <class... ArgsT>
inline std::shared_ptr<MemberExprRewriterFactoryBase>
createReportWarningRewriterFactory(
    std::pair<std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>
        Factory,
    const std::string &FuncName, Diagnostics MsgId, ArgsT... ArgsCreator) {
  return std::make_shared<MEReportWarningRewriterFactory<ArgsT...>>(
      Factory.second, FuncName, MsgId, ArgsCreator...);
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
  {FuncName, createMemberMERewriterFactory(FuncName, __VA_ARGS__)},
#define MEM_EXPR_ENTRY(FuncName, B, M)                                         \
  {FuncName, createMemberExprRewriterFactory(FuncName, B, M)},
#define FEATURE_REQUEST_FACTORY(FEATURE, x)                                    \
  createFeatureRequestFactory(FEATURE, x 0),
#define CUDA_MEMORY_TYPE_LITERAL_NUMBER_MIGRATION_FACTORY(x)                   \
  createCudaMemoryTypeLiteralNumberMigrationFactory(x 0),
#define WARNING_FACTORY_ENTRY(FuncName, Factory, ...)                          \
  {FuncName, createReportWarningRewriterFactory(Factory FuncName, __VA_ARGS__)},
#include "RulesLang/APINamesMemberExpr.inc"
#include "RulesLangLib/APINamesMemberExprCUB.inc"
#undef MEMBER_CALL_FACTORY_ENTRY
#undef MEM_BASE
      }));
}
} // namespace dpct
} // namespace clang
