//===--------------- CallExprRewriter.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CALL_EXPR_REWRITER_H
#define CALL_EXPR_REWRITER_H

#include "Diagnostics.h"


namespace clang {
namespace dpct {

class CallExprRewriter;
class FuncCallExprRewriter;
class MathCallExprRewriter;
class MathFuncNameRewriter;
class MathSimulatedRewriter;
class MathTypeCastRewriter;
class MathBinaryOperatorRewriter;
class MathUnsupportedRewriter;
class WarpFunctionRewriter;
class NoRewriteFuncNameRewriter;
template <class... MsgArgs> class UnsupportFunctionRewriter;

/*
Factory usage example:
using FuncCallExprRewriterFactory =
    CallExprRewriterFactory<FuncCallExprRewriter, std::string>;
*/
/// Base class in abstract factory pattern
class CallExprRewriterFactoryBase {
public:
  virtual std::shared_ptr<CallExprRewriter> create(const CallExpr *) const = 0;
  virtual ~CallExprRewriterFactoryBase() {}

  static std::unique_ptr<std::unordered_map<
      std::string, std::shared_ptr<CallExprRewriterFactoryBase>>>
      RewriterMap;
  static std::unique_ptr<std::unordered_map<
      std::string, std::shared_ptr<CallExprRewriterFactoryBase>>>
      MethodRewriterMap;
  static void initRewriterMap();
  RulePriority Priority = RulePriority::Fallback;
private:
  static void initRewriterMapAtomic();
  static void initRewriterMapCUB();
  static void initRewriterMapCUFFT();
  static void initRewriterMapCUBLAS();
  static void initRewriterMapCURAND();
  static void initRewriterMapCUSOLVER();
  static void initRewriterMapCUSPARSE();
  static void initRewriterMapComplex();
  static void initRewriterMapDriver();
  static void initRewriterMapMemory();
  static void initRewriterMapMisc();
  static void initRewriterMapNccl();
  static void initRewriterMapStream();
  static void initRewriterMapTexture();
  static void initRewriterMapThrust();
  static void initRewriterMapWarp();
  static void initRewriterMapCUDNN();
  static void initRewriterMapErrorHandling();
  static void initRewriterMapLIBCU();
  static void initRewriterMapEvent();
  static void initRewriterMapMath();
  static void initRewriterMapCooperativeGroups();
  static void initRewriterMapWmma();
  static void initMethodRewriterMapCUB();
  static void initMethodRewriterMapCooperativeGroups();
  static void initMethodRewriterMapLIBCU();
};

/// Abstract factory for all rewriter factories
template <class RewriterTy, class... Args>
class CallExprRewriterFactory : public CallExprRewriterFactoryBase {
  std::tuple<std::string, Args...> Initializer;

private:
  template <size_t... Idx>
  inline std::shared_ptr<CallExprRewriter>
  createRewriter(const CallExpr *Call, std::index_sequence<Idx...>) const {
    return std::shared_ptr<RewriterTy>(
        new RewriterTy(Call, std::get<Idx>(Initializer)...));
  }

public:
  CallExprRewriterFactory(StringRef SourceCalleeName, Args... Arguments)
      : Initializer(SourceCalleeName.str(), std::forward<Args>(Arguments)...) {}
  // Create a meaningful rewriter only if the CallExpr is not nullptr
  std::shared_ptr<CallExprRewriter>
  create(const CallExpr *Call) const override {
    if (!Call)
      return std::shared_ptr<CallExprRewriter>();
    return createRewriter(Call,
                          std::index_sequence_for<std::string, Args...>());
  }
};

using FuncCallExprRewriterFactory =
    CallExprRewriterFactory<FuncCallExprRewriter, std::string>;
template <class... MsgArgs>
using UnsupportFunctionRewriterFactory =
    CallExprRewriterFactory<UnsupportFunctionRewriter<MsgArgs...>, Diagnostics,
                            MsgArgs...>;

template <class Printer, class... Ts> class PrinterCreator {
  std::tuple<Ts...> Creators;

  template <class T, class Node> T create(T Val, const Node *) { return Val; }
  template <class Node>
  StringRef create(const std::string &Val, const Node *) { return Val; }
  template <class T, class Node>
  T create(std::function<T(const Node *)> &Func, const Node *C) {
    return Func(C);
  }
  template <class Node, size_t... Idx>
  Printer createPrinter(const Node *C, std::index_sequence<Idx...>) {
    return Printer(create(std::get<Idx>(Creators), C)...);
  }

public:
  PrinterCreator(Ts... Args) : Creators(Args...) {}
  template <class Node>
  Printer operator()(const Node *C) {
    return createPrinter(C, std::index_sequence_for<Ts...>());
  }
};

/// Base class for rewriting call expressions
class CallExprRewriter {
protected:
  // Call is guaranteed not to be nullptr
  const CallExpr *Call;
  StringRef SourceCalleeName;

protected:
  // All instances of the subclasses can only be constructed by corresponding
  // factories. As a result, the access modifiers of the constructors are
  // supposed to be protected instead of public.
  CallExprRewriter(const CallExpr *Call, StringRef SourceCalleeName)
      : Call(Call), SourceCalleeName(SourceCalleeName) {}
  bool NoRewrite = false;

public:
  ArgumentAnalysis Analyzer;
  virtual ~CallExprRewriter() {}

  /// This function should be overwritten to implement call expression
  /// rewriting.
  virtual std::optional<std::string> rewrite() = 0;
  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  inline void report(IDTy MsgID, bool UseTextBegin, Ts &&...Vals) {
    TransformSetTy TS;
    auto SL = Call->getBeginLoc();
    DiagnosticsUtils::report<IDTy, Ts...>(
        SL, MsgID, &TS, UseTextBegin, std::forward<Ts>(Vals)...);
    for (auto &T : TS)
      DpctGlobalInfo::getInstance().addReplacement(
          T->getReplacement(DpctGlobalInfo::getContext()));
  }

  bool isNoRewrite() { return NoRewrite; }

  bool getBlockLevelFormatFlag() const {
    return BlockLevelFormatFlag;
  }

protected:
  bool BlockLevelFormatFlag = false;
  std::vector<std::string> getMigratedArgs();
  std::string getMigratedArg(unsigned Index);
  std::string getMigratedArgWithExtraParens(unsigned Index);

  StringRef getSourceCalleeName() { return SourceCalleeName; }
};

class ConditionalRewriterFactory : public CallExprRewriterFactoryBase {
  std::function<bool(const CallExpr *)> Pred;
  std::shared_ptr<CallExprRewriterFactoryBase> First, Second;

protected:
  void setElse(std::shared_ptr<CallExprRewriterFactoryBase> SecondFactory) {
    Second = SecondFactory;
  }

public:
  template <class InputPred>
  ConditionalRewriterFactory(
      InputPred &&P, std::shared_ptr<CallExprRewriterFactoryBase> FirstFactory,
      std::shared_ptr<CallExprRewriterFactoryBase> SecondFactory)
      : Pred(std::forward<InputPred>(P)), First(FirstFactory),
        Second(SecondFactory) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) const override {
    if (Pred(C))
      return First->create(C);
    else
      return Second->create(C);
  }
};

class MathSpecificElseEmuRewriterFactory final
    : public ConditionalRewriterFactory {
public:
  template <class InputPred>
  MathSpecificElseEmuRewriterFactory(
      InputPred &&P, std::shared_ptr<CallExprRewriterFactoryBase> FirstFactory)
      : ConditionalRewriterFactory(P, FirstFactory, nullptr) {}
  using ConditionalRewriterFactory::setElse;
};

class CaseRewriterFactory : public CallExprRewriterFactoryBase {
public:
  using PredT = std::function<bool(const CallExpr *)>;
  using CaseT = std::pair<PredT, std::shared_ptr<CallExprRewriterFactoryBase>>;

  inline static const PredT true_pred = [](const CallExpr *) { return true; };
  
  std::vector<CaseT> Cases;

  template <class... CaseTs>
  CaseRewriterFactory(CaseTs&&... cases)
    : Cases{std::forward<CaseTs>(cases)...} {}
  
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) const override {
    for (const auto& [Pred, Factory] : Cases) {
      if (Pred(C)) {
	return Factory->create(C);
      }
    }
    throw std::runtime_error("Non-exhaustive CaseRewriterFactory");
  }
};

template <class... MsgArgs>
class ReportWarningRewriterFactory
    : public CallExprRewriterFactory<UnsupportFunctionRewriter<MsgArgs...>,
                                     Diagnostics, MsgArgs...> {
  using BaseT = CallExprRewriterFactory<UnsupportFunctionRewriter<MsgArgs...>,
                                        Diagnostics, MsgArgs...>;
  std::shared_ptr<CallExprRewriterFactoryBase> First;

public:
  ReportWarningRewriterFactory(
      std::shared_ptr<CallExprRewriterFactoryBase> FirstFactory,
      std::string FuncName, Diagnostics MsgID, MsgArgs... Args)
      : BaseT(FuncName, MsgID, Args...), First(FirstFactory) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) const override {
    auto R = BaseT::create(C);
    if (First)
      return First->create(C);
    return R;
  }
};

class AssignableRewriter : public CallExprRewriter {
  std::shared_ptr<CallExprRewriter> Inner;
  bool IsAssigned;
  bool IsInRetStmt;
  bool CheckAssigned;
  bool CheckInRetStmt;
  bool UseDpctCheckError;
  bool ExtraParen;

public:
  AssignableRewriter(const CallExpr *C,
                     std::shared_ptr<CallExprRewriter> InnerRewriter,
                     bool checkAssigned = true, bool checkInRetStmt = false,
                     bool useDpctCheckError = true, bool EP = false)
      : CallExprRewriter(C, ""), Inner(InnerRewriter), IsAssigned(false),
        IsInRetStmt(false), CheckAssigned(checkAssigned),
        CheckInRetStmt(checkInRetStmt), UseDpctCheckError(useDpctCheckError),
        ExtraParen(EP) {
    if (CheckAssigned) {
      IsAssigned = isAssigned(C);
    }
    if (CheckInRetStmt) {
      IsInRetStmt = isInRetStmt(C);
    }
    if (IsAssigned)
      requestFeature(HelperFeatureEnum::device_ext);
  }

  std::optional<std::string> rewrite() override {
    std::optional<std::string> &&Result = Inner->rewrite();
    if (Result.has_value()) {
      if ((CheckAssigned && IsAssigned) || (CheckInRetStmt && IsInRetStmt)) {
        if (UseDpctCheckError) {
          if (ExtraParen) {
            return "DPCT_CHECK_ERROR((" + Result.value() + "))";
          }
          return "DPCT_CHECK_ERROR(" + Result.value() + ")";
        } else {
          if (ExtraParen) {
            return "[&](){ (" + Result.value() + "); }()";
          }
          return "[&](){ " + Result.value() + "; }()";
        }
      }
    }
    return Result;
  }
};

class InsertAroundRewriter : public CallExprRewriter {
  std::string Prefix;
  std::string Suffix;
  std::shared_ptr<CallExprRewriter> Inner;

public:
  InsertAroundRewriter(const CallExpr *C, std::string Prefix,
                       std::string Suffix,
                       std::shared_ptr<CallExprRewriter> InnerRewriter)
      : CallExprRewriter(C, ""), Prefix(Prefix), Suffix(Suffix),
        Inner(InnerRewriter) {}

  std::optional<std::string> rewrite() override {
    std::optional<std::string> &&Result = Inner->rewrite();
    if (Result.has_value())
      return Prefix + Result.value() + Suffix;
    return Result;
  }
};

class RemoveAPIRewriter : public CallExprRewriter {
  bool IsAssigned = false;
  std::string CalleeName;
  std::string Message;

public:
  RemoveAPIRewriter(const CallExpr *C, std::string CalleeName,
                    std::string Message = "")
      : CallExprRewriter(C, CalleeName), IsAssigned(isAssigned(C)),
        CalleeName(CalleeName), Message(Message) {}

  std::optional<std::string> rewrite() override {
    std::string Msg =
        Message.empty() ? "this call is redundant in SYCL." : Message;
    if (IsAssigned) {
      report(Diagnostics::FUNC_CALL_REMOVED_0, false, CalleeName, Msg);
      return std::optional<std::string>("0");
    }
    report(Diagnostics::FUNC_CALL_REMOVED, false,
           CalleeName, Msg);
    return std::optional<std::string>("");
  }
};

class IfElseRewriter : public CallExprRewriter {
  std::shared_ptr<CallExprRewriter> Pred;
  std::shared_ptr<CallExprRewriter> IfBlock;
  std::shared_ptr<CallExprRewriter> ElseBlock;
  StringRef NL;
  StringRef Indent;

public:
  IfElseRewriter(const CallExpr *C, StringRef SourceName,
                 std::shared_ptr<CallExprRewriterFactoryBase> PredCreator,
                 std::shared_ptr<CallExprRewriterFactoryBase> IfBlockCreator,
                 std::shared_ptr<CallExprRewriterFactoryBase> ElseBlockCreator)
      : CallExprRewriter(C, ""), Pred(PredCreator->create(C)),
        IfBlock(IfBlockCreator->create(C)),
        ElseBlock(ElseBlockCreator->create(C)) {
    auto &SM = dpct::DpctGlobalInfo::getSourceManager();
    NL = getNL(getStmtExpansionSourceRange(C).getBegin(), SM);
    Indent = getIndent(getStmtExpansionSourceRange(C).getBegin(), SM);
  }

  std::optional<std::string> rewrite() override {
    std::optional<std::string> &&PredStr = Pred->rewrite();
    std::optional<std::string> &&IfBlockStr = IfBlock->rewrite();
    std::optional<std::string> &&ElseBlockStr = ElseBlock->rewrite();
    return "if(" + PredStr.value() + "){" + NL.str() + Indent.str() +
           Indent.str() + IfBlockStr.value() + ";" + NL.str() +
           Indent.str() + "} else {" + NL.str() + Indent.str() + Indent.str() +
           ElseBlockStr.value() + ";" + NL.str() + Indent.str() + "}";
  }
};

class AssignableRewriterFactory : public CallExprRewriterFactoryBase {
  std::shared_ptr<CallExprRewriterFactoryBase> Inner;
  bool CheckAssigned;
  bool CheckInRetStmt;
  bool UseDpctCheckError;
  bool ExtraParen;

public:
  AssignableRewriterFactory(
      std::shared_ptr<CallExprRewriterFactoryBase> InnerFactory,
      bool checkAssigned = true, bool checkInRetStmt = false,
      bool useDpctCheckError = true, bool EP = false)
      : Inner(InnerFactory), CheckAssigned(checkAssigned),
        CheckInRetStmt(checkInRetStmt), UseDpctCheckError(useDpctCheckError),
        ExtraParen(EP) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) const override {
    return std::make_shared<AssignableRewriter>(C, Inner->create(C),
                                                CheckAssigned, CheckInRetStmt,
                                                UseDpctCheckError, ExtraParen);
  }
};

class InsertAroundRewriterFactory : public CallExprRewriterFactoryBase {
  std::string Prefix;
  std::string Suffix;
  std::shared_ptr<CallExprRewriterFactoryBase> Inner;

public:
  InsertAroundRewriterFactory(
      std::shared_ptr<CallExprRewriterFactoryBase> InnerFactory,
      std::string Prefix, std::string Suffix)
      : Prefix(Prefix), Suffix(Suffix), Inner(InnerFactory) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) const override {
    return std::make_shared<InsertAroundRewriter>(C, Prefix, Suffix,
                                                  Inner->create(C));
  }
};

class RewriterFactoryWithFeatureRequest : public CallExprRewriterFactoryBase {
  std::shared_ptr<CallExprRewriterFactoryBase> Inner;
  HelperFeatureEnum Feature;

public:
  RewriterFactoryWithFeatureRequest(
      HelperFeatureEnum Feature,
      std::shared_ptr<CallExprRewriterFactoryBase> InnerFactory)
      : Inner(InnerFactory), Feature(Feature) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) const override {
    requestFeature(Feature);
    return Inner->create(C);
  }
};

class RewriterFactoryWithHeaderFile: public CallExprRewriterFactoryBase {
  std::shared_ptr<CallExprRewriterFactoryBase> Inner;
  HeaderType Header;

public:
  RewriterFactoryWithHeaderFile(
      HeaderType Header,
      std::shared_ptr<CallExprRewriterFactoryBase> InnerFactory)
      : Inner(InnerFactory), Header(Header) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) const override {
    DpctGlobalInfo::getInstance().insertHeader(C->getBeginLoc(), Header);
    return Inner->create(C);
  }
};

class RemoveCubTempStorageFactory : public CallExprRewriterFactoryBase {
  std::shared_ptr<CallExprRewriterFactoryBase> Inner;
public:
  RemoveCubTempStorageFactory(std::shared_ptr<CallExprRewriterFactoryBase> InnerFactory)
    : Inner(InnerFactory) {}

  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) const override;
};

class RewriterFactoryWithSubGroupSize : public CallExprRewriterFactoryBase {
  std::shared_ptr<CallExprRewriterFactoryBase> Inner;
  std::function<size_t(const CallExpr *, std::string &)> F;
  std::string Name;

public:
  RewriterFactoryWithSubGroupSize(
      std::function<size_t(const CallExpr *, std::string &)> Method,
      std::string NewFuncName,
      std::shared_ptr<CallExprRewriterFactoryBase> InnerFactory)
      : Inner(InnerFactory), F(Method), Name(NewFuncName) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) const override {
    auto FuncInfo =
        DeviceFunctionDecl::LinkRedecls(DpctGlobalInfo::getParentFunction(C));
    if (FuncInfo) {
      std::string VarNotEvaluated;
      unsigned int Size = F(C, VarNotEvaluated);
      if (Size != UINT_MAX) {
        FuncInfo->addSubGroupSizeRequest(Size, C->getBeginLoc(), Name);
      } else {
        FuncInfo->addSubGroupSizeRequest(Size, C->getBeginLoc(), Name,
                                         VarNotEvaluated);
      }
    }
    return Inner->create(C);
  }
};

/// Base class for rewriting function calls
class FuncCallExprRewriter : public CallExprRewriter {
protected:
  std::string TargetCalleeName;
  std::vector<std::string> RewriteArgList;

protected:
  FuncCallExprRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : CallExprRewriter(Call, SourceCalleeName),
        TargetCalleeName(TargetCalleeName) {}

public:
  virtual ~FuncCallExprRewriter() {}

  virtual std::optional<std::string> rewrite() override;

  friend FuncCallExprRewriterFactory;

protected:
  template <class... Args> void appendRewriteArg(Args &&...Arguments) {
    RewriteArgList.emplace_back(std::forward<Args...>(Arguments)...);
  }

  // Build string which is used to replace original expression.
  std::optional<std::string> buildRewriteString();

  void setTargetCalleeName(const std::string &Str) { TargetCalleeName = Str; }
};

class NoRewriteFuncNameRewriter : public CallExprRewriter {
  std::string NewFuncName;

public:
  NoRewriteFuncNameRewriter(const CallExpr *Call, StringRef SourceName,
                            StringRef NewName)
      : CallExprRewriter(Call, SourceCalleeName) {
    NewFuncName = NewName.str();
    NoRewrite = true;
  }

  std::optional<std::string> rewrite() override { return NewFuncName; }
};

struct ThrustFunctor {
  ThrustFunctor(const clang::Expr *E) : E(E) {}
  const clang::Expr *E;
};

template <class StreamT, class T> void print(StreamT &Stream, const T &Val) {
  Val.print(Stream);
}
template <class StreamT> void print(StreamT &Stream, const Expr *E) {
  ExprAnalysis EA;
  print(Stream, EA, E);
}
template <class StreamT> void print(StreamT &Stream, StringRef Str) {
  Stream << Str;
}
template <class StreamT>
void print(StreamT &Stream,
           const std::pair<std::string, const StringRef> &Pair) {
  Stream << Pair.first << Pair.second;
}
template <class StreamT, class T>
void print(StreamT &Stream, ExprAnalysis &EA, const T &Val) {
  print(Stream, Val);
}
template <class StreamT> void print(StreamT &Stream, const std::string &Str) {
  Stream << Str;
}
template <class StreamT>
void print(StreamT &Stream, const TemplateArgumentInfo &Arg) {
  print(Stream, Arg.getString());
}
template <class StreamT>
void print(StreamT &Stream, const ThrustFunctor &Functor) {
  FunctorAnalysis FA;
  FA.analyze(Functor.E);
  Stream << FA.getRewritePrefix() << FA.getReplacedString()
         << FA.getRewritePostfix();
}
template <class StreamT>
void print(StreamT &Stream, ExprAnalysis &EA, const Expr *E) {
  EA.analyze(E);
  Stream << EA.getRewritePrefix() << EA.getReplacedString()
         << EA.getRewritePostfix();
}

template <class StreamT>
void print(StreamT &Stream, std::pair<const CallExpr *, const Expr *> P) {
  ArgumentAnalysis AA;
  print(Stream, AA, P);
}

template <class StreamT>
void print(StreamT &Stream, ArgumentAnalysis &AA,
           std::pair<const CallExpr *, const Expr *> P) {
  AA.setCallSpelling(P.first);
  AA.analyze(P.second);
  Stream << AA.getRewritePrefix() << AA.getRewriteString()
         << AA.getRewritePostfix();
}

template <class StreamT>
void print(StreamT &Stream, ArgumentAnalysis &AA,
           TypeLoc TL) {
  ExprAnalysis EA;
  EA.analyze(TL);
  Stream << EA.getReplacedString();
}

template <class StreamT, class T1, class T2>
void print(StreamT &Stream, ArgumentAnalysis &AA, std::pair<T1, T2> P) {
  dpct::print(Stream, AA, P.first);
  dpct::print(Stream, AA, P.second);
}

template <class StreamT>
void printWithParens(StreamT &Stream, ExprAnalysis &EA, const Expr *E) {
  std::unique_ptr<ParensPrinter<StreamT>> Paren;
  E = E->IgnoreImplicitAsWritten();
  if (needExtraParens(E))
    Paren = std::make_unique<ParensPrinter<StreamT>>(Stream);
  print(Stream, EA, E);
}
template <class StreamT> void printWithParens(StreamT &Stream, const Expr *E) {
  ExprAnalysis EA;
  printWithParens(Stream, EA, E);
}

template <class StreamT>
void printWithParens(StreamT &Stream, ArgumentAnalysis &AA,
                     std::pair<const CallExpr *, const Expr *> P) {
  std::unique_ptr<ParensPrinter<StreamT>> Paren;
  P.second = P.second->IgnoreImplicitAsWritten();
  if (needExtraParens(P.second))
    Paren = std::make_unique<ParensPrinter<StreamT>>(Stream);
  print(Stream, AA, P);
}

template <class StreamT>
void printWithParens(StreamT &Stream,
                     std::pair<const CallExpr *, const Expr *> P) {
  ArgumentAnalysis AA;
  printWithParens(Stream, AA, P);
}

template <class StreamT> void printMemberOp(StreamT &Stream, bool IsArrow) {
  if (IsArrow)
    Stream << "->";
  else
    Stream << ".";
}

template <class StreamT>
void printCapture(StreamT &Stream, bool IsCaptureRef) {
  if (IsCaptureRef)
    Stream << "[&]";
  else
    Stream << "[=]";
}

class AddrOfExpr {
  bool DerefRemoved = false, NeedParens = false;
  const Expr *E = nullptr;
  const CallExpr *C = nullptr;

  template <class StreamT>
  void print(StreamT &Stream, ExprAnalysis &EA, bool IgnoreDerefOp) const {
    if (!DerefRemoved && !IgnoreDerefOp)
      Stream << "&";

    if (NeedParens) {
      printWithParens(Stream, EA, E);
    } else {
      dpct::print(Stream, EA, E);
    }
  }

  template <class StreamT>
  void print(StreamT &Stream, ArgumentAnalysis &AA, bool IgnoreDerefOp,
             std::pair<const CallExpr *, const Expr *> P) const {
    if (!DerefRemoved && !IgnoreDerefOp)
      Stream << "&";

    AA.setCallSpelling(C);
    if (NeedParens) {
      printWithParens(Stream, AA, P);
    } else {
      dpct::print(Stream, AA, P);
    }
  }

public:
  AddrOfExpr(const Expr *E, const CallExpr *C = nullptr);
  template <class StreamT>
  void printArg(StreamT &Stream, ArgumentAnalysis &A) const {
    print(Stream);
  }
  template <class StreamT> void printMemberBase(StreamT &Stream) const {
    ExprAnalysis EA;
    print(Stream, EA, true);
    printMemberOp(Stream, !DerefRemoved);
  }

  template <class StreamT> void print(StreamT &Stream) const {
    if (C == nullptr) {
      ExprAnalysis EA;
      print(Stream, EA, false);
    } else {
      ArgumentAnalysis AA;
      std::pair<const CallExpr *, const Expr *> ExprPair(C, E);
      print(Stream, AA, false, ExprPair);
    }
  }
};

class DerefExpr {
  bool AddrOfRemoved = false, NeedParens = false;
  const Expr *E = nullptr;
  const CallExpr * C = nullptr;

  template <class StreamT>
  void print(StreamT &Stream, ExprAnalysis &EA, bool IgnoreDerefOp) const {
    if (!AddrOfRemoved && !IgnoreDerefOp)
      Stream << "*";

    if (NeedParens) {
      printWithParens(Stream, EA, E);
    } else {
      dpct::print(Stream, EA, E);
    }
  }

  template <class StreamT>
  void print(StreamT &Stream, ArgumentAnalysis &AA, bool IgnoreDerefOp,
              std::pair<const CallExpr *, const Expr *> P) const {
    if (!AddrOfRemoved && !IgnoreDerefOp)
      Stream << "*";

    AA.setCallSpelling(C);
    if (NeedParens) {
      printWithParens(Stream, AA, P);
    } else {
      dpct::print(Stream, AA, P);
    }
  }

public:
  DerefExpr(const Expr *E, const CallExpr *C = nullptr);
  template <class StreamT>
  void printArg(StreamT &Stream, ArgumentAnalysis &A) const {
    print(Stream);
  }
  template <class StreamT> void printMemberBase(StreamT &Stream) const {
    ExprAnalysis EA;
    print(Stream, EA, true);
    printMemberOp(Stream, !AddrOfRemoved);
  }

  template <class StreamT> void print(StreamT &Stream) const {
    if (C == nullptr) {
      ExprAnalysis EA;
      print(Stream, EA, false);
    } else {
      ArgumentAnalysis AA;

      std::pair<const CallExpr*, const Expr*> ExprPair(C, E);
      print(Stream, AA, false, ExprPair);
    }
  }
};

template <class StreamT>
void print(StreamT &Stream,
           std::pair<const llvm::StringRef, clang::dpct::DerefExpr> Pair) {
  Stream << Pair.first;
  ArgumentAnalysis AA;
  Pair.second.printArg(Stream, AA);
}
template <class StreamT>
void print(StreamT &Stream,
           std::pair<std::pair<const llvm::StringRef, clang::dpct::DerefExpr>,
                     const llvm::StringRef>
               Pair) {
  print(Stream, Pair.first);
  Stream << Pair.second;
}

class RenameWithSuffix {
  StringRef OriginalName, SuffixStr;

public:
  RenameWithSuffix(StringRef Original, StringRef Suffix)
      : OriginalName(Original), SuffixStr(Suffix) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Stream << OriginalName;
    if (!SuffixStr.empty())
      Stream << "_" << SuffixStr;
  }
};

template <bool HasPrefixArg, class... ArgsT> class ArgsPrinter;
template <bool HasPrefixArg> class ArgsPrinter<HasPrefixArg> {
  mutable ArgumentAnalysis A;

public:
  template <class StreamT> void print(StreamT &) const {}
  template <class StreamT>
  void printArg(std::false_type, StreamT &Stream, const Expr *E) const {
    if(auto defaultArg = dyn_cast<CXXDefaultArgExpr>(E)){
      E = defaultArg->getExpr();
    }
    dpct::print(Stream, A, E);
  }

  template <class StreamT>
  void printArg(std::false_type, StreamT &Stream,
                std::pair<const CallExpr *, const Expr *> P) const {
    dpct::print(Stream, A, P);
  }

  template <class StreamT>
  void printArg(std::false_type, StreamT &Stream,
                TypeLoc TL) const {
    dpct::print(Stream, A, TL);
  }

  template <class StreamT, class T1, class T2>
  void printArg(std::false_type, StreamT &Stream, std::pair<T1, T2> P) const {
    dpct::print(Stream, A, P);
  }

  template <class StreamT>
  void printArg(std::false_type, StreamT &Stream, DerefExpr Arg) const {
    Arg.printArg(Stream, A);
  }
  template <class StreamT, class ArgT>
  void printArg(std::false_type, StreamT &Stream, const ArgT &Arg) const {
    dpct::print(Stream, Arg);
  }
  template <class StreamT, class ArgT>
  void printArg(std::false_type, StreamT &Stream,
                const std::vector<ArgT> &Vec) const {
    if (Vec.empty())
      return;
    auto Itr = Vec.begin();
    printArg(std::false_type(), Stream, *Itr);
    while (++Itr != Vec.end()) {
      printArg(std::true_type(), Stream, *Itr);
    }
  }
  template <class StreamT, class ArgT>
  void printArg(std::true_type, StreamT &Stream,
                const std::vector<ArgT> &Vec) const {
    for (auto &Arg : Vec) {
      printArg(std::true_type(), Stream, Arg);
    }
  }
  template <class StreamT, class ArgT>
  void printArg(std::true_type, StreamT &Stream, ArgT &&Arg) const {
    Stream << ", ";
    printArg(std::false_type(), Stream, std::forward<ArgT>(Arg));
  }

  ArgsPrinter() = default;
  ArgsPrinter(const ArgsPrinter &) {}
};
template <bool HasPrefixArg, class FirstArgT, class... RestArgsT>
class ArgsPrinter<HasPrefixArg, FirstArgT, RestArgsT...>
    : public ArgsPrinter<true, RestArgsT...> {
  using Base = ArgsPrinter<true, RestArgsT...>;
  FirstArgT First;

public:
  template <class InputFirstArgT, class... InputRestArgsT>
  ArgsPrinter(InputFirstArgT &&FirstArg, InputRestArgsT &&...RestArgs)
      : Base(std::forward<InputRestArgsT>(RestArgs)...),
        First(std::forward<InputFirstArgT>(FirstArg)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Base::printArg(std::integral_constant<bool, HasPrefixArg>(), Stream, First);
    Base::print(Stream);
  }
};

template <class StreamT>
void printBase(StreamT &Stream, std::pair<const CallExpr *, const Expr *> P,
               bool IsArrow) {
  printWithParens(Stream, P);
  printMemberOp(Stream, IsArrow);
}

template <class StreamT>
void printBase(StreamT &Stream, const Expr *E, bool IsArrow) {
  printWithParens(Stream, E);
  printMemberOp(Stream, IsArrow);
}
template <class StreamT>
void printBase(StreamT &Stream, const DerefExpr &D, bool) {
  D.printMemberBase(Stream);
}
template <class StreamT, class T>
void printBase(StreamT &Stream, const T &Val, bool IsArrow) {
  print(Stream, Val);
  printMemberOp(Stream, IsArrow);
}

template <class CalleeT, class... CallArgsT> class CallExprPrinter {
  CalleeT Callee;
  ArgsPrinter<false, CallArgsT...> Args;

public:
  CallExprPrinter(const CalleeT &Callee, CallArgsT &&...Args)
      : Callee(Callee), Args(std::forward<CallArgsT>(Args)...) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Callee);
    ParensPrinter<StreamT> Parens(Stream);
    Args.print(Stream);
  }
};

template <class NameT, class... TemplateArgsT> class TemplatedNamePrinter {
  NameT Name;
  ArgsPrinter<false, TemplateArgsT...> TAs;
public:
  TemplatedNamePrinter(NameT Name, TemplateArgsT &&...TAs)
      : Name(Name), TAs(std::forward<TemplateArgsT>(TAs)...) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Name);
    Stream << "<";
    TAs.print(Stream);
    Stream << ">";
  }
};

template <class NameT, class... TemplateArgsT> class CtadTemplatedNamePrinter {
  NameT Name;
  ArgsPrinter<false, TemplateArgsT...> TAs;
public:
  CtadTemplatedNamePrinter(NameT Name, TemplateArgsT &&...TAs)
      : Name(Name), TAs(std::forward<TemplateArgsT>(TAs)...) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Name);
    if (!DpctGlobalInfo::isCtadEnabled()) {
      Stream << "<";
      TAs.print(Stream);
      Stream << ">";
    }
  }
};

// Print a type with no template.
template <class NameT> class TypeNamePrinter {
  NameT Name;

public:
  TypeNamePrinter(NameT Name) : Name(Name) {}

  template <typename StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Name);
  }
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

template <class BaseT, class MemberT> class StaticMemberExprPrinter {
  BaseT Base;
  MemberT Member;
public:
  StaticMemberExprPrinter(BaseT &&Base, MemberT &&Member)
    : Base(std::forward<BaseT>(Base)), Member(std::forward<MemberT>(Member)) {}

  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Base);
    Stream << "::";
    dpct::print(Stream, Member);
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

template <class BaseT, class ArgValueT> class ArraySubscriptExprPrinter {
  BaseT Base;
  ArgValueT ArgValue;

public:
  ArraySubscriptExprPrinter(BaseT base, ArgValueT &&Arg)
      : Base(base), ArgValue(std::forward<ArgValueT>(Arg)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Base);
    Stream << "[";
    dpct::print(Stream, ArgValue);
    Stream << "]";
  }
};

template <class TypeInfoT, class SubExprT> class CastExprPrinter {
  TypeInfoT TypeInfo;
  SubExprT SubExpr;
  bool ExtraParen;

public:
  CastExprPrinter(TypeInfoT &&T, SubExprT &&S, bool EP = false)
      : TypeInfo(std::forward<TypeInfoT>(T)),
        SubExpr(std::forward<SubExprT>(S)), ExtraParen(EP) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Stream << "(";
    dpct::print(Stream, TypeInfo);
    Stream << ")";
    if(ExtraParen)
      Stream << "(";
    dpct::print(Stream, SubExpr);
    if(ExtraParen)
      Stream << ")";
  }
};

template <BinaryOperatorKind Op, class LValueT, class RValueT>
class BinaryOperatorPrinter {
  LValueT LVal;
  RValueT RVal;

  static std::string OpStr;

public:
  BinaryOperatorPrinter(LValueT &&L, RValueT &&R)
      : LVal(std::forward<LValueT>(L)), RVal(std::forward<RValueT>(R)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, LVal);
    Stream << " " << OpStr << " ";
    dpct::print(Stream, RVal);
  }
};

template <UnaryOperatorKind UO, class ArgValueT>
class UnaryOperatorPrinter {
  ArgValueT ArgValue;

  static std::string UOStr;

public:
  UnaryOperatorPrinter(ArgValueT &&Arg)
      : ArgValue(std::forward<ArgValueT>(Arg)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Stream << UOStr;
    dpct::print(Stream, ArgValue);
  }
};
template <UnaryOperatorKind UO, class ArgValueT>
std::string UnaryOperatorPrinter<UO, ArgValueT>::UOStr =
    UnaryOperator::getOpcodeStr(UO).str();

template <BinaryOperatorKind Op, class LValueT, class RValueT>
std::string BinaryOperatorPrinter<Op, LValueT, RValueT>::OpStr =
    BinaryOperator::getOpcodeStr(Op).str();

template <class LValueT, class RValueT>
using AssignExprPrinter =
    BinaryOperatorPrinter<BinaryOperatorKind::BO_Assign, LValueT, RValueT>;

template <class ArgT> class DeleterCallExprRewriter : public CallExprRewriter {
  ArgT Arg;

public:
  DeleterCallExprRewriter(const CallExpr *C, StringRef Source,
                          std::function<ArgT(const CallExpr *)> ArgCreator)
      : CallExprRewriter(C, Source), Arg(ArgCreator(C)) {}
  std::optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    OS << "delete ";
    printWithParens(OS, Arg);
    return Result;
  }
};

template <class ArgT> class ToStringExprRewriter : public CallExprRewriter {
  ArgT Arg;

public:
  ToStringExprRewriter(const CallExpr *C, StringRef Source,
                       std::function<ArgT(const CallExpr *)> ArgCreator)
      : CallExprRewriter(C, Source), Arg(ArgCreator(C)) {}
  std::optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    print(OS, Arg);
    return Result;
  }
};

template <class... ArgsT>
class NewExprPrinter : CallExprPrinter<StringRef, ArgsT...> {
  using Base = CallExprPrinter<StringRef, ArgsT...>;

public:
  NewExprPrinter(StringRef TypeName, ArgsT &&...Args)
      : Base(TypeName, std::forward<ArgsT>(Args)...) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Stream << "new ";
    Base::print(Stream);
  }
};

template<class SubExprT>
class TypenameExprPrinter {
  SubExprT SubExpr;
public:
  TypenameExprPrinter(SubExprT &&SubExpr) : SubExpr(std::forward<SubExprT>(SubExpr)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Stream << "typename ";
    dpct::print(Stream, SubExpr);
  }
};

template <class SubExprT> class ZeroInitializerPrinter {
  SubExprT SubExpr;

public:
  ZeroInitializerPrinter(SubExprT &&SubExpr)
      : SubExpr(std::forward<SubExprT>(SubExpr)) {}
  template <typename StreamT> void print(StreamT &&Stream) const {
    dpct::print(Stream, SubExpr);
    Stream << "{}";
  }
};

template <class FirstPrinter, class... RestPrinter>
class MultiStmtsPrinter : public MultiStmtsPrinter<RestPrinter...> {
  using Base = MultiStmtsPrinter<RestPrinter...>;
  FirstPrinter First;

public:
  MultiStmtsPrinter(SourceRange Range, SourceManager &SM, FirstPrinter &&First,
                    RestPrinter &&...Rest)
      : Base(Range, SM, std::move(Rest)...), First(std::move(First)) {}

  MultiStmtsPrinter(FirstPrinter &&First, RestPrinter &&...Rest)
      : Base(std::move(Rest)...), First(std::move(First)) {}

  template <class StreamT> void print(StreamT &Stream) const {
    Base::printStmt(Stream, First);
    Base::print(Stream);
  }
};

template <class LastPrinter> class MultiStmtsPrinter<LastPrinter> {
  LastPrinter Last;
  StringRef Indent;
  StringRef NL;
  bool isInMacroDef;

protected:
  template <class StreamT, class PrinterT>
  void printStmt(StreamT &Stream, const PrinterT &Printer) const {
    dpct::print(Stream, Printer);
    if (isInMacroDef) {
      Stream << "; \\" << NL << Indent;
    } else {
      Stream << "; " << NL << Indent;
    }
  }

public:
  MultiStmtsPrinter(SourceRange Range, SourceManager &SM, LastPrinter &&Last)
      : Last(std::move(Last)), Indent(getIndent(Range.getBegin(), SM)),
        NL(getNL(Range.getBegin(), SM)),
        isInMacroDef(isInMacroDefinition(Range.getBegin(), Range.getBegin()) &&
                     isInMacroDefinition(Range.getEnd(), Range.getEnd())) {}

  MultiStmtsPrinter(LastPrinter &&Last)
      : Last(std::move(Last)), Indent(" "), NL(""), isInMacroDef(false) {}

  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Last);
  }
};

template <class... StmtPrinter>
class LambdaPrinter {
  bool IsCaptureRef;
  MultiStmtsPrinter<StmtPrinter...> MultiStmts;

public:
  LambdaPrinter(bool IsCaptureRef, StmtPrinter &&...Printer)
      : IsCaptureRef(IsCaptureRef), MultiStmts(std::move(Printer)...) {}

  template <class StreamT> void print(StreamT &Stream) const {
    printCapture(Stream, IsCaptureRef);
    Stream << "()";
    CurlyBracketsPrinter<StreamT> CurlyBracket(Stream);
    MultiStmts.print(Stream);
    Stream << ";";
  }
};

template <class FirstPrinter, class... RestPrinter>
class CommaExprPrinter : CommaExprPrinter<RestPrinter...> {
  using Base = CommaExprPrinter<RestPrinter...>;
  FirstPrinter First;

public:
  CommaExprPrinter(FirstPrinter &&First, RestPrinter &&...Rest)
      : Base(std::move(Rest)...), First(std::move(First)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, First);
    Stream << ", ";
    Base::print(Stream);
  }
};

template <class LastPrinter> class CommaExprPrinter<LastPrinter> {
  LastPrinter Last;

public:
  CommaExprPrinter(LastPrinter &&Last) : Last(std::move(Last)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Last);
  }
};

template <class Printer>
class PrinterRewriter : Printer, public CallExprRewriter {
public:
  template <class... ArgsT>
  PrinterRewriter(const CallExpr *C, StringRef Source, ArgsT &&...Args)
      : Printer(std::forward<ArgsT>(Args)...), CallExprRewriter(C, Source) {}
  template <class... ArgsT>
  PrinterRewriter(const CallExpr *C, StringRef Source,
                  const std::function<ArgsT(const CallExpr *)> &...ArgCreators)
      : PrinterRewriter(C, Source, ArgCreators(C)...) {}
  std::optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Printer::print(OS);
    return OS.str();
  }
};

template <class... StmtPrinters>
class PrinterRewriter<MultiStmtsPrinter<StmtPrinters...>>
    : MultiStmtsPrinter<StmtPrinters...>, public CallExprRewriter {
  using Base = MultiStmtsPrinter<StmtPrinters...>;

public:
  PrinterRewriter(const CallExpr *C, StringRef Source,
                  StmtPrinters &&...Printers)
      : Base(getDefinitionRange(C->getBeginLoc(), C->getEndLoc()),
             DpctGlobalInfo::getSourceManager(), std::move(Printers)...),
        CallExprRewriter(C, Source) {}
  PrinterRewriter(
      const CallExpr *C, StringRef Source,
      const std::function<StmtPrinters(const CallExpr *)> &...PrinterCreators)
      : PrinterRewriter(C, Source, PrinterCreators(C)...) {}
  std::optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Base::print(OS);
    return OS.str();
  }
};

template <class BaseT, class ArgValueT>
class ArraySubscriptRewriter
    : public PrinterRewriter<ArraySubscriptExprPrinter<BaseT, ArgValueT>> {
public:
  ArraySubscriptRewriter(
      const CallExpr *C, const std::string &SourceName,
      const std::function<BaseT(const CallExpr *)> &BaseCreator,
      const std::function<ArgValueT(const CallExpr *)> &ArgCreator)
      : PrinterRewriter<ArraySubscriptExprPrinter<BaseT, ArgValueT>>(
            C, SourceName, BaseCreator(C), ArgCreator(C)) {}
};

template <class... ArgsT>
class TemplatedCallExprRewriter
    : public PrinterRewriter<CallExprPrinter<
          TemplatedNamePrinter<StringRef, std::vector<TemplateArgumentInfo>>,
          ArgsT...>> {
public:
  TemplatedCallExprRewriter(
      const CallExpr *C, StringRef Source,
      const std::function<
          TemplatedNamePrinter<StringRef, std::vector<TemplateArgumentInfo>>(
              const CallExpr *)> &CalleeCreator,
      const std::function<ArgsT(const CallExpr *)> &...ArgsCreator)
      : PrinterRewriter<CallExprPrinter<
            TemplatedNamePrinter<StringRef, std::vector<TemplateArgumentInfo>>,
            ArgsT...>>(C, Source, CalleeCreator(C), ArgsCreator(C)...) {}
};

template <class BaseT, class MemberT>
class MemberExprRewriter
    : public PrinterRewriter<MemberExprPrinter<BaseT, MemberT>> {
public:
  MemberExprRewriter(
      const CallExpr *C, StringRef Source,
      const std::function<BaseT(const CallExpr *)> &BaseCreator, bool IsArrow,
      const std::function<MemberT(const CallExpr *)> &MemberCreator)
      : PrinterRewriter<MemberExprPrinter<BaseT, MemberT>>(
            C, Source, BaseCreator(C), IsArrow, MemberCreator(C)) {}
};

template <class BaseT, class... ArgsT>
class MemberCallExprRewriter
    : public PrinterRewriter<MemberCallPrinter<BaseT, StringRef, ArgsT...>> {
public:
  MemberCallExprRewriter(
      const CallExpr *C, StringRef Source,
      const std::function<BaseT(const CallExpr *)> &BaseCreator, bool IsArrow,
      StringRef Member,
      const std::function<ArgsT(const CallExpr *)> &...ArgsCreator)
      : PrinterRewriter<MemberCallPrinter<BaseT, StringRef, ArgsT...>>(
            C, Source, BaseCreator(C), IsArrow, Member, ArgsCreator(C)...) {}
  MemberCallExprRewriter(
      const CallExpr *C, StringRef Source, const BaseT &BaseCreator,
      bool IsArrow, StringRef Member,
      const std::function<ArgsT(const CallExpr *)> &...ArgsCreator)
      : PrinterRewriter<MemberCallPrinter<BaseT, StringRef, ArgsT...>>(
            C, Source, BaseCreator, IsArrow, Member, ArgsCreator(C)...) {}
};

template <class CalleeT, class... ArgsT>
class SimpleCallExprRewriter : public CallExprRewriter {
  CallExprPrinter<CalleeT, ArgsT...> Printer;

public:
  SimpleCallExprRewriter(
      const CallExpr *C, StringRef Source,
      const std::function<CallExprPrinter<CalleeT, ArgsT...>(const CallExpr *)>
          &PrinterFunctor)
      : CallExprRewriter(C, Source), Printer(PrinterFunctor(C)) {}
  std::optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Printer.print(OS);
    return OS.str();
  }
};

template <BinaryOperatorKind BO, class LValueT, class RValueT>
class BinaryOpRewriter
    : public PrinterRewriter<BinaryOperatorPrinter<BO, LValueT, RValueT>> {
public:
  BinaryOpRewriter(const CallExpr *C, StringRef Source,
                   const std::function<LValueT(const CallExpr *)> &LCreator,
                   const std::function<RValueT(const CallExpr *)> &RCreator)
      : PrinterRewriter<BinaryOperatorPrinter<BO, LValueT, RValueT>>(
            C, Source, LCreator(C), RCreator(C)) {}
};

template <UnaryOperatorKind UO, class ArgValueT>
class UnaryOpRewriter
    : public PrinterRewriter<UnaryOperatorPrinter<UO, ArgValueT>> {
public:
  UnaryOpRewriter(const CallExpr *C, StringRef Source,
                   const std::function<ArgValueT(const CallExpr *)> &ArgCreator)
      : PrinterRewriter<UnaryOperatorPrinter<UO, ArgValueT>> (
            C, Source, ArgCreator(C)) {}
};

template <class ArgValueT>
class DerefExprRewriter : public PrinterRewriter<DerefExpr> {
public:
  DerefExprRewriter(
      const CallExpr *C, StringRef Source,
      const std::function<ArgValueT(const CallExpr *)> &ArgCreator)
      : PrinterRewriter<dpct::DerefExpr>(C, Source, ArgCreator(C)) {}
};

class SubGroupPrinter {
  const CallExpr *Call;
public:
  SubGroupPrinter(const CallExpr *C) : Call(C) {}
  static SubGroupPrinter create(const CallExpr *C) {
    return SubGroupPrinter(C);
  }
  template <class StreamT> void print(StreamT &Stream) const {
    DpctGlobalInfo::printSubGroup(Stream, Call);
  }
};

class ItemPrinter {
  const CallExpr *Call;
public:
  ItemPrinter(const CallExpr *C) : Call(C) {}
  static ItemPrinter create(const CallExpr *C) { return ItemPrinter(C);
  }
  template <class StreamT> void print(StreamT &Stream) const {
    DpctGlobalInfo::printItem(Stream, Call);
  }
};

class GroupPrinter {
  const CallExpr *Call;
public:
  GroupPrinter(const CallExpr *C) : Call(C) {}
  static GroupPrinter create(const CallExpr *C) { return GroupPrinter(C); }
  template <class StreamT> void print(StreamT &Stream) const {
    DpctGlobalInfo::printGroup(Stream, Call);
  }
};

template <class... MsgArgs>
class UnsupportFunctionRewriter : public CallExprRewriter {
  template <class T>
  std::string getMsgArg(const std::function<T(const CallExpr *)> &Func,
                        const CallExpr *C) {
    return getMsgArg(Func(C), C);
  }
  template <class T>
  static std::string getMsgArg(const T &InputArg, const CallExpr *) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    print(OS, InputArg);
    return OS.str();
  }

public:
  UnsupportFunctionRewriter(const CallExpr *CE, StringRef CalleeName,
                            Diagnostics MsgID, const MsgArgs &...Args)
      : CallExprRewriter(CE, CalleeName) {
    report(MsgID, false, getMsgArg(Args, CE)...);
  }

  std::optional<std::string> rewrite() override { return std::nullopt; }

  friend UnsupportFunctionRewriterFactory<MsgArgs...>;
};

std::function<std::string(const CallExpr *)> makeQueueStr();
std::function<std::pair<const CallExpr *, const Expr *>(const CallExpr *)>
makeCallArgCreatorWithCall(unsigned Idx);
std::function<DerefExpr(const CallExpr *)> makeDerefExprCreator(unsigned Idx);
std::function<std::string(const CallExpr *C)> getReplacedType(size_t Idx);
std::function<std::string(const CallExpr *C)> getDerefedType(size_t Idx);
std::function<std::string(const CallExpr *)> makeDeviceStr();

class UserDefinedRewriter : public CallExprRewriter {
  std::string ResultStr;

public:
  UserDefinedRewriter(const CallExpr *CE, const OutputBuilder &OB,
                      const MetaRuleObject::Attributes &RuleAttributes = {})
      : CallExprRewriter(CE, "") {
    NoRewrite = RuleAttributes.ReplaceCalleeNameOnly;
    // build result string with call
    llvm::raw_string_ostream OS(ResultStr);
    buildRewriterStr(Call, OS, OB);
    OS.flush();
  }
  std::optional<std::string> rewrite() override {
    return ResultStr;
  }

  void buildRewriterStr(const CallExpr *Call, llvm::raw_string_ostream &OS,
                        const OutputBuilder &OB) {
    switch (OB.Kind) {
    case (OutputBuilder::Kind::Top):
      for (auto &ob : OB.SubBuilders) {
        buildRewriterStr(Call, OS, *ob);
      }
      return;
    case (OutputBuilder::Kind::String):
      OS << OB.Str;
      return;
    case (OutputBuilder::Kind::Arg): {
      if (OB.ArgIndex >= Call->getNumArgs()) {
        OS << "";
        return;
      }
      ArgumentAnalysis AA;
      AA.setCallSpelling(Call);
      AA.analyze(Call->getArg(OB.ArgIndex));
      OS << AA.getRewriteString();
      return;
    }
    case (OutputBuilder::Kind::Queue): {
      OS << makeQueueStr()(Call);
      return;
    }
    case (OutputBuilder::Kind::Context):
      OS << MapNames::getDpctNamespace() << "get_default_context()";
      return;
    case (OutputBuilder::Kind::Device): {
      OS << makeDeviceStr()(Call);
      return;
    }
    case (OutputBuilder::Kind::Deref): {
      makeDerefExprCreator(OB.ArgIndex)(Call).print(OS);
      return;
    }
    case (OutputBuilder::Kind::TypeName): {
      OS << getReplacedType(OB.ArgIndex)(Call);
      return;
    }
    case (OutputBuilder::Kind::AddrOf): {
      if (OB.ArgIndex >= Call->getNumArgs()) {
        OS << "";
        return;
      }
      OS << "&(";
      ArgumentAnalysis AA;
      AA.setCallSpelling(Call);
      AA.analyze(Call->getArg(OB.ArgIndex));
      OS << AA.getRewriteString();
      OS << ")";
      return;
    }
    case (OutputBuilder::Kind::DerefedTypeName): {
      OS << getDerefedType(OB.ArgIndex)(Call);
      return;
    }
    }
    DpctDebugs() << "[OutputBuilder::Kind] Unexpected value: " << OB.Kind
                 << "\n";
    assert(0);
  }
};

class UserDefinedRewriterFactory : public CallExprRewriterFactoryBase {
  // Information for building the result string from the original function call
  OutputBuilder OB;
  std::string OutStr;
  std::vector<std::string> &Includes;
  MetaRuleObject::Attributes RuleAttributes;

  struct NullRewriter : public CallExprRewriter {
    NullRewriter(const CallExpr *C, StringRef Name)
        : CallExprRewriter(C, Name) {}

    std::optional<std::string> rewrite() override { return std::nullopt; }
  };

public:
  static bool hasExplicitTemplateArgs(const CallExpr *C) {
    auto Callee = C->getCallee();
    if (!Callee)
      return false;

    Callee = Callee->IgnoreImpCasts();
    if (auto DRE = clang::dyn_cast<DeclRefExpr>(Callee))
      return DRE->hasExplicitTemplateArgs();

    return false;
  }

public:
  UserDefinedRewriterFactory(MetaRuleObject &R)
      : OutStr(R.Out), Includes(R.Includes),
        RuleAttributes(R.RuleAttributes) {
    Priority = R.Priority;
    OB.Kind = OutputBuilder::Kind::Top;
    OB.RuleName = R.RuleId;
    OB.RuleFile = R.RuleFile;
    OB.parse(OutStr);
  }

  UserDefinedRewriterFactory(
      MetaRuleObject &R, std::shared_ptr<MetaRuleObject::ClassMethod> MethodPtr)
      : OutStr(MethodPtr->Out), Includes(R.Includes) {
    Priority = R.Priority;
    OB.Kind = OutputBuilder::Kind::Top;
    OB.RuleName = R.RuleId;
    OB.parse(OutStr);
  }

  std::shared_ptr<CallExprRewriter>
  create(const CallExpr *Call) const override {
    if (!Call)
      return std::shared_ptr<UserDefinedRewriter>();

    if (hasExplicitTemplateArgs(Call) && !RuleAttributes.HasExplicitTemplateArgs)
      return std::make_shared<NullRewriter>(Call, "");

    for (auto &Header : Includes)
      DpctGlobalInfo::getInstance().insertHeader(Call->getBeginLoc(), Header);

    return std::make_shared<UserDefinedRewriter>(Call, OB, RuleAttributes);
  }
};

std::shared_ptr<CallExprRewriterFactoryBase>
createUserDefinedRewriterFactory(const std::string &, MetaRuleObject &);
std::shared_ptr<CallExprRewriterFactoryBase>
createUserDefinedMethodRewriterFactory(
    const std::string &, MetaRuleObject &,
    std::shared_ptr<MetaRuleObject::ClassMethod>);

class CheckParamType {
  unsigned Idx;
  std::string TypeName;

public:
  CheckParamType(unsigned I, std::string Name) : Idx(I), TypeName(Name) {}
  bool operator()(const CallExpr *C) {
    std::string ParamType = getParamTypeStr(C, Idx);
    if (ParamType.empty())
      return true;
    return ParamType.find(TypeName) != std::string::npos;
  }
};

class CheckArgType {
  unsigned Idx;
  std::string TypeName;

public:
  CheckArgType(unsigned I, std::string Name) : Idx(I), TypeName(Name) {}
  bool operator()(const CallExpr *C) {
    std::string ArgType = getArgTypeStr(C, Idx);
    if (ArgType.empty())
      return true;
    return ArgType.find(TypeName) != std::string::npos;
  }
};

class CheckMemberBaseType {
  std::string TypeName;

public:
  CheckMemberBaseType(std::string Name)
    : TypeName(std::move(Name)) {}
  bool operator()(const CallExpr *C) {
    std::string ArgType = getBaseTypeStr(C);
    if (ArgType.empty())
      return true;
    return ArgType.find(TypeName) != std::string::npos;
  }
};

class CheckEnumArgStr {
  unsigned Idx;
  std::string EnumArgValueStr;

public:
  CheckEnumArgStr(unsigned I, const std::string &EnumArgValue)
      : Idx(I), EnumArgValueStr(EnumArgValue) {}
  bool operator()(const CallExpr *C) {
    if (C->getNumArgs() <= Idx)
      return false;

    if (auto DRE = dyn_cast<DeclRefExpr>(C->getArg(Idx))) {
      if (auto ECD = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
        return ECD->getNameAsString() == EnumArgValueStr;
      }
    }
    return false;
  }
};

template<class BO>
class CheckIntergerTemplateArgValue {
  unsigned Idx;
  std::int64_t CompareValue;

public:
  CheckIntergerTemplateArgValue(unsigned int Idx, std::int64_t CompareValue)
      : Idx(Idx), CompareValue(CompareValue) {}
  bool getTemplateArgAsInt64(const CallExpr *C, std::int64_t &Val) {
    const FunctionDecl *FD = C->getDirectCallee();
    if (!FD)
      return false;
    auto TSA = FD->getTemplateSpecializationArgs();
    if (!TSA)
      return false;
    if (Idx >= TSA->size())
      return false;
    if (TSA->get(Idx).getKind() != clang::TemplateArgument::ArgKind::Integral)
      return false;
    auto I = TSA->get(Idx).getAsIntegral();
    if (I.getSignificantBits() > 64)
      return false;
    Val = I.getExtValue();
    return true;
  }
  bool operator()(const CallExpr *C) {
    std::int64_t Val = 0;
    bool Res = getTemplateArgAsInt64(C, Val);
    return Res && BO()(Val, CompareValue);
  }
};

using CheckIntergerTemplateArgValueNE = CheckIntergerTemplateArgValue<std::not_equal_to<std::int64_t>>;
using CheckIntergerTemplateArgValueLE = CheckIntergerTemplateArgValue<std::less_equal<std::int64_t>>;

std::function<bool(const CallExpr *C)> hasManagedAttr(int Idx);

} // namespace dpct
} // namespace clang

#endif // !__CALL_EXPR_REWRITER_H__
