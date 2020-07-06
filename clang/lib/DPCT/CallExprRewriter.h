//===--- CallExprRewriter.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2019 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

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
class ReorderFunctionRewriter;
class UnsupportFunctionRewriter;
class TemplatedCallExprRewriter;

/*
Factory usage example:
using BinaryOperatorExprRewriterFactory =
    CallExprRewriterFactory<BinaryOperatorExprRewriter, BinaryOperatorKind>;
*/
/// Base class in abstract factory pattern
class CallExprRewriterFactoryBase {
public:
  virtual std::shared_ptr<CallExprRewriter> create(const CallExpr *) = 0;
  virtual ~CallExprRewriterFactoryBase() {}

  static const std::unordered_map<std::string,
                                  std::shared_ptr<CallExprRewriterFactoryBase>>
      RewriterMap;
};

/// Abstract factory for all rewriter factories
template <class RewriterTy, class... Args>
class CallExprRewriterFactory : public CallExprRewriterFactoryBase {
  std::tuple<std::string, Args...> Initializer;

private:
  template <size_t... Idx>
  inline std::shared_ptr<CallExprRewriter>
  createRewriter(const CallExpr *Call, std::index_sequence<Idx...>) {
    return std::shared_ptr<RewriterTy>(
        new RewriterTy(Call, std::get<Idx>(Initializer)...));
  }

public:
  CallExprRewriterFactory(StringRef SourceCalleeName, Args... Arguments)
      : Initializer(SourceCalleeName.str(), std::forward<Args>(Arguments)...) {}
  // Create a meaningful rewriter only if the CallExpr is not nullptr
  std::shared_ptr<CallExprRewriter> create(const CallExpr *Call) override {
    if (!Call)
      return std::shared_ptr<CallExprRewriter>();
    return createRewriter(Call,
                          std::index_sequence_for<std::string, Args...>());
  }
};

using FuncCallExprRewriterFactory =
    CallExprRewriterFactory<FuncCallExprRewriter, std::string>;
using MathFuncNameRewriterFactory =
    CallExprRewriterFactory<MathFuncNameRewriter, std::string>;
using MathUnsupportedRewriterFactory =
    CallExprRewriterFactory<MathUnsupportedRewriter, std::string>;
using MathSimulatedRewriterFactory =
    CallExprRewriterFactory<MathSimulatedRewriter, std::string>;
using MathTypeCastRewriterFactory =
    CallExprRewriterFactory<MathTypeCastRewriter, std::string>;
using MathBinaryOperatorRewriterFactory =
    CallExprRewriterFactory<MathBinaryOperatorRewriter, BinaryOperatorKind>;
using WarpFunctionRewriterFactory =
    CallExprRewriterFactory<WarpFunctionRewriter, std::string>;
using ReorderFunctionRewriterFactory = CallExprRewriterFactory<
    ReorderFunctionRewriter, std::string,
    std::vector<unsigned> /*Rewrite arguments index list in-order*/>;
using UnsupportFunctionRewriterFactory =
    CallExprRewriterFactory<UnsupportFunctionRewriter, Diagnostics>;
using TemplatedCallExprRewriterFactory =
    CallExprRewriterFactory<TemplatedCallExprRewriter, std::string>;

/// Base class for rewriting call expressions
class CallExprRewriter {
protected:
  // Call is guaranteed not to be nullptr
  const CallExpr *Call;
  StringRef SourceCalleeName;
  ArgumentAnalysis Analyzer;

protected:
  // All instances of the subclasses can only be constructed by corresponding
  // factories. As a result, the access modifiers of the constructors are
  // supposed to be protected instead of public.
  CallExprRewriter(const CallExpr *Call, StringRef SourceCalleeName)
      : Call(Call), SourceCalleeName(SourceCalleeName) {}

public:
  virtual ~CallExprRewriter() {}

  /// This function should be overwrited to implement call expression rewriting.
  virtual Optional<std::string> rewrite() = 0;

protected:
  std::vector<std::string> getMigratedArgs();
  std::string getMigratedArg(unsigned Index);

  StringRef getSourceCalleeName() { return SourceCalleeName; }

  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  inline void report(IDTy MsgID, bool UseTextBegin, Ts &&... Vals) {
    TransformSetTy TS;
    DiagnosticsUtils::report<IDTy, Ts...>(
        Call->getBeginLoc(), MsgID, DpctGlobalInfo::getCompilerInstance(), &TS,
        UseTextBegin, std::forward<Ts>(Vals)...);
    for (auto &T : TS)
      DpctGlobalInfo::getInstance().addReplacement(
          T->getReplacement(DpctGlobalInfo::getContext()));
  }
};

class ConditionalRewriterFactory : public CallExprRewriterFactoryBase {
  std::function<bool(const CallExpr *)> Pred;
  std::shared_ptr<CallExprRewriterFactoryBase> First, Second;

public:
  template <class InputPred>
  ConditionalRewriterFactory(
      InputPred &&P, std::shared_ptr<CallExprRewriterFactoryBase> FirstFactory,
      std::shared_ptr<CallExprRewriterFactoryBase> SecondFactory)
      : Pred(std::forward<InputPred>(P)), First(FirstFactory),
        Second(SecondFactory) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) override {
    if (Pred(C))
      return First->create(C);
    else
      return Second->create(C);
  }
};

class AssignableRewriter : public CallExprRewriter {
  std::shared_ptr<CallExprRewriter> Inner;
  bool IsAssigned;

public:
  template <class... InitArgs>
  AssignableRewriter(const CallExpr *C,
                     std::shared_ptr<CallExprRewriter> InnerRewriter)
      : CallExprRewriter(C, ""), Inner(InnerRewriter),
        IsAssigned(isAssigned(C)) {}

  Optional<std::string> rewrite() override {
    Optional<std::string> &&Result = Inner->rewrite();
    if (Result.hasValue() && IsAssigned)
      return "(" + Result.getValue() + ", 0)";
    return Result;
  }
};

class AssignableRewriterFactory : public CallExprRewriterFactoryBase {
  std::shared_ptr<CallExprRewriterFactoryBase> Inner;

public:
  AssignableRewriterFactory(
      std::shared_ptr<CallExprRewriterFactoryBase> InnerFactory)
      : Inner(InnerFactory) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) override {
    return std::make_shared<AssignableRewriter>(C, Inner->create(C));
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

  virtual Optional<std::string> rewrite() override;

  friend FuncCallExprRewriterFactory;

protected:
  template <class... Args> void appendRewriteArg(Args &&... Arguments) {
    RewriteArgList.emplace_back(std::forward<Args...>(Arguments)...);
  }

  // Build string which is used to replace original expession.
  Optional<std::string> buildRewriteString();

  void setTargetCalleeName(const std::string &Str) { TargetCalleeName = Str; }
};

/// Base class for rewriting math function calls
class MathCallExprRewriter : public FuncCallExprRewriter {
public:
  virtual Optional<std::string> rewrite() override;

protected:
  MathCallExprRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : FuncCallExprRewriter(Call, SourceCalleeName, TargetCalleeName) {}

  void reportUnsupportedRoundingMode();
};

/// The rewriter for renaming math function calls
class MathFuncNameRewriter : public MathCallExprRewriter {
protected:
  MathFuncNameRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : MathCallExprRewriter(Call, SourceCalleeName, TargetCalleeName) {}

public:
  virtual Optional<std::string> rewrite() override;

protected:
  std::string getNewFuncName();

  friend MathFuncNameRewriterFactory;

private:
  static const std::vector<std::string> SingleFuctions;
  static const std::vector<std::string> DoubleFuctions;
};

/// The rewriter for warning on unsupported math functions
class MathUnsupportedRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathUnsupportedRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                          StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual Optional<std::string> rewrite() override;

  friend MathUnsupportedRewriterFactory;
};

/// The rewriter for replacing math function calls with type casting expressions
class MathTypeCastRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathTypeCastRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual Optional<std::string> rewrite() override;

  friend MathTypeCastRewriterFactory;
};

/// The rewriter for replacing math function calls with emulations
class MathSimulatedRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathSimulatedRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                        StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual Optional<std::string> rewrite() override;

  friend MathSimulatedRewriterFactory;
};

/// The rewriter for replacing math function calls with binary operator
/// expressions
class MathBinaryOperatorRewriter : public MathCallExprRewriter {
  std::string LHS, RHS;
  BinaryOperatorKind Op;

protected:
  MathBinaryOperatorRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                             BinaryOperatorKind Op)
      : MathCallExprRewriter(Call, SourceCalleeName, ""), Op(Op) {}

public:
  virtual ~MathBinaryOperatorRewriter() {}

  virtual Optional<std::string> rewrite() override;

protected:
  void setLHS(std::string L) { LHS = L; }
  void setRHS(std::string R) { RHS = R; }

  // Build string which is used to replace original expession.
  inline Optional<std::string> buildRewriteString() {
    if (LHS == "")
      return buildString(BinaryOperator::getOpcodeStr(Op), RHS);
    return buildString(LHS, " ", BinaryOperator::getOpcodeStr(Op), " ", RHS);
  }

  friend MathBinaryOperatorRewriterFactory;
};

/// The rewriter for migrating warp functions
class WarpFunctionRewriter : public FuncCallExprRewriter {
private:
  static const std::map<std::string, std::string> WarpFunctionsMap;
  void reportNoMaskWarning() {
    report(Diagnostics::MASK_UNSUPPORTED, false, TargetCalleeName);
  }

protected:
  WarpFunctionRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : FuncCallExprRewriter(Call, SourceCalleeName, TargetCalleeName) {}

public:
  virtual Optional<std::string> rewrite() override;

protected:
  std::string getNewFuncName();

  friend WarpFunctionRewriterFactory;
};

/// The rewriter for reordering function arguments
class ReorderFunctionRewriter : public FuncCallExprRewriter {
  const std::vector<unsigned> &RewriterArgsIdx;

public:
  ReorderFunctionRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                          StringRef TargetCalleeName,
                          const std::vector<unsigned> &ArgsIdx)
      : FuncCallExprRewriter(Call, SourceCalleeName, TargetCalleeName),
        RewriterArgsIdx(ArgsIdx) {}

  virtual Optional<std::string> rewrite() override;

  friend ReorderFunctionRewriterFactory;
};

class TemplatedCallExprRewriter : public FuncCallExprRewriter {
  std::vector<std::string> TemplateArgs;
  void buildTemplateArgsList();
  void buildTemplateArgsList(const ArrayRef<TemplateArgumentLoc> &Args);

public:
  TemplatedCallExprRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                            StringRef TargetCalleeName)
      : FuncCallExprRewriter(Call, SourceCalleeName, TargetCalleeName) {}

  virtual Optional<std::string> rewrite() override;

  friend TemplatedCallExprRewriterFactory;
};

template <class StreamT, class T> void print(StreamT &Stream, const T &Val) {
  Val.print(Stream);
}
template <class StreamT> void print(StreamT &Stream, const Expr *E) {
  ExprAnalysis EA;
  print(Stream, EA, E);
}
template <class StreamT>
void print(StreamT &Stream, ExprAnalysis &EA, const Expr *E) {
  EA.analyze(E);
  Stream << EA.getRewritePrefix() << EA.getReplacedString()
         << EA.getRewritePostfix();
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

template <class StreamT> void printMemberOp(StreamT &Stream, bool IsArrow) {
  if (IsArrow)
    Stream << "->";
  else
    Stream << ".";
}

class DerefExpr {
  bool AddrOfRemoved = false, NeedParens = false;
  const Expr *E = nullptr;

  template <class StreamT>
  void print(StreamT &Stream, ExprAnalysis &EA, bool IgnoreDerefOp) const {
    std::unique_ptr<ParensPrinter<StreamT>> Parens;
    if (!AddrOfRemoved && !IgnoreDerefOp)
      Stream << "*";

    printWithParens(Stream, EA, E);
  }

  DerefExpr() = default;

public:
  template <class StreamT>
  void printArg(StreamT &Stream, ArgumentAnalysis &A) const {
    print(Stream, A, false);
  }
  template <class StreamT> void printMemberBase(StreamT &Stream) const {
    ExprAnalysis EA;
    print(Stream, EA, true);
    printMemberOp(Stream, !AddrOfRemoved);
  }
  template <class StreamT> void print(StreamT &Stream) const {
    ExprAnalysis EA;
    print(Stream, EA, false);
  }

  static DerefExpr create(const Expr *E);
};

template <bool HasPrefixArg, class... ArgsT> class ArgsPrinter;
template <bool HasPrefixArg> class ArgsPrinter<HasPrefixArg> {
  mutable ArgumentAnalysis A;

public:
  template <class StreamT> void print(StreamT &) const {}
  template <class StreamT> void printArg(StreamT &Stream, const Expr *E) const {
    dpct::print(Stream, A, E);
  }
  template <class StreamT, class ArgT>
  void printArg(StreamT &Stream, const ArgT &Arg) const {
    Arg.printArg(Stream, A);
  }
  template <class StreamT>
  void printComma(StreamT &Stream, std::true_type) const {
    Stream << ", ";
  }
  template <class StreamT>
  void printComma(StreamT &Stream, std::false_type) const {}

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
  ArgsPrinter(InputFirstArgT &&FirstArg, InputRestArgsT &&... RestArgs)
      : Base(std::forward<InputRestArgsT>(RestArgs)...),
        First(std::forward<InputFirstArgT>(FirstArg)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Base::printComma(Stream, std::integral_constant<bool, HasPrefixArg>());
    Base::printArg(Stream, First);
    Base::print(Stream);
  }
};

template <class... ArgsExtratorT>
class CallArgsPrinter : public ArgsPrinter<false, ArgsExtratorT...> {
  using Base = ArgsPrinter<false, ArgsExtratorT...>;

public:
  template <class... InputArgsT>
  CallArgsPrinter(InputArgsT &&... Args)
      : Base(std::forward<InputArgsT>(Args)...) {}
  template <class StreamT> void print(StreamT &Stream) const {
    ParensPrinter<StreamT> Parens(Stream);
    Base::print(Stream);
  }
};

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

template <class BaseT, class... CallArgsT> class MemberCallPrinter {
  BaseT Base;
  bool IsArrow;
  StringRef MemberName;
  CallArgsPrinter<CallArgsT...> Args;

public:
  MemberCallPrinter(BaseT &&Base, bool IsArrow, StringRef MemberName,
                    CallArgsT &&... Args)
      : Base(std::forward<BaseT>(Base)), IsArrow(IsArrow),
        MemberName(MemberName), Args(std::forward<CallArgsT>(Args)...) {}

  template <class StreamT> void print(StreamT &Stream) const {
    printBase(Stream, Base, IsArrow);
    Stream << MemberName;
    Args.print(Stream);
  }
};

template <class LValueT, class RValueT> class AssignExprPrinter {
  LValueT LVal;
  RValueT RVal;

public:
  AssignExprPrinter(LValueT &&L, RValueT &&R)
      : LVal(std::forward<LValueT>(L)), RVal(std::forward<RValueT>(R)) {}
  template <class StreamT> void print(StreamT &Stream) {
    dpct::print(Stream, LVal);
    Stream << " = ";
    dpct::print(Stream, RVal);
  }
};

template <class ArgT> class DeleterCallExprRewriter : public CallExprRewriter {
  ArgT Arg;

public:
  DeleterCallExprRewriter(const CallExpr *C, StringRef Source,
                          std::function<ArgT(const CallExpr *)> ArgCreator)
      : CallExprRewriter(C, Source), Arg(ArgCreator(C)) {}
  Optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    OS << "delete ";
    printWithParens(OS, Arg);
    return Result;
  }
};

template <class Printer>
class PrinterRewriter : public Printer, public CallExprRewriter {
public:
  template <class... ArgsT>
  PrinterRewriter(const CallExpr *C, StringRef Source, ArgsT &&... Args)
      : Printer(std::forward<ArgsT>(Args)...), CallExprRewriter(C, Source) {}
  Optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Printer::print(OS);
    return Result;
  }
};

template <class BaseT, class... ArgsT>
class MemberCallExprRewriter
    : public PrinterRewriter<MemberCallPrinter<BaseT, ArgsT...>> {
public:
  MemberCallExprRewriter(
      const CallExpr *C, StringRef Source,
      std::function<BaseT(const CallExpr *)> &BaseCreator, bool IsArrow,
      StringRef Member, std::function<ArgsT(const CallExpr *)> &... ArgsCreator)
      : PrinterRewriter<MemberCallPrinter<BaseT, ArgsT...>>(
            C, Source, BaseCreator(C), IsArrow, Member, ArgsCreator(C)...) {}
};

template <class LValueT, class RValueT>
class AssignExprRewriter
    : public PrinterRewriter<AssignExprPrinter<LValueT, RValueT>> {
public:
  AssignExprRewriter(const CallExpr *C, StringRef Source,
                     std::function<LValueT(const CallExpr *)> &LCreator,
                     std::function<RValueT(const CallExpr *)> &RCreator)
      : PrinterRewriter<AssignExprPrinter<LValueT, RValueT>>(
            C, Source, LCreator(C), RCreator(C)) {}
};

class UnsupportFunctionRewriter : public CallExprRewriter {
  Diagnostics ID;

public:
  UnsupportFunctionRewriter(const CallExpr *CE, StringRef CalleeName,
                            Diagnostics MsgID)
      : CallExprRewriter(CE, CalleeName), ID(MsgID) {}

  Optional<std::string> rewrite() override {
    report(ID, false, getSourceCalleeName());
    return Optional<std::string>();
  }

  friend UnsupportFunctionRewriterFactory;
};

} // namespace dpct
} // namespace clang

#endif // !__CALL_EXPR_REWRITER_H__
