//===--- CallExprRewriter.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef CALL_EXPR_REWRITER_H
#define CALL_EXPR_REWRITER_H

#include "AnalysisInfo.h"

namespace clang {
namespace syclct {

class CallExprRewriter;
class FuncCallExprRewriter;
class MathCallExprRewriter;
class MathFuncNameRewriter;
class MathSimulatedRewriter;
class MathTypeCastRewriter;
class MathBinaryOperatorRewriter;
class MathUnsupportedRewriter;
class WarpFunctionRewriter;

/*
Factory usage example:
using BinaryOperatorExprRewriterFactory =
    CallExprRewriterFactory<BinaryOperatorExprRewriter, BinaryOperatorKind>;
*/
class CallExprRewriterFactoryBase {
public:
  virtual std::shared_ptr<CallExprRewriter> create(const CallExpr *) = 0;
  virtual ~CallExprRewriterFactoryBase() {}

  static const std::unordered_map<std::string,
                                  std::shared_ptr<CallExprRewriterFactoryBase>>
      RewriterMap;
};
template <class RewriterTy, class... Args>
class CallExprRewriterFactory : public CallExprRewriterFactoryBase {
  std::tuple<std::string, Args...> Initializer;

private:
  template <size_t... Idx>
  inline std::shared_ptr<CallExprRewriter>
  createRewriter(const CallExpr *Call, llvm::index_sequence<Idx...>) {
    return std::shared_ptr<RewriterTy>(
        new RewriterTy(Call, std::get<Idx>(Initializer)...));
  }

public:
  CallExprRewriterFactory(StringRef SourceCalleeName, Args... Arguments)
      : Initializer(SourceCalleeName, std::move(Arguments)...) {}
  // Create a meaningful rewriter only if the CallExpr is not nullptr
  std::shared_ptr<CallExprRewriter> create(const CallExpr *Call) override {
    if (!Call)
      return std::shared_ptr<CallExprRewriter>();
    return createRewriter(Call,
                          llvm::index_sequence_for<std::string, Args...>());
  }
};

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

  // This function should be overwrited to implement call expression rewriting.
  virtual Optional<std::string> rewrite() = 0;

protected:
  std::vector<std::string> getMigratedArgs();
  std::string getMigratedArg(unsigned Index);

  StringRef getSourceCalleeName() { return SourceCalleeName; }

  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  inline void report(IDTy MsgID, Ts &&... Vals) {
    TransformSetTy TS;
    DiagnosticsUtils::report<IDTy, Ts...>(
        Call->getBeginLoc(), MsgID, SyclctGlobalInfo::getCompilerInstance(),
        &TS, std::forward<Ts>(Vals)...);
    for (auto &T : TS)
      SyclctGlobalInfo::getInstance().addReplacement(
          T->getReplacement(SyclctGlobalInfo::getContext()));
  }
};

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

protected:
  template <class... Args> void appendRewriteArg(Args &&... Arguments) {
    RewriteArgList.emplace_back(std::forward<Args...>(Arguments)...);
  }

  // Build string which is used to replace original expession.
  Optional<std::string> buildRewriteString();

  void setTargetCalleeName(const std::string &Str) {
    TargetCalleeName = Str;
  }
};

class MathCallExprRewriter : public FuncCallExprRewriter {
public:
  virtual Optional<std::string> rewrite() override;

protected:
  MathCallExprRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
    : FuncCallExprRewriter(Call, SourceCalleeName, TargetCalleeName) {}

  void reportUnsupportedRoundingMode();
};

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
};

class MathUnsupportedRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathUnsupportedRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                          StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual Optional<std::string> rewrite() override;

  friend MathUnsupportedRewriterFactory;
};

class MathTypeCastRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathTypeCastRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual Optional<std::string> rewrite() override;

  friend MathTypeCastRewriterFactory;
};

class MathSimulatedRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathSimulatedRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                        StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual Optional<std::string> rewrite() override;

  friend MathSimulatedRewriterFactory;
};

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

class WarpFunctionRewriter : public FuncCallExprRewriter {
private:
  static const std::map<std::string, std::string> WarpFunctionsMap;
  void reportNoMaskWarning() {
    report(Diagnostics::MASK_UNSUPPORTED, TargetCalleeName);
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

} // namespace syclct
} // namespace clang

#endif // !__CALL_EXPR_REWRITER_H__
