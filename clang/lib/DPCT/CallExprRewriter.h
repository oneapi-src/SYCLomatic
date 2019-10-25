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
class TexFunctionRewriter;
class UnsupportFunctionRewriter;

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
  CallExprRewriterFactory(StringRef SourceCalleeName, Args &&... Arguments)
      : Initializer(SourceCalleeName, std::move(Arguments)...) {}
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
using TexFunctionRewriterFactory =
    CallExprRewriterFactory<TexFunctionRewriter, std::string>;
using UnsupportFunctionRewriterFactory =
    CallExprRewriterFactory<UnsupportFunctionRewriter, Diagnostics>;

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
  inline void report(IDTy MsgID, Ts &&... Vals) {
    TransformSetTy TS;
    DiagnosticsUtils::report<IDTy, Ts...>(Call->getBeginLoc(), MsgID,
                                          DpctGlobalInfo::getCompilerInstance(),
                                          &TS, std::forward<Ts>(Vals)...);
    for (auto &T : TS)
      DpctGlobalInfo::getInstance().addReplacement(
          T->getReplacement(DpctGlobalInfo::getContext()));
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

class TexFunctionRewriter : public FuncCallExprRewriter {
  void setTextureInfo();

public:
  TexFunctionRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                      StringRef TargetCalleeName)
      : FuncCallExprRewriter(Call, SourceCalleeName, TargetCalleeName) {
    setTextureInfo();
  }

  friend TexFunctionRewriterFactory;
};

class UnsupportFunctionRewriter : public CallExprRewriter {
  Diagnostics ID;

public:
  UnsupportFunctionRewriter(const CallExpr *CE, StringRef CalleeName,
                            Diagnostics MsgID)
      : CallExprRewriter(CE, CalleeName), ID(MsgID) {}

  Optional<std::string> rewrite() override {
    report(ID, getSourceCalleeName());
    return Optional<std::string>();
  }

  friend UnsupportFunctionRewriterFactory;
};

} // namespace dpct
} // namespace clang

#endif // !__CALL_EXPR_REWRITER_H__
