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
class CallExprRewriter {
  const CallExpr *Call;
  ArgumentAnalysis Analyzer;

public:
  CallExprRewriter(const CallExpr *Call) : Call(Call) {}
  virtual ~CallExprRewriter() {}

  // This function should be overwrited to implement call expression rewriting.
  virtual Optional<std::string> rewrite() = 0;

protected:
  bool isValid() { return Call; }
  std::vector<std::string> getMigratedArgs();
  std::string getMigratedArg(unsigned Index);

  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  inline void report(IDTy MsgID, Ts &&... Vals) {
    TransformSetTy TS;
    DiagnosticsUtils::report<IDTy, Ts...>(
        Call->getBeginLoc(), MsgID, SyclctGlobalInfo::getCompilerInstance(), TS,
        std::forward<Ts>(Vals)...);
    for (auto &T : TS)
      SyclctGlobalInfo::getInstance().addReplacement(
          T->getReplacement(SyclctGlobalInfo::getContext()));
  }
};

class FuncCallExprRewriter : public CallExprRewriter {
  StringRef CalleeName;
  std::vector<std::string> RewriteArgList;

public:
  FuncCallExprRewriter(const CallExpr *Call, StringRef CalleeName)
      : CallExprRewriter(Call), CalleeName(CalleeName) {}
  virtual ~FuncCallExprRewriter() {}

  virtual Optional<std::string> rewrite() override;

protected:
  template <class... Args> void appendRewriteArg(Args &&... Arguments) {
    RewriteArgList.emplace_back(std::forward<Args...>(Arguments)...);
  }

  // Build string which is used to replace original expession.
  Optional<std::string> buildRewriteString();
};

class BinaryOperatorRewriter : public CallExprRewriter {
  std::string LHS, RHS;
  BinaryOperatorKind Op;

public:
  BinaryOperatorRewriter(const CallExpr *Call, BinaryOperatorKind Op)
      : CallExprRewriter(Call), Op(Op) {}
  virtual ~BinaryOperatorRewriter() {}

  virtual Optional<std::string> rewrite() override;

protected:
  void setLHS(std::string L) { LHS = L; }
  void setRHS(std::string R) { RHS = R; }

  // Build string which is used to replace original expession.
  inline Optional<std::string> buildRewriteString() {
    return buildString(LHS, " ", BinaryOperator::getOpcodeStr(Op), " ", RHS);
  }
};

/*
Factory usage example:
using BinaryOperatorRewriterFactory =
    CallExprRewriterFactory<BinaryOperatorRewriter, BinaryOperatorKind>;
*/
class CallExprRewriterFactoryBase {
public:
  virtual std::shared_ptr<CallExprRewriter> create(const CallExpr *) = 0;
  virtual ~CallExprRewriterFactoryBase() {}

  static const std::unordered_map<std::string,
                           std::shared_ptr<CallExprRewriterFactoryBase>>
      CallMap;
};
template <class RewriterTy, class... Args>
class CallExprRewriterFactory : public CallExprRewriterFactoryBase {
  std::tuple<Args...> Initializer;

private:
  template <size_t... Idx>
  inline std::shared_ptr<CallExprRewriter>
  createRewriter(const CallExpr *Call, llvm::index_sequence<Idx...>) {
    return std::make_shared<RewriterTy>(Call, std::get<Idx>(Initializer)...);
  }

public:
  CallExprRewriterFactory(Args... Arguments)
      : Initializer(std::move(Arguments)...) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *Call) override {
    return createRewriter(Call, llvm::index_sequence_for<Args...>());
  }
};

using BinaryOperatorRewriterFactory =
    CallExprRewriterFactory<BinaryOperatorRewriter, BinaryOperatorKind>;
using FuncCallExprRewriterFactory =
    CallExprRewriterFactory<FuncCallExprRewriter, std::string>;
} // namespace syclct
} // namespace clang

#endif // !__CALL_EXPR_REWRITER_H__
