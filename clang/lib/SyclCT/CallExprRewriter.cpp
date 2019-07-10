//===--- CallExprRewriter.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "CallExprRewriter.h"

namespace clang {
namespace syclct {

std::string CallExprRewriter::getMigratedArg(unsigned Idx) {
  if (!Call)
    return "";
  Analyzer.analyze(Call->getArg(Idx));
  return Analyzer.getReplacedString();
}

std::vector<std::string> CallExprRewriter::getMigratedArgs() {
  std::vector<std::string> ArgList;
  for (unsigned i = 0; i < Call->getNumArgs(); ++i)
    ArgList.emplace_back(getMigratedArg(i));
  return ArgList;
}

Optional<std::string> FuncCallExprRewriter::rewrite() {
  RewriteArgList = getMigratedArgs();
  return buildRewriteString();
}

Optional<std::string> FuncCallExprRewriter::buildRewriteString() {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << TargetCalleeName << "(";
  for (auto &Arg : RewriteArgList)
    OS << Arg << ", ";
  OS.flush();
  return RewriteArgList.empty() ? Result.append(")")
                                : Result.replace(Result.length() - 2, 2, ")");
}

Optional<std::string> UnsupportedFuncCallExprRewriter::rewrite() {
  report(Diagnostics::NOTSUPPORTED, SourceCalleeName);
  return Base::rewrite();
}

Optional<std::string> SimulatedFuncCallExprRewriter::rewrite() {
  report(Diagnostics::MATH_SIMULATION, SourceCalleeName, TargetCalleeName);
  return Base::rewrite();
}

Optional<std::string> BinaryOperatorExprRewriter::rewrite() {
  setLHS(getMigratedArg(0));
  setRHS(getMigratedArg(1));
  return buildRewriteString();
}

#define REWRITER_FACTORY_ENTRY(FuncName, RewriterTy, ...)                      \
  { FuncName, std::make_shared<RewriterTy>(FuncName, __VA_ARGS__) }
#define FUNC_FACTORY_ENTRY(FuncName, RewriterName)                             \
  REWRITER_FACTORY_ENTRY(FuncName, FuncCallExprRewriterFactory, RewriterName)
#define UNSUPPORTED_FUNC_FACTORY_ENTRY(FuncName)                               \
  REWRITER_FACTORY_ENTRY(FuncName, UnsupportedFuncCallExprRewriterFactory,     \
                         FuncName)
#define SIMULATED_FUNC_FACTORY_ENTRY(FuncName, RewriterName)                   \
  REWRITER_FACTORY_ENTRY(FuncName, SimulatedFuncCallExprRewriterFactory,       \
                         RewriterName)
#define BO_FACTORY_ENTRY(FuncName, OpKind)                                     \
  REWRITER_FACTORY_ENTRY(FuncName, BinaryOperatorExprRewriterFactory, OpKind)

const std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>
    CallExprRewriterFactoryBase::RewriterMap = {
        /* CallMap definition examples with macros:
        FUNC_FACTORY_ENTRY("test_func", "test_success"),
        BO_FACTORY_ENTRY("test_add", BO_Add)
        */
};
} // namespace syclct
} // namespace clang
