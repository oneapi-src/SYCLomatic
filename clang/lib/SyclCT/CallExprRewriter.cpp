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
  if (!isValid())
    return Optional<std::string>();
  RewriteArgList = getMigratedArgs();
  return buildRewriteString();
}

Optional<std::string> FuncCallExprRewriter::buildRewriteString() {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << CalleeName << "(";
  for (auto &Arg : RewriteArgList)
    OS << Arg << ", ";
  OS.flush();
  return RewriteArgList.empty() ? Result.append(")")
                                : Result.replace(Result.length() - 2, 2, ")");
}

Optional<std::string> BinaryOperatorRewriter::rewrite() {
  if (!isValid())
    return Optional<std::string>();
  setLHS(getMigratedArg(0));
  setRHS(getMigratedArg(1));
  return buildRewriteString();
}

#define REWRITER_FACTORY_ENTRY(FuncName, RewriterTy, ...)                      \
  {FuncName, std::make_shared<RewriterTy>(__VA_ARGS__)}
#define FUNC_FACTORY_ENTRY(FuncName, RewriterName)                    \
  REWRITER_FACTORY_ENTRY(FuncName, FuncCallExprRewriterFactory, RewriterName)
#define BO_FACTORY_ENTRY(FuncName, OpKind)                            \
  REWRITER_FACTORY_ENTRY(FuncName, BinaryOperatorRewriterFactory, OpKind)

const std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>
    CallExprRewriterFactoryBase::CallMap = {
        /* CallMap definition examples with macros:
        FUNC_FACTORY_ENTRY("test_func", "test_success"),
        BO_FACTORY_ENTRY("test_add", BO_Add)
        */
};
} // namespace syclct
} // namespace clang