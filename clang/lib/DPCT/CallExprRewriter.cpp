//===--------------- CallExprRewriter.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"
#include <memory>

namespace clang {
namespace dpct {

std::shared_ptr<CallExprRewriterFactoryBase>
createUserDefinedRewriterFactory(const std::string &Source, MetaRuleObject &R) {
  return std::make_shared<UserDefinedRewriterFactory>(R);
}

std::shared_ptr<CallExprRewriterFactoryBase>
createUserDefinedMethodRewriterFactory(
    const std::string &Source, MetaRuleObject &R,
    std::shared_ptr<MetaRuleObject::ClassMethod> MethodPtr) {
  return std::make_shared<UserDefinedRewriterFactory>(R, MethodPtr);
}

std::function<bool(const CallExpr *C)> hasManagedAttr(int Idx) {
  return [=](const CallExpr *C) -> bool {
    const Expr *Arg = C->getArg(Idx)->IgnoreImpCasts();
    if (auto CSCE = dyn_cast_or_null<CStyleCastExpr>(Arg)) {
      Arg = CSCE->getSubExpr();
    }
    if (auto UO = dyn_cast_or_null<UnaryOperator>(Arg)) {
      Arg = UO->getSubExpr();
    }
    if (auto ArgDRE = dyn_cast_or_null<DeclRefExpr>(Arg)) {
      auto D = ArgDRE->getDecl();
      if (D->hasAttr<HIPManagedAttr>()) {
        return true;
      }
    }
    return false;
  };
}

DerefExpr DerefExpr::create(const Expr *E, const CallExpr * C) {
  DerefExpr D;
  D.C = C;
  // If E is UnaryOperator or CXXOperatorCallExpr D.E will has value
  D.E = getDereferencedExpr(E);
  if (D.E) {
    D.E = D.E->IgnoreParens();
    D.AddrOfRemoved = true;
  } else {
    D.E = E;
  }

  D.NeedParens = needExtraParens(E);
  return D;
}

std::string CallExprRewriter::getMigratedArg(unsigned Idx) {
  Analyzer.setCallSpelling(Call);
  Analyzer.analyze(Call->getArg(Idx));
  return Analyzer.getRewritePrefix() + Analyzer.getRewriteString() +
         Analyzer.getRewritePostfix();
}

std::vector<std::string> CallExprRewriter::getMigratedArgs() {
  std::vector<std::string> ArgList;
  Analyzer.setCallSpelling(Call);
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

std::unique_ptr<std::unordered_map<
    std::string, std::shared_ptr<CallExprRewriterFactoryBase>>>
    CallExprRewriterFactoryBase::RewriterMap = std::make_unique<std::unordered_map<
    std::string, std::shared_ptr<CallExprRewriterFactoryBase>>>();

std::unique_ptr<std::unordered_map<
    std::string, std::shared_ptr<CallExprRewriterFactoryBase>>>
    CallExprRewriterFactoryBase::MethodRewriterMap = std::make_unique<std::unordered_map<
    std::string, std::shared_ptr<CallExprRewriterFactoryBase>>>();

void CallExprRewriterFactoryBase::initRewriterMap() {
  initRewriterMapCUB();
  initRewriterMapCUFFT();
  initRewriterMapCUBLAS();
  initRewriterMapCURAND();
  initRewriterMapCUSOLVER();
  initRewriterMapComplex();
  initRewriterMapDriver();
  initRewriterMapMemory();
  initRewriterMapNccl();
  initRewriterMapStream();
  initRewriterMapTexture();
  initRewriterMapThrust();
  initRewriterMapWarp();
  initRewriterMapCUDNN();
  initRewriterMapErrorHandling();
  initRewriterMapLIBCU();
  initRewriterMapEvent();
  initRewriterMapMath();
  initRewriterMapCooperativeGroups();
  initMethodRewriterMap();
}

} // namespace dpct
} // namespace clang
