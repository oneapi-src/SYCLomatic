//===---LIBCUAPIMigration.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "LIBCUAPIMigration.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "Statics.h"
#include "MapNames.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"

namespace clang {
namespace dpct {

using namespace clang;
using namespace clang::dpct;
using namespace clang::ast_matchers;

void LIBCUAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto LIBCUAPIHasNames = [&]() {
    return hasAnyName("cuda::std::atomic_thread_fence");
  };
  MF.addMatcher(callExpr(callee(functionDecl(LIBCUAPIHasNames()))).bind("call"), this);
}


void LIBCUAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  std::string APIName;
  std::ostringstream OS;
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call")) {
    auto DC = CE->getDirectCallee();
    APIName = DC->getNameAsString();
    if(APIName == "atomic_thread_fence"){
      auto FirArg = CE->getArg(0);
      ExprAnalysis FirEA(FirArg);
      FirEA.analyze();
      OS << MapNames::getClNamespace()<<"atomic_fence"<<"("<< FirEA.getReplacedString()<< ")";
      emplaceTransformation(new ReplaceStmt(CE, OS.str()));
    }
    
  }
}
  
void LIBCUMemberFuncRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto LIBCUMemberFuncHasNamses = [&]() {
    return hasAnyName("load","store","exchange","compare_exchange_weak","compare_exchange_strong",
                      "fetch_add", "fetch_sub", "fetch_and", "fetch_or", "fetch_xor");
  };
  auto LIBCUTypesHasNamses = [&]() {
    return hasAnyName("cuda::atomic","cuda::std::atomic");
  };
  MF.addMatcher(cxxMemberCallExpr(
                    allOf(on(hasType(hasCanonicalType(qualType(
                              hasDeclaration(namedDecl(LIBCUTypesHasNamses())))))),
                          callee(cxxMethodDecl(LIBCUMemberFuncHasNamses()))))
                    .bind("memberCallExpr"),
                this);  
}

void LIBCUMemberFuncRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {

  if (auto CMCE = getNodeAsType<CXXMemberCallExpr>(Result, "memberCallExpr")) {
    dpct::ExprAnalysis EA;
    EA.analyze(CMCE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

} // dpct
} // clang