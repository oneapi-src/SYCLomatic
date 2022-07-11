//===------------------ LIBCUAPIMigration.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call")) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

void LIBCUTypeRule::registerMatcher(ast_matchers::MatchFinder &MF){
  auto TargetTypeName = [&]() { return hasAnyName(
    "cuda::atomic","cuda::std::atomic"); 
    };

  MF.addMatcher(typeLoc(
                    loc(qualType(hasDeclaration(namedDecl(TargetTypeName())))))
                    .bind("loc"),
                this);
}

void LIBCUTypeRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto TL = getNodeAsType<TypeLoc>(Result, "loc")) {
    ExprAnalysis EA;
    EA.analyze(*TL);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

void LIBCUMemberFuncRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto LIBCUMemberFuncHasNamses = [&]() {
    return hasAnyName("load","store","exchange","compare_exchange_weak","compare_exchange_strong",
                      "fetch_add", "fetch_sub");
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