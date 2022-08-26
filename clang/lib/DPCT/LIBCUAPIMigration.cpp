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
#include "CallExprRewriter.h"
#include "Diagnostics.h"
#include "MapNames.h"
#include "Statics.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"

namespace {
bool CheckTopNamespaceIsCuda(const clang::CallExpr *CE) {
  const auto *DC = CE->getDirectCallee();
  if (!DC)
    return false;
  if (const auto *NSD =
          clang::dyn_cast<clang::NamespaceDecl>(DC->getDeclContext())) {
    while (NSD->isInline())
      NSD = clang::dyn_cast<clang::NamespaceDecl>(NSD->getDeclContext());
    if (NSD->getName() == "__detail")
      NSD = clang::dyn_cast<clang::NamespaceDecl>(NSD->getDeclContext());
    return NSD->getName() == "cuda";
  } else
    return false;
}

bool CheckNamespaceIsCudaStd(const clang::CallExpr *CE) {
  const auto *DC = CE->getDirectCallee();
  if (!DC)
    return false;
  if (const auto *NSD =
          clang::dyn_cast<clang::NamespaceDecl>(DC->getDeclContext())) {
    while (NSD->isInline())
      NSD = clang::dyn_cast<clang::NamespaceDecl>(NSD->getDeclContext());
    if (NSD->getName() == "__detail")
      NSD = clang::dyn_cast<clang::NamespaceDecl>(NSD->getDeclContext());
    if (NSD->getName() == "std") {
      NSD = clang::dyn_cast<clang::NamespaceDecl>(NSD->getDeclContext());
      return NSD->getName() == "cuda";
    } else
      return false;
  } else
    return false;
}

} // namespace

namespace clang {
namespace dpct {

using namespace clang;
using namespace clang::dpct;
using namespace clang::ast_matchers;

void LIBCURule::processLIBCUMemberCall(const CXXMemberCallExpr *MC) {
  ExprAnalysis EA;
  EA.analyze(MC);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

void LIBCURule::processLIBCUTypeLoc(const TypeLoc *TL) {
  ExprAnalysis EA;
  EA.analyze(*TL);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

void LIBCURule::processLIBCUFuncCall(const CallExpr *CE) {
  if (CheckTopNamespaceIsCuda(CE) || CheckNamespaceIsCudaStd(CE)) {
    ExprAnalysis EA;
    EA.analyze(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

void LIBCURule::processLIBCUUsingDirectiveDecl(const UsingDirectiveDecl *UDD) {
  std::string NamespaceName = UDD->getNominatedNamespace()->getNameAsString();
  if (NamespaceName == "cuda") {
    if (const auto *NSD = dyn_cast<NamespaceDecl>(UDD->getNominatedNamespace())) {
      if (!DpctGlobalInfo::isInRoot(NSD->getLocation())) {
        emplaceTransformation(new ReplaceDecl(UDD, ""));
      }
    }
  }
}

void LIBCURule::registerMatcher(ast_matchers::MatchFinder &MF) {
  {
    auto LIBCUMemberFuncHasNames = [&]() {
      return hasAnyName("load", "store", "exchange", "compare_exchange_weak",
                        "compare_exchange_strong", "fetch_add", "fetch_sub");
    };
    auto LIBCUTypesHasNames = [&]() {
      return hasAnyName("cuda::atomic", "cuda::std::atomic");
    };
    MF.addMatcher(cxxMemberCallExpr(
                      allOf(on(hasType(hasCanonicalType(qualType(hasDeclaration(
                                namedDecl(LIBCUTypesHasNames())))))),
                            callee(cxxMethodDecl(LIBCUMemberFuncHasNames()))))
                      .bind("MemberCall"),
                  this);
  }

  {
    auto LIBCUTypesNames = [&]() { return hasAnyName("atomic"); };
    MF.addMatcher(typeLoc(loc(hasCanonicalType(qualType(
                              hasDeclaration(namedDecl(LIBCUTypesNames()))))))
                      .bind("TypeLoc"),
                  this);
  }

  {
    auto LIBCUAPIHasNames = [&]() { return hasAnyName("atomic_thread_fence"); };
    MF.addMatcher(
        callExpr(callee(functionDecl(LIBCUAPIHasNames()))).bind("FuncCall"),
        this);
  }

  { MF.addMatcher(usingDirectiveDecl().bind("UsingDirectiveDecl"), this); }
}

void LIBCURule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CXXMemberCallExpr *MC =
          getNodeAsType<CXXMemberCallExpr>(Result, "MemberCall")) {
    processLIBCUMemberCall(MC);
  } else if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FuncCall")) {
    processLIBCUFuncCall(CE);
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "TypeLoc")) {
    processLIBCUTypeLoc(TL);
  } else if (auto UDD = getNodeAsType<UsingDirectiveDecl>(
                 Result, "UsingDirectiveDecl")) {
    processLIBCUUsingDirectiveDecl(UDD);
  }
}

} // namespace dpct
} // namespace clang