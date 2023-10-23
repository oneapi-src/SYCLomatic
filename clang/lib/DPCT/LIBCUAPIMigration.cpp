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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"

namespace clang {
namespace dpct {

using namespace clang::ast_matchers;

void LIBCURule::processLIBCUUsingDirectiveDecl(const UsingDirectiveDecl *UDD) {
  llvm::StringRef NamespaceName = UDD->getNominatedNamespace()->getName();
  if (NamespaceName == "cuda") {
    if (const auto *NSD =
            dyn_cast<NamespaceDecl>(UDD->getNominatedNamespace())) {
      if (DpctGlobalInfo::isInCudaPath(NSD->getLocation())) {
        emplaceTransformation(new ReplaceDecl(UDD, ""));
      }
    }
  }
}

void LIBCURule::registerMatcher(ast_matchers::MatchFinder &MF) {
  {
    auto LIBCUMemberFuncHasNames = [&]() {
      return hasAnyName("load", "store", "exchange", "compare_exchange_weak",
                        "compare_exchange_strong", "fetch_add", "fetch_sub",
                        "at");
    };
    auto LIBCUTypesHasNames = [&]() {
      return hasAnyName("cuda::atomic", "cuda::std::atomic",
                        "cuda::std::array");
    };
    MF.addMatcher(cxxMemberCallExpr(
                      allOf(on(hasType(hasCanonicalType(qualType(hasDeclaration(
                                namedDecl(LIBCUTypesHasNames())))))),
                            callee(cxxMethodDecl(LIBCUMemberFuncHasNames()))))
                      .bind("MemberCall"),
                  this);
  }

  {
    auto LIBCUTypesNames = [&]() {
      return hasAnyName("atomic", "cuda::std::complex", "cuda::std::array",
                        "cuda::std::tuple");
    };
    MF.addMatcher(typeLoc(loc(hasCanonicalType(qualType(
                              hasDeclaration(namedDecl(LIBCUTypesNames()))))))
                      .bind("TypeLoc"),
                  this);
  }

  {
    auto LIBCUAPIHasNames = [&]() {
      return hasAnyName("cuda::atomic_thread_fence",
                        "cuda::std::atomic_thread_fence",
                        "cuda::std::make_tuple", "cuda::std::get");
    };
    MF.addMatcher(
        callExpr(callee(functionDecl(LIBCUAPIHasNames()))).bind("FuncCall"),
        this);
  }

  { MF.addMatcher(usingDirectiveDecl().bind("UsingDirectiveDecl"), this); }
}

void LIBCURule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {
  ExprAnalysis EA;
  if (const CXXMemberCallExpr *MC =
          getNodeAsType<CXXMemberCallExpr>(Result, "MemberCall")) {
    EA.analyze(MC);
  } else if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FuncCall")) {
    EA.analyze(CE);
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "TypeLoc")) {
    EA.analyze(*TL);
  } else if (auto UDD = getNodeAsType<UsingDirectiveDecl>(
                 Result, "UsingDirectiveDecl")) {
    processLIBCUUsingDirectiveDecl(UDD);
    return;
  } else {
    return;
  }
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

void LibraryTypeLocRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto TargetTypeName = [&]() {
    return hasAnyName("csrsv2Info_t", "cusparseSolvePolicy_t");
  };

  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(TargetTypeName())))))
          .bind("loc"),
      this);
}

void LibraryTypeLocRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto TL = getNodeAsType<TypeLoc>(Result, "loc")) {
    ExprAnalysis EA;
    EA.analyze(*TL);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

} // namespace dpct
} // namespace clang