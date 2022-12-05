//===--------------- NCCLAPIMigration.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NCCLAPIMigration.h"

#include "ASTTraversal.h"
#include "ExprAnalysis.h"

#include <iostream>

using namespace clang::dpct;
using namespace clang::ast_matchers;

void clang::dpct::NCCLRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(namedDecl(
                            hasAnyName("ncclUniqueId", "ncclComm_t"))))))
                    .bind("type"),
                this);
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName(
                   "ncclGetVersion", "ncclGetUniqueId", "ncclCommInitRank"))))
          .bind("call"),
      this);
}

void clang::dpct::NCCLRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  ExprAnalysis EA;
  if (auto TL = getNodeAsType<TypeLoc>(Result, "type")) {
    EA.analyze(*TL);
  } else if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call")) {
    EA.analyze(CE);
  } else {
    return;
  }
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}
