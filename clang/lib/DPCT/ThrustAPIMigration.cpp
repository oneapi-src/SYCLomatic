//===--------------- ThrustAPIMigration.cpp
//---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThrustAPIMigration.h"

#include "ASTTraversal.h"
#include "ExprAnalysis.h"

namespace clang {
namespace dpct {

using namespace clang;
using namespace clang::dpct;
using namespace clang::ast_matchers;

void ThrustRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  // API register
  auto ThrustAPIHasNames = [&]() {
    return hasAnyName("log10", "sqrt", "pow", "sin", "cos", "tan", "asin",
                      "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh",
                      "atanh", "abs", "polar", "exp", "log");
  };
  MF.addMatcher(callExpr(callee(functionDecl(allOf(
                             hasDeclContext(namespaceDecl(hasName("thrust"))),
                             ThrustAPIHasNames()))))
                    .bind("thrustFuncCall"),
                this);

  // TYPE register
  auto ThrustTypeHasNames = [&]() { return hasAnyName("thrust::complex"); };
  MF.addMatcher(typeLoc(loc(hasCanonicalType(qualType(
                            hasDeclaration(namedDecl(ThrustTypeHasNames()))))))
                    .bind("thrustTypeLoc"),
                this);
  
}

void ThrustRule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {
  ExprAnalysis EA;
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "thrustFuncCall")) {
    EA.analyze(CE);
  }  else if (auto TL = getNodeAsType<TypeLoc>(Result, "thrustTypeLoc")) {
    EA.analyze(*TL);
  }else {
    return;
  }
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}
} // namespace dpct
} // namespace clang