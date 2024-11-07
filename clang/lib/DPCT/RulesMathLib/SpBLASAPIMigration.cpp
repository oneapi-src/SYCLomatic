//===------------------ SpBLASAPIMigration.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SpBLASAPIMigration.h"

namespace clang {
namespace dpct {

using namespace clang::ast_matchers;

void SpBLASTypeLocRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto TargetTypeName = [&]() {
    return hasAnyName("csrsv2Info_t", "cusparseSolvePolicy_t",
                      "cusparseAction_t");
  };

  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(TargetTypeName())))))
          .bind("loc"),
      this);
}

void SpBLASTypeLocRule::runRule(
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