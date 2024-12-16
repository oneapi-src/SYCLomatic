//===-------------------- NvtxAPIMigration.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===//

#include "NvtxAPIMigration.h"
#include "AnalysisInfo.h"

namespace clang {
namespace dpct {

using namespace clang::ast_matchers;

void NvtxRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(callExpr(callee(functionDecl(
                             hasAnyName("nvtxNameCudaStreamA", "nvtxRangePushA",
                                        "nvtxRangePushW", "nvtxRangePop"))))
                    .bind("Call"),
                this);
}

void NvtxRule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {
  ExprAnalysis EA;
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "Call")) {
    EA.analyze(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

} // namespace dpct
} // namespace clang