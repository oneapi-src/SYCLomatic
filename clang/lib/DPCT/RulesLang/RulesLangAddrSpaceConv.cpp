//===--------------- RulesLangAddrSpaceConv.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "RuleInfra/ExprAnalysis.h"
#include "RulesLang.h"
#include "Utility.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/Support/Casting.h"

namespace clang {
namespace dpct {

using namespace clang::ast_matchers;

void RulesLangAddrSpaceConvRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      callExpr(callee(functionDecl(hasName("__cvta_generic_to_shared"))))
          .bind("call"),
      this);
}

void RulesLangAddrSpaceConvRule::runRule(
    const MatchFinder::MatchResult &Result) {
  const auto *CE = getNodeAsType<CallExpr>(Result, "call");
  if (!CE)
    return;
  // Check if meets below conditions:
  // (1) A vardecl's init value is the "call" (or after type cast).
  // (2) The var is only used as the asm stmt parameter.

  // Check (1)
  const auto *DS = DpctGlobalInfo::findAncestor<DeclStmt>(CE);
  if (!DS)
    return;
  const auto *VD =
      DS->isSingleDecl() ? dyn_cast<VarDecl>(DS->getSingleDecl()) : nullptr;
  if (!VD)
    return;
  const auto *Init = VD->getInit();
  if (!Init)
    return;
  if (Init->IgnoreCasts() != CE)
    return;

  // Check (2)
  const auto *Ctx = VD->getDeclContext();
  const auto *FD = dyn_cast<FunctionDecl>(Ctx);
  if (!FD)
    return;
  if (!FD->hasBody())
    return;
  std::set<const clang::DeclRefExpr *> DREs =
      matchTargetDREInScope(VD, FD->getBody());
  if (DREs.size() != 1)
    return;
  const auto *DRE = *DREs.begin();
  const auto *AS = DpctGlobalInfo::findAncestor<AsmStmt>(DRE);
  if (!AS)
    return;

  // Generate replacement
  std::string ReplacementStr = "auto " + VD->getNameAsString() + " = " +
                               ExprAnalysis::ref(CE->getArg(0)) + ";";
  emplaceTransformation(new ReplaceDecl(VD, std::move(ReplacementStr)));
}

} // namespace dpct
} // namespace clang
