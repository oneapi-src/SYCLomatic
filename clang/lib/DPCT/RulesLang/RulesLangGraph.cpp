//===--------------- RulesLangGraph.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RuleInfra/ExprAnalysis.h"
#include "RuleInfra/MigrationStatistics.h"
#include "RulesLang.h"
#include "Utility.h"

#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Cuda.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::dpct;
using namespace clang::tooling;

extern clang::tooling::UnifiedPath
    DpctInstallPath; // Installation directory for this tool
extern DpctOption<opt, bool> ProcessAll;
extern DpctOption<opt, bool> AsyncHandler;

namespace clang {
namespace dpct {

void GraphRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName("cudaGraphInstantiate", "cudaGraphLaunch",
                      "cudaGraphExecDestroy", "cudaGraphAddEmptyNode",
                      "cudaGraphAddDependencies", "cudaGraphExecUpdate",
                      "cudaGraphNodeGetType", "cudaGraphGetNodes",
                      "cudaGraphGetRootNodes", "cudaGraphDestroy");
  };
  MF.addMatcher(
      callExpr(callee(functionDecl(functionName()))).bind("FunctionCall"),
      this);

  auto typeName = [&]() { return hasAnyName("cudaKernelNodeParams"); };
  MF.addMatcher(
      memberExpr(hasObjectExpression(hasType(type(hasUnqualifiedDesugaredType(
                     recordType(hasDeclaration(recordDecl(typeName()))))))))
          .bind("Type"),
      this);
}

void GraphRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto ME = getNodeAsType<MemberExpr>(Result, "Type")) {
    auto BaseTy = DpctGlobalInfo::getUnqualifiedTypeName(
        ME->getBase()->getType().getDesugaredType(*Result.Context),
        *Result.Context);
    auto MemberName = ME->getMemberNameInfo().getAsString();
    if (BaseTy == "cudaKernelNodeParams") {
      auto FieldName = KernelNodeParamNames[MemberName];
      if (FieldName.empty()) {
        report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
               DpctGlobalInfo::getOriginalTypeName(ME->getBase()->getType()) +
                   "::" + ME->getMemberDecl()->getName().str());
        return;
      }
      // if(FieldName == "func"){
      //   if(auto BO = dyn_cast<BinaryOperator>(getParentAsAssignedBO(ME, *Result.Context))){
      //     const Expr *RHS = BO->getRHS();
      //     const Expr *StrippedRHS = RHS->IgnoreParenCasts();
      //     std::string RHSStr;
      //     llvm::raw_string_ostream OS(RHSStr);
      //     std::cout <<"RHSSTR: " <<RHSStr << "\n";
      //     StrippedRHS->printPretty(OS, nullptr, Result.Context->getPrintingPolicy());


      //     // Create the replacement string using dpct::wrapper_register
      //     auto ReplacementStr = "set_func.dpct::wrapper_register(&" + RHSStr + "_wrapper).get()";
      //     std::cout<< "ReplacementSTR:" << ReplacementStr << "\n";

      //     // Replace the assignment with the set_func method call
      //     // emplaceTransformation(ReplaceMemberAssignAsSetMethod(
      //     //     BO, ME, FieldName, ReplacementStr));
      //     emplaceTransformation(new ReplaceText(getStmtExpansionSourceRange(RHS).getBegin(),
      //     ReplacementStr.length(),
      //                                         std::move(ReplacementStr)));
      //     return;
      //   }
      // }
      if (auto BO = getParentAsAssignedBO(ME, *Result.Context)) {
        StringRef ReplacedArg = "";
        emplaceTransformation(
            ReplaceMemberAssignAsSetMethod(BO, ME, FieldName, ReplacedArg));
      } else {
        emplaceTransformation(new RenameFieldInMemberExpr(
            ME, buildString("get_", FieldName, "()")));
      }
    }
    return;
  }
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    return;
  }
  ExprAnalysis EA(CE);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

const Expr *GraphRule::getParentAsAssignedBO(const Expr *E,
                                             ASTContext &Context) {
  auto Parents = Context.getParents(*E);
  if (Parents.size() > 0)
    return getAssignedBO(Parents[0].get<Expr>(), Context);
  return nullptr;
}

// Return the binary operator if E is the lhs of an assign expression, otherwise
// nullptr.
const Expr *GraphRule::getAssignedBO(const Expr *E, ASTContext &Context) {
  if (dyn_cast<MemberExpr>(E)) {
    // Continue finding parents when E is MemberExpr.
    return getParentAsAssignedBO(E, Context);
  } else if (auto ICE = dyn_cast<ImplicitCastExpr>(E)) {
    // Stop finding parents and return nullptr when E is ImplicitCastExpr,
    // except for ArrayToPointerDecay cast.
    if (ICE->getCastKind() == CK_ArrayToPointerDecay) {
      return getParentAsAssignedBO(E, Context);
    }
  } else if (auto ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    // Continue finding parents when E is ArraySubscriptExpr, and remove
    // subscript operator anyway for texture object's member.
    emplaceTransformation(new ReplaceToken(
        Lexer::getLocForEndOfToken(ASE->getLHS()->getEndLoc(), 0,
                                   Context.getSourceManager(),
                                   Context.getLangOpts()),
        ASE->getRBracketLoc(), ""));
    return getParentAsAssignedBO(E, Context);
  } else if (auto BO = dyn_cast<BinaryOperator>(E)) {
    // If E is BinaryOperator, return E only when it is assign expression,
    // otherwise return nullptr.
    if (BO->getOpcode() == BO_Assign)
      return BO;
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == OO_Equal) {
      return COCE;
    }
  }
  return nullptr;
}

} // namespace dpct
} // namespace clang
