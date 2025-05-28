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

void GraphAnalysisRule::registerMatcher(MatchFinder &MF) {
  auto kernelNodeTypeName = [&]() {
    return hasAnyName("cudaKernelNodeParams");
  };
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(type(hasUnqualifiedDesugaredType(
              recordType(hasDeclaration(recordDecl(kernelNodeTypeName()))))))))
          .bind("KernelNodeType"),
      this);
}

void GraphAnalysisRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto ME = getNodeAsType<MemberExpr>(Result, "KernelNodeType")) {
    auto BaseTy = DpctGlobalInfo::getUnqualifiedTypeName(
        ME->getBase()->getType().getDesugaredType(*Result.Context),
        *Result.Context);
    auto MemberName = ME->getMemberNameInfo().getAsString();
    if (BaseTy == "cudaKernelNodeParams") {
      DpctGlobalInfo::setUseWrapperRegisterFnPtr();
    }
  }
}

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

  MF.addMatcher(
      memberExpr(hasObjectExpression(
                     hasType(asString("cudaGraphExecUpdateResultInfo"))),
                 member(hasAnyName("result", "errorNode", "errorFromNode")))
          .bind("execUpdateResult"),
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
      if (FieldName == "func") {
        auto BinaryOp = getParentAsAssignedBO(ME, *Result.Context, this);
        if (!BinaryOp) {
          emplaceTransformation(new RenameFieldInMemberExpr(
              ME, buildString("get_", FieldName, "()")));
          return;
        }
        auto BO = dyn_cast<BinaryOperator>(BinaryOp);
        if (!BO) {
          return;
        }
        auto *LHS = BO->getLHS()->IgnoreCasts();
        auto *ME_LHS = dyn_cast<MemberExpr>(LHS);
        if (!ME_LHS) {
          return;
        }
        auto *Base = ME_LHS->getBase()->IgnoreImpCasts();
        auto *DRE = dyn_cast<DeclRefExpr>(Base);
        if (!DRE) {
          return;
        }
        auto *VD = dyn_cast<VarDecl>(DRE->getDecl());
        if (!VD) {
          return;
        }
        std::string VarName = VD->getNameAsString();
        auto *RHS = BO->getRHS()->IgnoreCasts();
        auto *RHS_DRE = dyn_cast<DeclRefExpr>(RHS);
        if (!RHS_DRE) {
          return;
        }
        if (auto RhsVarDecl = dyn_cast<VarDecl>(RHS_DRE->getDecl())) {
          StringRef ReplacedArg = "";
          emplaceTransformation(
              ReplaceMemberAssignAsSetMethod(BO, ME, FieldName, ReplacedArg));
          return;
        }
        auto *FD = dyn_cast<FunctionDecl>(RHS_DRE->getDecl());
        if (!FD) {
          return;
        }
        std::string FuncName = FD->getNameAsString();
        std::string WrapperName = FuncName;
        std::string AccessOperator =
            VD->getType()->isPointerType() ? "->" : ".";
        std::string ReplacementStr =
            VarName + AccessOperator +
            "set_func((void*) dpct::wrapper_register(&" + WrapperName;
        emplaceTransformation(new ReplaceToken(
            BO->getBeginLoc(), BO->getEndLoc(), std::move(ReplacementStr)));
        emplaceTransformation(new InsertAfterStmt(BO, ")"));
      }
      if (auto BO = getParentAsAssignedBO(ME, *Result.Context, this)) {
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
  if (auto ME = getNodeAsType<MemberExpr>(Result, "execUpdateResult")) {
    auto MD = ME->getMemberDecl();
    const Expr *Base = ME->getBase();
    std::string MemberName = MD->getNameAsString();
    if (MemberName == "result" || MemberName == "errorNode" ||
        MemberName == "errorFromNode") {
      if (auto *DRE = dyn_cast<DeclRefExpr>(Base)) {
        SourceLocation StartLoc = Base->getBeginLoc();
        SourceLocation EndLoc = ME->getEndLoc();
        const SourceManager &SM = *Result.SourceManager;
        EndLoc = Lexer::getLocForEndOfToken(EndLoc, 0, SM,
                                            Result.Context->getLangOpts());
        std::string VarNameStr = DRE->getNameInfo().getAsString();
        emplaceTransformation(
            new ReplaceToken(StartLoc, EndLoc, std::move(VarNameStr)));
      }
      return;
    }
  }
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    return;
  }
  ExprAnalysis EA(CE);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

} // namespace dpct
} // namespace clang
