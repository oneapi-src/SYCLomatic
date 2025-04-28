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
    return hasAnyName(
        "cudaGraphInstantiate", "cudaGraphLaunch", "cudaGraphExecDestroy",
        "cudaGraphAddEmptyNode", "cudaGraphAddDependencies",
        "cudaGraphExecUpdate", "cudaGraphNodeGetType", "cudaGraphGetNodes",
        "cudaGraphGetRootNodes", "cudaGraphDestroy", "cudaGraphAddKernelNode",
        "cudaGraphKernelNodeGetParams", "cudaGraphKernelNodeSetParams");
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

  MF.addMatcher(memberExpr(hasObjectExpression(hasType(
                               asString("cudaGraphExecUpdateResultInfo"))),
                           member(hasName("result")))
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
        if (auto BO = dyn_cast<BinaryOperator>(
                getParentAsAssignedBO(ME, *Result.Context))) {
          auto *LHS = BO->getLHS()->IgnoreCasts();
          if (auto *ME = dyn_cast<MemberExpr>(LHS)) {
            auto *Base = ME->getBase()->IgnoreImpCasts();
            if (auto *DRE = dyn_cast<DeclRefExpr>(Base)) {
              if (auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
                std::string VarName = VD->getNameAsString();
                auto *RHS = BO->getRHS()->IgnoreCasts();
                if (auto *RHS_DRE = dyn_cast<DeclRefExpr>(RHS)) {
                  if (auto *FD = dyn_cast<FunctionDecl>(RHS_DRE->getDecl())) {
                    std::string FuncName = FD->getNameAsString();
                    std::string WrapperName = FuncName;
                    std::string AccessOperator =
                        VD->getType()->isPointerType() ? "->" : ".";
                    std::string ReplacementStr =
                        VarName + AccessOperator +
                        "set_func("
                        "(void*) dpct::wrapper_register(&" +
                        WrapperName;
                    emplaceTransformation(
                        new ReplaceToken(BO->getBeginLoc(), BO->getEndLoc(),
                                         std::move(ReplacementStr)));
                    emplaceTransformation(new InsertAfterStmt(BO, ")"));
                    return;
                  }
                }
              }
            }
          }
        }
      }
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
  if (auto ME = getNodeAsType<MemberExpr>(Result, "execUpdateResult")) {
    auto MD = ME->getMemberDecl();
    const Expr *Base = ME->getBase();
    if (MD->getNameAsString() == "result") {
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

// Return the binary operator if E is the lhs of an assign expression,
// otherwise nullptr.
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
