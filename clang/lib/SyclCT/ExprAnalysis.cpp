//===--- ExprAnalysis.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "ExprAnalysis.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprOpenMP.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtGraphTraits.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtOpenMP.h"

namespace clang {
namespace syclct {

#define ANALYSIS_EXPR(EXPR)                                                    \
  case Stmt::EXPR##Class:                                                      \
    return analysisExpr(static_cast<const EXPR *>(Expression));

std::map<const Expr *, std::string> ArgumentAnalysis::DefaultArgMap;

void TemplateDependentReplacement::replace(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  SourceStr.replace(Offset, Length, TemplateList[TemplateIndex].getAsString());
}

TemplateDependentStringInfo::TemplateDependentStringInfo(
    const std::string &SrcStr,
    const std::map<size_t, std::shared_ptr<TemplateDependentReplacement>>
        &InTDRs)
    : SourceStr(SrcStr) {
  for (auto TDR : InTDRs)
    TDRs.emplace_back(TDR.second->alterSource(SourceStr));
}

std::string TemplateDependentStringInfo::getReplacedString(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  std::string SrcStr(SourceStr);
  for (auto Itr = TDRs.rbegin(); Itr != TDRs.rend(); ++Itr)
    (*Itr)->replace(TemplateList);
  std::swap(SrcStr, SourceStr);
  return SrcStr;
}

SourceLocation ExprAnalysis::getExprLocation(SourceLocation Loc) {
  if (SM.isMacroArgExpansion(Loc))
    return SM.getSpellingLoc(Loc);
  else
    return SM.getExpansionLoc(Loc);
}

std::pair<size_t, size_t> ExprAnalysis::getOffsetAndLength(SourceLocation Loc) {
  if (Loc.isInvalid())
    return std::pair<size_t, size_t>(0, SrcLength);
  Loc = getExprLocation(Loc);
  return std::pair<size_t, size_t>(
      getOffset(Loc),
      Lexer::MeasureTokenLength(Loc, SM, Context.getLangOpts()));
}

std::pair<size_t, size_t>
ExprAnalysis::getOffsetAndLength(SourceLocation BeginLoc,
                                 SourceLocation EndLoc) {
  if (EndLoc.isValid()) {
    auto Begin = getOffset(getExprLocation(BeginLoc));
    auto End = getOffsetAndLength(EndLoc);
    return std::pair<size_t, size_t>(Begin, End.first - Begin + End.second);
  }
  return getOffsetAndLength(BeginLoc);
}

void ExprAnalysis::initExpression(const Expr *Expression) {
  E = Expression;
  SrcBegin = 0;
  if (E && E->getBeginLoc().isValid()) {
    std::tie(SrcBegin, SrcLength) =
        getOffsetAndLength(E->getBeginLoc(), E->getEndLoc());
    ReplSet.init(std::string(
        SM.getCharacterData(getExprLocation(E->getBeginLoc())), SrcLength));
  } else {
    SrcLength = 0;
    ReplSet.init("");
  }
}

void StringReplacements::replaceString() {
  SourceStr.reserve(SourceStr.length() + ShiftLength);
  auto Itr = ReplMap.rbegin();
  while (Itr != ReplMap.rend()) {
    Itr->second->replaceString();
    ++Itr;
  }
  ReplMap.clear();
}

ExprAnalysis::ExprAnalysis(const Expr *Expression)
    : Context(SyclctGlobalInfo::getContext()),
      SM(SyclctGlobalInfo::getSourceManager()) {
  initExpression(Expression);
}

void ExprAnalysis::analysisExpression(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
#define STMT(CLASS, PARENT) ANALYSIS_EXPR(CLASS)
#define STMT_RANGE(BASE, FIRST, LAST)
#define LAST_STMT_RANGE(BASE, FIRST, LAST)
#define ABSTRACT_STMT(STMT)
#include "clang/AST/StmtNodes.inc"
  default:
    return;
  }
}

void ExprAnalysis::analysisExpr(const CXXConstructExpr *Ctor) {
  const std::string Dim3Constructor = "dim3";
  if (Ctor->getConstructor()->getDeclName().getAsString() == Dim3Constructor) {
    std::string ArgsString = "cl::sycl::range<3>(";
    ArgumentAnalysis A;
    for (auto Arg : Ctor->arguments()) {
      A.analysis(Arg);
      ArgsString += A.getReplacedString() + ", ";
    }
    ArgsString.replace(ArgsString.length() - 2, 2, ")");
    addReplacement(Ctor, ArgsString);
  }
}

void ExprAnalysis::analysisExpr(const MemberExpr *ME) {
  TypeInfo Ty(ME->getBase()->getType());
  if (Ty.getBaseName() == "cl::sycl::range<3>")
    addReplacement(
        ME->getOperatorLoc(), ME->getMemberLoc(),
        MapNames::findReplacedName(MapNames::Dim3MemberNamesMap,
                                   ME->getMemberNameInfo().getAsString()));
}

void ExprAnalysis::analysisExpr(const UnaryExprOrTypeTraitExpr *UETT) {
  if (UETT->getKind() == UnaryExprOrTypeTrait::UETT_SizeOf) {
    auto TyInfo = UETT->getArgumentTypeInfo();
    TypeInfo Ty(TyInfo->getType());
    addReplacement(TyInfo->getTypeLoc().getBeginLoc(),
                   UETT->getRParenLoc().getLocWithOffset(-1), Ty.getBaseName());
  }
}
void ExprAnalysis::analysisExpr(const CallExpr *CE) {
  auto Func = CE->getDirectCallee();
  const std::string FuncName = CE->getDirectCallee()->getNameAsString();
  if (MathFunctionsRule::SingleDoubleFunctionNamesMap.count(FuncName) != 0) {
    std::string NewFuncName =
        MathFunctionsRule::SingleDoubleFunctionNamesMap.at(FuncName);
    std::string ArgsString = "(";
    ArgumentAnalysis A;
    for (auto Arg : CE->arguments()) {
      A.analysis(Arg);
      ArgsString += A.getReplacedString() + ", ";
    }
    ArgsString.replace(ArgsString.length() - 2, 2, ")");
    addReplacement(CE, NewFuncName + ArgsString);
  }
}
const std::string &ArgumentAnalysis::getDefaultArgument(const Expr *E) {
  auto &Str = DefaultArgMap[E];
  if (Str.empty())
    Str = getStmtSpelling(E, SyclctGlobalInfo::getContext());
  return Str;
}

void KernelArgumentAnalysis::analysisExpression(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
    ANALYSIS_EXPR(DeclRefExpr)
  default:
    return ExprAnalysis::analysisExpression(Expression);
  }
}

void KernelArgumentAnalysis::analysisExpr(const DeclRefExpr *DRE) {
  if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    auto &VI = DeclMap[VD];
    if (!VI) {
      auto LocInfo = SyclctGlobalInfo::getLocInfo(VD);
      VI = std::make_shared<VarInfo>(LocInfo.second, LocInfo.first, VD);
    }
  }
  Base::analysisExpr(DRE);
}
} // namespace syclct
} // namespace clang