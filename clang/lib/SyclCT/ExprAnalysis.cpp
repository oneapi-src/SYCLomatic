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

const std::vector<TemplateArgumentInfo> ArraySizeExprAnalysis::NullList;
std::map<const Expr *, std::string> ArgumentAnalysis::DefaultArgMap;

std::pair<size_t, size_t>
StringReplacements::getOffsetAndLength(SourceLocation SL) {
  if (SL.isInvalid())
    return std::pair<size_t, size_t>(0, SourceStr.length());
  auto &SM = Context.getSourceManager();
  auto Loc = SM.getExpansionLoc(SL);
  return std::pair<size_t, size_t>(
      SM.getDecomposedLoc(Loc).second - SourceBegin,
      Lexer::MeasureTokenLength(Loc, SM, Context.getLangOpts()));
}

std::pair<size_t, size_t>
StringReplacements::getOffsetAndLength(SourceLocation BeginLoc,
                                       SourceLocation EndLoc) {
  auto &SM = Context.getSourceManager();
  if (EndLoc.isValid()) {
    auto Begin = SM.getFileOffset(SM.getExpansionLoc(BeginLoc)) - SourceBegin;
    auto End = getOffsetAndLength(EndLoc);
    return std::pair<size_t, size_t>(Begin, End.first - Begin + End.second);
  }
  return getOffsetAndLength(BeginLoc);
}

void StringReplacements::init(const Expr *E) {
  ReplMap.clear();
  if (E && E->getBeginLoc().isValid()) {
    SourceBegin = Context.getSourceManager()
                      .getDecomposedExpansionLoc(E->getBeginLoc())
                      .second;
    SourceStr = getStmtSpelling(E, Context);
  } else {
    SourceBegin = 0;
    SourceStr.clear();
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
    : Context(SyclctGlobalInfo::getContext()), ReplSet(Context) {
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

void ArraySizeExprAnalysis::analysisExpression(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
    ANALYSIS_EXPR(DeclRefExpr)
  default:
    return Base::analysisExpression(Expression);
  }
}

void ArraySizeExprAnalysis::analysisExpr(const DeclRefExpr *DRE) {
  if (auto TemplateDecl = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl()))
    addReplacement(DRE, TemplateList[TemplateDecl->getIndex()].getAsString());
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
}
} // namespace syclct
} // namespace clang