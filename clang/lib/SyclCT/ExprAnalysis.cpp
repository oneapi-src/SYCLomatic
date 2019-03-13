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

namespace clang {
namespace syclct {

void ExprAnalysis::setExpr(const Expr *Expression) {
  E = Expression;
  ExprBeginOffset = SyclctGlobalInfo::getSourceManager()
                        .getDecomposedExpansionLoc(E->getBeginLoc())
                        .second;
  ExprString = getStmtSpelling(Expression, SyclctGlobalInfo::getContext());
}

std::pair<size_t, size_t> ExprAnalysis::getOffsetAndLength(const Expr *TE) {
  auto &SM = SyclctGlobalInfo::getSourceManager();
  auto Begin = SM.getDecomposedExpansionLoc(TE->getBeginLoc()).second;
  auto EndLoc = SM.getExpansionLoc(TE->getEndLoc());
  return std::pair<size_t, size_t>(
      Begin - ExprBeginOffset,
      SM.getDecomposedLoc(EndLoc).second - Begin +
          Lexer::MeasureTokenLength(
              EndLoc, SM, SyclctGlobalInfo::getContext().getLangOpts()));
}

void ArraySizeExprAnalysis::analysisDeclRefExpr(const DeclRefExpr *DRE) {
  if (DRE)
    if (auto TemplateDecl = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl()))
      addReplacement(DRE,
                     (*TemplateList)[TemplateDecl->getIndex()].getAsString());
}
} // namespace syclct
} // namespace clang