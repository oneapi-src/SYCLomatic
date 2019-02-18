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
  auto Begin = E->getBeginLoc();
  ExprString = std::string(
      clang::syclct::SyclctGlobalInfo::getSourceManager().getCharacterData(
          Begin),
      E->getEndLoc().getRawEncoding() - Begin.getRawEncoding());
}

void ArraySizeExprAnalysis::analysisDeclRefExpr(const DeclRefExpr *DRE) {
  if (DRE)
    if (auto TemplateDecl = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl()))
      addReplacement(
          DRE, (*TemplateList)[TemplateDecl->getIndex()].getAsCallArgument());
}
} // namespace syclct
} // namespace clang