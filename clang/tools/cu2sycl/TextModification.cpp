//===--- TextModification.cpp ---------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "TextModification.h"

#include "clang/AST/Attr.h"

using namespace clang;
using namespace clang::cu2sycl;
using namespace clang::tooling;

Replacement ReplaceExpr::getReplacement(const SourceManager &SM) const {
  return Replacement(SM, TheExpr, ReplacementString);
}

Replacement RemoveAttr::getReplacement(const SourceManager &SM) const {
  SourceRange AttrRange = TheAttr->getRange();
  SourceLocation ARB = AttrRange.getBegin();
  SourceLocation ARE = AttrRange.getEnd();
  SourceLocation ExpB = SM.getExpansionLoc(ARB);
  // No need to invoke getExpansionLoc again if the location is the same.
  SourceLocation ExpE = (ARB == ARE) ? ExpB : SM.getExpansionLoc(ARE);
  return Replacement(SM, CharSourceRange::getTokenRange(ExpB, ExpE), "");
}
