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

using namespace clang;
using namespace clang::cu2sycl;
using namespace clang::tooling;

Replacement ReplaceExpr::getReplacement(const SourceManager &SM) const {
  SourceRange SR = TheExpr->getSourceRange();
  return Replacement(SM, SR.getBegin(), getLength(SR, SM), ReplacementString);
}
