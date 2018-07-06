//===--- Refinement.cpp ---------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "Refinement.h"

using namespace clang;
using namespace clang::cu2sycl;

// TODO: Remove this ItemLinearIDMatcher when some working matcher is added.
void ItemLinearIDMatcher::run(TransformSetTy &TS, const AnalysisManager &A) {
  /* Not implemented. */
}
