//===--- SourceTransformation.cpp -----------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "SourceTransformation.h"

using namespace clang;
using namespace clang::cu2sycl;
using namespace clang::tooling;

Replacement CudaBlockDim::getReplacement(const SourceManager &SM) const {
  SourceRange SR = ME.getSourceRange();
  // TODO: do not assume the argument is named "item"
  return Replacement(SM, SR.getBegin(), getLength(SR, SM),
                     "item.get_local_range().get(" + std::to_string(Dimension) +
                         ")");
}

Replacement CudaThreadIdx::getReplacement(const SourceManager &SM) const {
  SourceRange SR = ME.getSourceRange();
  // TODO: do not assume the argument is named "item"
  return Replacement(SM, SR.getBegin(), getLength(SR, SM),
                     "item.get_local(" + std::to_string(Dimension) + ")");
}
