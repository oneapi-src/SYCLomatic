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

const char CudaBlockDim::ID = 0;
const char CudaThreadIdx::ID = 0;
const char SyclItemLinearID::ID = 0;

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

Replacement SyclItemLinearID::getReplacement(const SourceManager &SM) const {
  // TODO: do not assume the argument is named "item"
  return Replacement(SM, Begin, getLength(Begin, End, SM),
                     "item.get_linear_id()");
}
