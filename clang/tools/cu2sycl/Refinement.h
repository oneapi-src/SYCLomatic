//===--- Refinement.h -----------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef CU2SYCL_REFINEMENT_H
#define CU2SYCL_REFINEMENT_H

#include "Analysis.h"
#include "SourceTransformation.h"

namespace clang {
namespace cu2sycl {
class Refinement {
public:
  virtual ~Refinement() {}
  virtual void run(TransformSetTy &TS, const AnalysisManager &A) = 0;
};

class ItemLinearIDMatcher : public Refinement {
public:
  void run(TransformSetTy &TS, const AnalysisManager &A) override;
};

class RefinementManager {
  std::vector<std::unique_ptr<Refinement>> Storage;

public:
  void emplaceOptimization(Refinement *SP) { Storage.emplace_back(SP); }

  void run(TransformSetTy &TS, const AnalysisManager &A) const {
    for (const auto &I : Storage) {
      I->run(TS, A);
    }
  }
};
} // namespace cu2sycl
} // namespace clang

#endif // CU2SYCL_REFINEMENT_H
