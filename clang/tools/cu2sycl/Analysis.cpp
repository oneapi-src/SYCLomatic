//===--- Analysis.cpp -----------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "Analysis.h"

#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::cu2sycl;

void AnalysisManager::emplaceAnalysis(Analysis *A) {
  Storage.emplace(A->getAnalysisID(), A);
  A->registerMatcher(Matchers);
}

Analysis *AnalysisManager::getAnalysis(const char *ID) {
  auto Found = Storage.find(ID);
  assert(Found != Storage.end() && "Nothing found");
  return Found->second.get();
}

LLVM_DUMP_METHOD void AnalysisManager::dump() const {
  for (const auto &I : Storage) {
    I.second->dump();
  }
}

void KernelInvocationAnalysis::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cudaKernelCallExpr().bind("kernel"), this);
}

void KernelInvocationAnalysis::run(const MatchFinder::MatchResult &Result) {
  const CUDAKernelCallExpr *CE =
      Result.Nodes.getNodeAs<CUDAKernelCallExpr>("kernel");
  assert(CE && "Unknown result");
  Invocations.push_back(CE);
}

LLVM_DUMP_METHOD void KernelInvocationAnalysis::dump() const {
  for (const auto &I : Invocations) {
    I->dump();
  }
}

const char KernelInvocationAnalysis::ID = 0;
