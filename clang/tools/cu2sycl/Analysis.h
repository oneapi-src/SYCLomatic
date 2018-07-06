//===--- Analysis.h -------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef CU2SYCL_ANALYSIS_H
#define CU2SYCL_ANALYSIS_H

#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace cu2sycl {

class Analysis : public ast_matchers::MatchFinder::MatchCallback {
public:
  virtual const char *getAnalysisID() const = 0;
  virtual void registerMatcher(ast_matchers::MatchFinder &M) = 0;
  virtual void dump() const = 0;
};

class AnalysisManager {
  std::unordered_map<const char *, std::unique_ptr<Analysis>> Storage;
  ast_matchers::MatchFinder Matchers;

public:
  void emplaceAnalysis(Analysis *A);

  Analysis *getAnalysis(const char *ID);

  template <typename T> Analysis *getAnalysis() const {
    return getAnalysis(&T::ID);
  }

  void matchAST(ASTContext &Context) { Matchers.matchAST(Context); }

  void dump() const;
};

// Traverses AST looking for kernel invocations.
struct KernelInvocationAnalysis : public Analysis {
  static const char ID;
  std::vector<const CUDAKernelCallExpr *> Invocations;

  const char *getAnalysisID() const override { return &ID; }

  void registerMatcher(ast_matchers::MatchFinder &MF) override;

  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  void dump() const override;
};

} // namespace cu2sycl
} // namespace clang

#endif // CU2SYCL_ANALYSIS_H
