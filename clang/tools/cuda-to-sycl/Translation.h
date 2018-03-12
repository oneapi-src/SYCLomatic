//===--- Translation.h ----------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef CU2SYCL_TRANSLATION_H
#define CU2SYCL_TRANSLATION_H

#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "SourceTransformation.h"

namespace clang {
namespace cu2sycl {

class CudaMatcher : public ast_matchers::MatchFinder::MatchCallback {
protected:
  TransformSetTy *TransformSet = nullptr;

public:
  virtual void registerMatcher(ast_matchers::MatchFinder &MF) = 0;
  void setTransformSet(TransformSetTy &TS);
};

class ThreadIdxMatcher : public CudaMatcher {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

class BlockDimMatcher : public CudaMatcher {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

class TranslationManager {
  std::vector<std::unique_ptr<CudaMatcher>> Storage;
  ast_matchers::MatchFinder Matchers;

public:
  void emplaceCudaMatcher(CudaMatcher *M);
  void matchAST(ASTContext &Context, TransformSetTy &TS);
};

} // namespace cu2sycl
} // namespace clang

#endif // CU2SYCL_TRANSLATION_H
