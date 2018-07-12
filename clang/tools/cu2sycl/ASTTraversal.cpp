//===--- ASTTraversal.cpp -------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "ASTTraversal.h"

#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::cu2sycl;

void ThreadIdxMatcher::registerMatcher(MatchFinder &MF) {
  // TODO: match type of threadIdx?
  MF.addMatcher(
      memberExpr(hasObjectExpression(opaqueValueExpr(hasSourceExpression(
                     declRefExpr(to(varDecl(hasName("threadIdx"))))))))
          .bind("threadIdx"),
      this);
}

void ThreadIdxMatcher::run(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = Result.Nodes.getNodeAs<MemberExpr>("threadIdx");
  assert(ME && "Unknown result");

  ValueDecl *VD = ME->getMemberDecl();
  StringRef Member = VD->getName();
  unsigned Dimension;

  // TODO: match { ".x" ".y" ".z" } instead of this magic names
  if (Member == "__fetch_builtin_x")
    Dimension = 0;
  else if (Member == "__fetch_builtin_y")
    Dimension = 1;
  else if (Member == "__fetch_builtin_z")
    Dimension = 2;
  else
    llvm_unreachable("Unknown member name");

  // TODO: do not assume the argument is named "item"
  std::string Replacement = "item.get_local(" + std::to_string(Dimension) + ")";
  emplaceTransformation(new ReplaceExpr(ME, std::move(Replacement)));
}

void BlockDimMatcher::registerMatcher(MatchFinder &MF) {
  // TODO: match type of blockIdx?
  MF.addMatcher(
      memberExpr(hasObjectExpression(opaqueValueExpr(hasSourceExpression(
                     declRefExpr(to(varDecl(hasName("blockDim"))))))))
          .bind("blockDim"),
      this);
}

void BlockDimMatcher::run(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = Result.Nodes.getNodeAs<MemberExpr>("blockDim");
  assert(ME && "Unknown result");

  ValueDecl *VD = ME->getMemberDecl();
  StringRef Member = VD->getName();
  unsigned Dimension;

  // TODO: match { ".x" ".y" ".z" } instead of this magic names
  if (Member == "__fetch_builtin_x")
    Dimension = 0;
  else if (Member == "__fetch_builtin_y")
    Dimension = 1;
  else if (Member == "__fetch_builtin_z")
    Dimension = 2;
  else
    llvm_unreachable("Unknown member name");

  // TODO: do not assume the argument is named "item"
  std::string Replacement =
      "item.get_local_range().get(" + std::to_string(Dimension) + ")";
  emplaceTransformation(new ReplaceExpr(ME, std::move(Replacement)));
}

void ASTTraversalManager::matchAST(ASTContext &Context, TransformSetTy &TS) {
  for (auto &I : Storage) {
    I->registerMatcher(Matchers);
    if (auto TR = dyn_cast<TranslationRule>(&*I))
      TR->setTransformSet(TS);
  }
  Matchers.matchAST(Context);
}
