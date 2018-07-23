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

void IterationSpaceBuiltinRule::registerMatcher(MatchFinder &MF) {
  // TODO: check that threadIdx is not a local variable.
  MF.addMatcher(
      memberExpr(hasObjectExpression(opaqueValueExpr(hasSourceExpression(
                     declRefExpr(to(varDecl(hasAnyName("threadIdx", "blockDim",
                                                       "blockIdx", "gridDim"))
                                        .bind("varDecl")))))))
          .bind("memberExpr"),
      this);
}

void IterationSpaceBuiltinRule::run(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = Result.Nodes.getNodeAs<MemberExpr>("memberExpr");
  const VarDecl *VD = Result.Nodes.getNodeAs<VarDecl>("varDecl");
  assert(ME && VD && "Unknown result");

  ValueDecl *Field = ME->getMemberDecl();
  StringRef FieldName = Field->getName();
  unsigned Dimension;

  if (FieldName == "__fetch_builtin_x")
    Dimension = 0;
  else if (FieldName == "__fetch_builtin_y")
    Dimension = 1;
  else if (FieldName == "__fetch_builtin_z")
    Dimension = 2;
  else
    llvm_unreachable("Unknown field name");

  // TODO: do not assume the argument is named "item"
  std::string Replacement = "item";
  StringRef BuiltinName = VD->getName();

  if (BuiltinName == "threadIdx")
    Replacement += ".get_local(";
  else if (BuiltinName == "blockDim")
    Replacement += ".get_local_range().get(";
  else if (BuiltinName == "blockIdx")
    Replacement += ".get_group(";
  else if (BuiltinName == "gridDim")
    Replacement += ".get_num_groups(";
  else
    llvm_unreachable("Unknown builtin variable");

  Replacement += std::to_string(Dimension);
  Replacement += ")";
  emplaceTransformation(new ReplaceExpr(ME, std::move(Replacement)));
}

void FunctionAttrsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      functionDecl(anyOf(hasAttr(attr::CUDAGlobal), hasAttr(attr::CUDADevice),
                         hasAttr(attr::CUDAHost)))
          .bind("functionDecl"),
      this);
}

void FunctionAttrsRule::run(const MatchFinder::MatchResult &Result) {
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>("functionDecl");
  const AttrVec &AV = FD->getAttrs();

  for (const Attr *A : AV) {
    attr::Kind AK = A->getKind();
    if (AK == attr::CUDAGlobal || AK == attr::CUDADevice ||
        AK == attr::CUDAHost)
      emplaceTransformation(new RemoveAttr(A));
  }
}

void ASTTraversalManager::matchAST(ASTContext &Context, TransformSetTy &TS) {
  for (auto &I : Storage) {
    I->registerMatcher(Matchers);
    if (auto TR = dyn_cast<TranslationRule>(&*I))
      TR->setTransformSet(TS);
  }
  Matchers.matchAST(Context);
}
