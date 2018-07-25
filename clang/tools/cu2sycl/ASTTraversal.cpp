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
  emplaceTransformation(new ReplaceStmt(ME, std::move(Replacement)));
}

void ErrorHandlingIfStmtRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      // Match if-statement that has no else and has a condition of either an
      // operator!= or a variable of type enum.
      ifStmt(unless(hasElse(anything())),
             hasCondition(
                 anyOf(binaryOperator(hasOperatorName("!=")).bind("op!="),
                       ignoringImpCasts(
                           declRefExpr(hasType(hasCanonicalType(enumType())))
                               .bind("var")))))
          .bind("errIf"),
      this);
}

static bool isVarRef(const Expr *E) {
  if (auto D = dyn_cast<DeclRefExpr>(E))
    return isa<VarDecl>(D->getDecl());
  else
    return false;
}

static std::string getVarType(const Expr *E) {
  return E->getType().getCanonicalType().getUnqualifiedType().getAsString();
}

static bool isCudaFailureCheck(const BinaryOperator *Op) {
  auto Lhs = Op->getLHS()->IgnoreImplicit();
  auto Rhs = Op->getRHS()->IgnoreImplicit();

  const Expr *Literal = nullptr;
  if (isVarRef(Lhs) && getVarType(Lhs) == "enum cudaError")
    Literal = Rhs;
  else if (isVarRef(Rhs) && getVarType(Rhs) == "enum cudaError")
    Literal = Lhs;
  else
    return false;

  if (auto IntLit = dyn_cast<IntegerLiteral>(Literal)) {
    if (IntLit->getValue() != 0)
      return false;
  } else if (auto D = dyn_cast<DeclRefExpr>(Literal)) {
    auto EnumDecl = dyn_cast<EnumConstantDecl>(D->getDecl());
    if (!EnumDecl)
      return false;
    // Check for cudaSuccess or CUDA_SUCCESS.
    if (EnumDecl->getInitVal() != 0)
      return false;
  } else {
    // The expression is neither an int literal nor an enum value.
    return false;
  }

  return true;
}

static bool isCudaFailureCheck(const DeclRefExpr *E) {
  return isVarRef(E) && getVarType(E) == "enum cudaError";
}

static bool isErrorHandling(const Stmt *Block) {
  // TODO: For now our definition of error handling is an empty Then-clause.
  return Block->child_begin() == Block->child_end();
}

void ErrorHandlingIfStmtRule::run(const MatchFinder::MatchResult &Result) {
  if (auto Op = Result.Nodes.getNodeAs<BinaryOperator>("op!=")) {
    if (!isCudaFailureCheck(Op))
      return;
  } else {
    auto CondVar = Result.Nodes.getNodeAs<DeclRefExpr>("var");
    if (!isCudaFailureCheck(CondVar))
      return;
  }

  auto If = Result.Nodes.getNodeAs<IfStmt>("errIf");
  if (!isErrorHandling(If->getThen()))
    return;

  emplaceTransformation(new ReplaceStmt(If, ""));
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

// Rule for types replacements in var. declarations.
void TypeInVarDeclRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(varDecl(hasType(cxxRecordDecl(hasName("cudaDeviceProp"))))
                    .bind("TypeInVarDecl"),

                this);
}

void TypeInVarDeclRule::run(const MatchFinder::MatchResult &Result) {
  const VarDecl *D = Result.Nodes.getNodeAs<VarDecl>("TypeInVarDecl");
  emplaceTransformation(
      new ReplaceTypeInVarDecl(D, "cu2sycl::sycl_device_info"));
}

// Rule for cudaDeviceProp variables.
void DevicePropVarRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(qualType(hasCanonicalType(recordType(
              hasDeclaration(cxxRecordDecl(hasName("cudaDeviceProp")))))))))
          .bind("DevicePropVar"),
      this);
}

void DevicePropVarRule::run(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = Result.Nodes.getNodeAs<MemberExpr>("DevicePropVar");
  auto Search = PropNamesMap.find(ME->getMemberNameInfo().getAsString());
  if (Search == PropNamesMap.end()) {
    // TODO report translation error
    return;
  }
  emplaceTransformation(new RenameFieldInMemberExpr(ME, Search->second + "()"));
}

// Rule for enums constants.
void EnumConstantRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(
                                hasType(enumDecl(hasName("cudaComputeMode"))))))
                    .bind("EnumConstant"),
                this);
}

void EnumConstantRule::run(const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *E = Result.Nodes.getNodeAs<DeclRefExpr>("EnumConstant");
  assert(E && "Unknown result");
  auto Search = EnumNamesMap.find(E->getNameInfo().getName().getAsString());
  if (Search == EnumNamesMap.end()) {
    // TODO report translation error
    return;
  }
  emplaceTransformation(new ReplaceStmt(E, "cu2sycl::" + Search->second));
}

void ASTTraversalManager::matchAST(ASTContext &Context, TransformSetTy &TS) {
  for (auto &I : Storage) {
    I->registerMatcher(Matchers);
    if (auto TR = dyn_cast<TranslationRule>(&*I))
      TR->setTransformSet(TS);
  }
  Matchers.matchAST(Context);
}
