//===--- ASTTraversal.h ---------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef CU2SYCL_AST_TRAVERSAL_H
#define CU2SYCL_AST_TRAVERSAL_H

#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "TextModification.h"

namespace clang {
namespace cu2sycl {

/// Base class for all translator-related AST traversals.
class ASTTraversal : public ast_matchers::MatchFinder::MatchCallback {
public:
  /// Specify what nodes need to be matched by this ASTTraversal.
  virtual void registerMatcher(ast_matchers::MatchFinder &MF) = 0;

  /// Specify what needs to be done for each matched node.
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override = 0;

  virtual bool isTranslationRule() const { return false; }
};

/// Base class for translation rules.
///
/// The purpose of a TranslationRule is to populate TransformSet with
/// SourceTransformation's.
class TranslationRule : public ASTTraversal {
  friend class ASTTraversalManager;

  TransformSetTy *TransformSet = nullptr;
  void setTransformSet(TransformSetTy &TS) { TransformSet = &TS; }

protected:
  /// Add \a TM to the set of transformations.
  ///
  /// The ownership of the TM is transferred to the TransformSet.
  void emplaceTransformation(TextModification *TM) {
    TransformSet->emplace_back(TM);
  }

public:
  bool isTranslationRule() const override { return true; }
  static bool classof(const ASTTraversal *T) { return T->isTranslationRule(); }
};

/// Translation rule for iteration space builtin variables (threadIdx, etc).
class IterationSpaceBuiltinRule : public TranslationRule {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for CUDA function attributes.
///
/// This rule removes __global__, __device__ and __host__ function attributes.
class FunctionAttrsRule : public TranslationRule {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for types replacements in var. declarations.
class TypeInVarDeclRule : public TranslationRule {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for removing of error hanlding if-stmt
class ErrorHandlingIfStmtRule : public TranslationRule {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for cudaDeviceProp variables.
class DevicePropVarRule : public TranslationRule {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  static const std::map<std::string, std::string> PropNamesMap;
};

// Translation rule for enums constants.
class EnumConstantRule : public TranslationRule {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  static const std::map<std::string, std::string> EnumNamesMap;
};

/// Pass manager for ASTTraversal instances.
class ASTTraversalManager {
  std::vector<std::unique_ptr<ASTTraversal>> Storage;
  ast_matchers::MatchFinder Matchers;

public:
  /// Add \a TR to the manager.
  ///
  /// The ownership of the TR is transferred to the ASTTraversalManager.
  void emplaceTranslationRule(TranslationRule *TR) { Storage.emplace_back(TR); }

  /// Run all emplaced ASTTraversal's over the given AST and populate \a TS.
  void matchAST(ASTContext &Context, TransformSetTy &TS);
};

} // namespace cu2sycl
} // namespace clang

#endif // CU2SYCL_AST_TRAVERSAL_H
