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

class ASTTraversal;
using ASTTraversalConstructor = std::function<ASTTraversal *()>;

class ASTTraversalMetaInfo {
public:
  static std::unordered_map<const char *, std::string> &getNameTable() {
    static std::unordered_map<const char *, std::string> Table;
    return Table;
  }

  static std::unordered_map<std::string, const char *> &getIDTable() {
    static std::unordered_map<std::string, const char *> Table;
    return Table;
  }

  static const char *getID(const std::string &Name) {
    auto &IdTable = getIDTable();
    if (IdTable.find(Name) != IdTable.end()) {
      return IdTable[Name];
    }
    return nullptr;
  }

  static std::unordered_map<const char *, ASTTraversalConstructor> &getConstructorTable() {
    static std::unordered_map<const char *, ASTTraversalConstructor> FactoryMap;
    return FactoryMap;
  }

  static void registerRule(const char *ID, const std::string &Name,
                           ASTTraversalConstructor Factory) {
    getConstructorTable()[ID] = Factory;
    getIDTable()[Name] = ID;
    getNameTable()[ID] = Name;
  }
};

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

  /// Dereference.
  /// returns "deviceProp" for exression `&deviceProp`
  std::string DereferenceArg(const clang::Expr *E) {
    if (isa<clang::UnaryOperator>(E)) {
      const clang::UnaryOperator *arg = (const clang::UnaryOperator *)E;
      if (arg->getOpcode() == UO_AddrOf) {
        clang::DeclRefExpr *decl = ( clang::DeclRefExpr *)arg->getSubExpr();
        return  decl->getNameInfo().getName().getAsString();
      }
    }
    /// TODO implement dereference for the general case, not only for foo(&a).
    /// TODO for now, report "can't compile".
    return "";
  }

public:
  bool isTranslationRule() const override { return true; }
  static bool classof(const ASTTraversal *T) { return T->isTranslationRule(); }
};

template <typename T> class NamedTranslationRule : public TranslationRule {
public:
  static const char ID;
};

template <typename T> const char NamedTranslationRule<T>::ID(0);

/// Translation rule for iteration space builtin variables (threadIdx, etc).
class IterationSpaceBuiltinRule
    : public NamedTranslationRule<IterationSpaceBuiltinRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for CUDA function attributes.
///
/// This rule removes __global__, __device__ and __host__ function attributes.
class FunctionAttrsRule : public NamedTranslationRule<FunctionAttrsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for types replacements in var. declarations.
class TypeInVarDeclRule : public NamedTranslationRule<TypeInVarDeclRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  static const std::map<std::string, std::string> TypeNamesMap;
};

/// Translation rule for return types replacements.
class ReturnTypeRule : public NamedTranslationRule<ReturnTypeRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for removing of error hanlding if-stmt
class ErrorHandlingIfStmtRule
    : public NamedTranslationRule<ErrorHandlingIfStmtRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for cudaDeviceProp variables.
class DevicePropVarRule : public NamedTranslationRule<DevicePropVarRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  static const std::map<std::string, std::string> PropNamesMap;
};

// Translation rule for enums constants.
class EnumConstantRule : public NamedTranslationRule<EnumConstantRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  static const std::map<std::string, std::string> EnumNamesMap;
};

// Translation rule for cudaError enums constants.
class ErrorConstantsRule : public NamedTranslationRule<ErrorConstantsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for function calls.
class FunctionCallRule : public NamedTranslationRule<FunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

// Translation rule for memory management routine.
// Current implementation is intentionally simplistic. The following things need
// a more detailed design:
//   - interplay with error handling (possible solution is that we keep function
//     signature as close to original as possible, so return error codes when
//     original functions return them);
//   - SYCL memory buffers are typed. Using a "char" type is definitely a hack.
//     Using better type information requires some kind of global analysis and
//     heuristics, as well as a mechnism for user hint (like "treat all buffers
//     as float-typed")'
//   - interplay with streams need to be designed. I.e. cudaMemcpyAsync() need
//     to be defined;
//   - transformation rules are currently unordered, which create potential
//     ambiguity, so need to understand how to handle function call arguments,
//     which are modified by other rules.
//
// TODO:
//   - trigger include of runtime library.
class MemoryTranslationRule : public NamedTranslationRule<MemoryTranslationRule> {
  const Expr* stripConverts(const Expr *E) const;
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Pass manager for ASTTraversal instances.
class ASTTraversalManager {
  std::vector<std::unique_ptr<ASTTraversal>> Storage;
  ast_matchers::MatchFinder Matchers;

public:
  /// Add \a TR to the manager.
  ///
  /// The ownership of the TR is transferred to the ASTTraversalManager.
  void emplaceTranslationRule(const char *ID) {
    assert(ASTTraversalMetaInfo::getConstructorTable().find(ID) !=
           ASTTraversalMetaInfo::getConstructorTable().end());
    Storage.emplace_back(std::unique_ptr<ASTTraversal>(
                            ASTTraversalMetaInfo::getConstructorTable()[ID]()));
  }

  void emplaceAllRules() {
    for (auto &F : ASTTraversalMetaInfo::getConstructorTable()) {
      Storage.emplace_back(std::unique_ptr<ASTTraversal>(F.second()));
    }
  }

  /// Run all emplaced ASTTraversal's over the given AST and populate \a TS.
  void matchAST(ASTContext &Context, TransformSetTy &TS);
};

template <typename T> class RuleRegister {
public:
  RuleRegister(const char *ID, const std::string &Name) {
    ASTTraversalMetaInfo::registerRule(ID, Name, [] { return new T; });
  }
};

#define REGISTER_RULE(TYPE_NAME)                                               \
  RuleRegister<TYPE_NAME> g_##TYPE_NAME(&TYPE_NAME::ID, #TYPE_NAME);

} // namespace cu2sycl
} // namespace clang

#endif // CU2SYCL_AST_TRAVERSAL_H
