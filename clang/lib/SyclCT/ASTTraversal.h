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

#ifndef SYCLCT_AST_TRAVERSAL_H
#define SYCLCT_AST_TRAVERSAL_H

#include "Diagnostics.h"
#include "TextModification.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Frontend/CompilerInstance.h"

#include <sstream>
#include <unordered_set>

namespace clang {
namespace syclct {

enum RuleType {
  // rule applied to cude source
  ApplyToCudaFile = 1,

  // rule applied to cplusplus source
  ApplyToCppFile = 2,
};

typedef struct {
  int RType;
  std::vector<std::string> RulesDependon;
} CommonRuleProperty;

class IncludesCallbacks : public PPCallbacks {
  TransformSetTy &TransformSet;
  SourceManager &SM;
  std::unordered_set<std::string> SeenFiles;
  bool SyclHeaderInserted;

public:
  IncludesCallbacks(TransformSetTy &TransformSet, SourceManager &SM)
      : TransformSet(TransformSet), SM(SM), SyclHeaderInserted(false) {}
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;
};

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

  static std::unordered_map<const char *, ASTTraversalConstructor> &
  getConstructorTable() {
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

/// Base class for all compatibility tool-related AST traversals.
class ASTTraversal : public ast_matchers::MatchFinder::MatchCallback {
public:
  /// Specify what nodes need to be matched by this ASTTraversal.
  virtual void registerMatcher(ast_matchers::MatchFinder &MF) = 0;

  /// Specify what needs to be done for each matched node.
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override = 0;

  virtual bool isTranslationRule() const { return false; }
};

class ASTTraversalManager;

/// Base class for translation rules.
///
/// The purpose of a TranslationRule is to populate TransformSet with
/// SourceTransformation's.
class TranslationRule : public ASTTraversal {
  friend class ASTTraversalManager;
  ASTTraversalManager *TM;

  TransformSetTy *TransformSet = nullptr;
  void setTransformSet(TransformSetTy &TS) { TransformSet = &TS; }

protected:
  /// Add \a TM to the set of transformations.
  ///
  /// The ownership of the TM is transferred to the TransformSet.
  void emplaceTransformation(TextModification *TM) {
    TransformSet->emplace_back(TM);
  }

  const CompilerInstance &getCompilerInstance();

  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  void report(SourceLocation SL, IDTy MsgID, Ts &&... Vals) {
    DiagnosticsUtils::report<IDTy, Ts...>(SL, MsgID, getCompilerInstance(),
                                          TransformSet,
                                          std::forward<Ts>(Vals)...);
  }

  /// Dereference.
  /// returns "deviceProp" for exression `&deviceProp`
  std::string DereferenceArg(const clang::Expr *E) {
    if (isa<clang::UnaryOperator>(E)) {
      const clang::UnaryOperator *arg = (const clang::UnaryOperator *)E;
      if (arg->getOpcode() == UO_AddrOf) {
        clang::DeclRefExpr *decl = (clang::DeclRefExpr *)arg->getSubExpr();
        return decl->getNameInfo().getName().getAsString();
      }
    }
    /// TODO implement dereference for the general case, not only for foo(&a).
    /// TODO for now, report "can't compile".
    return "";
  }

public:
  bool isTranslationRule() const override { return true; }
  static bool classof(const ASTTraversal *T) { return T->isTranslationRule(); }

  // @RulesDependent : rules are sepeerate by ","
  void SetRuleProperty(int RType, std::string RulesDependent = "") {
    std::vector<std::string> RulesNames;
    // Separate rule string into list by comma
    if (RulesDependent != "") {
      std::size_t Current, Previous = 0;
      Current = RulesDependent.find(',');
      while (Current != std::string::npos) {
        RulesNames.push_back(
            RulesDependent.substr(Previous, Current - Previous));
        Previous = Current + 1;
        Current = RulesDependent.find(',', Previous);
      }
      RulesNames.push_back(RulesDependent.substr(Previous, Current - Previous));
    }
    RuleProperty.RType = RType;
    RuleProperty.RulesDependon = RulesNames;
  }
  CommonRuleProperty GetRuleProperty() { return RuleProperty; }

private:
  CommonRuleProperty RuleProperty;
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
  IterationSpaceBuiltinRule() { SetRuleProperty(ApplyToCudaFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for CUDA function attributes.
///
/// This rule removes __global__, __device__ and __host__ function attributes.
class FunctionAttrsRule : public NamedTranslationRule<FunctionAttrsRule> {
public:
  FunctionAttrsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for types replacements in var. declarations.
class TypeInVarDeclRule : public NamedTranslationRule<TypeInVarDeclRule> {
public:
  TypeInVarDeclRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  static const std::map<std::string, std::string> TypeNamesMap;
};

class ReplaceDim3CtorRule : public NamedTranslationRule<ReplaceDim3CtorRule> {
  std::pair<const CXXConstructExpr *, bool>
  rewriteSyntax(const ast_matchers::MatchFinder::MatchResult &Result);
  void rewriteArglist(const std::pair<const CXXConstructExpr *, bool> &);

public:
  ReplaceDim3CtorRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for return types replacements.
class ReturnTypeRule : public NamedTranslationRule<ReturnTypeRule> {
public:
  ReturnTypeRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for removing of error hanlding if-stmt
class ErrorHandlingIfStmtRule
    : public NamedTranslationRule<ErrorHandlingIfStmtRule> {
public:
  ErrorHandlingIfStmtRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for cudaDeviceProp variables.
class DevicePropVarRule : public NamedTranslationRule<DevicePropVarRule> {
public:
  DevicePropVarRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  static const std::map<std::string, std::string> PropNamesMap;
};

// Translation rule for enums constants.
class EnumConstantRule : public NamedTranslationRule<EnumConstantRule> {
public:
  EnumConstantRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  static const std::map<std::string, std::string> EnumNamesMap;
};

// Translation rule for cudaError enums constants.
class ErrorConstantsRule : public NamedTranslationRule<ErrorConstantsRule> {
public:
  ErrorConstantsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for function calls.
class FunctionCallRule : public NamedTranslationRule<FunctionCallRule> {
public:
  FunctionCallRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

class KernelCallRule : public NamedTranslationRule<KernelCallRule> {
public:
  KernelCallRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

// Translation rule for memory management routine.
// Current implementation is intentionally simplistic. The following things
// need a more detailed design:
//   - interplay with error handling (possible solution is that we keep
//   function
//     signature as close to original as possible, so return error codes when
//     original functions return them);
//   - SYCL memory buffers are typed. Using a "char" type is definitely a
//   hack.
//     Using better type information requires some kind of global analysis and
//     heuristics, as well as a mechnism for user hint (like "treat all
//     buffers as float-typed")'
//   - interplay with streams need to be designed. I.e. cudaMemcpyAsync() need
//     to be defined;
//   - transformation rules are currently unordered, which create potential
//     ambiguity, so need to understand how to handle function call arguments,
//     which are modified by other rules.
//
// TODO:
//   - trigger include of runtime library.
class MemoryTranslationRule
    : public NamedTranslationRule<MemoryTranslationRule> {
  const Expr *stripConverts(const Expr *E) const;

public:
  MemoryTranslationRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for inserting iteration space argument.
///
/// This rule inserts cl::sycl::nd_item<3> item as the first argument to
/// kernels.
class KernelIterationSpaceRule
    : public NamedTranslationRule<KernelIterationSpaceRule> {
public:
  KernelIterationSpaceRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Pass manager for ASTTraversal instances.
class ASTTraversalManager {
  std::vector<std::unique_ptr<ASTTraversal>> Storage;
  ast_matchers::MatchFinder Matchers;

public:
  const CompilerInstance &CI;
  // Set per matchAST invocation
  ASTContext *Context = nullptr;
  ASTTraversalManager(const CompilerInstance &CI) : CI(CI) {}
  /// Add \a TR to the manager.
  ///
  /// The ownership of the TR is transferred to the ASTTraversalManager.
  void emplaceTranslationRule(const char *ID) {
    assert(ASTTraversalMetaInfo::getConstructorTable().find(ID) !=
           ASTTraversalMetaInfo::getConstructorTable().end());
    Storage.emplace_back(std::unique_ptr<ASTTraversal>(
        ASTTraversalMetaInfo::getConstructorTable()[ID]()));
  }

  void emplaceAllRules(int SourceFileFlag) {
    for (auto &F : ASTTraversalMetaInfo::getConstructorTable()) {
      auto RuleObj = (TranslationRule *)F.second();
      CommonRuleProperty RuleProperty = RuleObj->GetRuleProperty();

      auto RType = RuleProperty.RType;
      auto RulesDependon = RuleProperty.RulesDependon;
      // To do:if RulesDependon is not null here, need order the rule set

      // Add rules current rule Name depends on
      for (auto const &RuleName : RulesDependon) {
        auto *ID = ASTTraversalMetaInfo::getID(RuleName);
        if (!ID) {
          llvm::errs() << "[ERROR] Rule\"" << RuleName << "\" not found\n";
          std::exit(1);
        }
        emplaceTranslationRule(ID);
      }

      if (RType & SourceFileFlag) {
        Storage.emplace_back(std::unique_ptr<ASTTraversal>(F.second()));
      }
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

} // namespace syclct
} // namespace clang

#endif // SYCLCT_AST_TRAVERSAL_H
