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
#include "MapNames.h"
#include "TextModification.h"
#include "Utility.h"

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

  static const std::string getName(const char *ID) {
    auto &NameTable = getNameTable();
    if (NameTable.find(ID) != NameTable.end()) {
      return NameTable[ID];
    }
    std::string NullStr;
    return NullStr;
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

/// Pass manager for ASTTraversal instances.
class ASTTraversalManager {
  std::vector<std::unique_ptr<ASTTraversal>> Storage;
  ast_matchers::MatchFinder Matchers;

public:
  const CompilerInstance &CI;
  const std::string InRoot;
  // Set per matchAST invocation
  ASTContext *Context = nullptr;
  ASTTraversalManager(const CompilerInstance &CI, const std::string &IR)
      : CI(CI), InRoot(IR) {}
  /// Add \a TR to the manager.
  ///
  /// The ownership of the TR is transferred to the ASTTraversalManager.
  void emplaceTranslationRule(const char *ID) {
    assert(ASTTraversalMetaInfo::getConstructorTable().find(ID) !=
           ASTTraversalMetaInfo::getConstructorTable().end());
    Storage.emplace_back(std::unique_ptr<ASTTraversal>(
        ASTTraversalMetaInfo::getConstructorTable()[ID]()));
  }

  void emplaceAllRules(int SourceFileFlag);

  /// Run all emplaced ASTTraversal's over the given AST and populate \a TS.
  void matchAST(ASTContext &Context, TransformSetTy &TS);
};

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

  // Record of reported comments for each line
  // Map a source line (file name + line number) to a set of reported comment
  // ids
  static std::unordered_map<std::string,
                            std::unordered_set</* Comment ID */ int>>
      ReportedComment;

  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  void report(SourceLocation SL, IDTy MsgID, Ts &&... Vals) {
    DiagnosticsUtils::report<IDTy, Ts...>(SL, MsgID, getCompilerInstance(),
                                          TransformSet,
                                          std::forward<Ts>(Vals)...);
  }

  template <typename... Ts>
  void report(SourceLocation SL, Comments MsgID, Ts &&... Vals) {
    auto &SM = getCompilerInstance().getSourceManager();

    // Concatenate source file and line number (eg: xxx.cpp:4)
    std::string SourceAndLine;
    llvm::raw_string_ostream RSO(SourceAndLine);
    RSO << SM.getBufferName(SL) << ":" << SM.getPresumedLineNumber(SL);
    RSO.flush();

    if (ReportedComment.count(SourceAndLine) == 0) {
      // No comment has been reported for this line before.
      ReportedComment[SourceAndLine].insert(static_cast<int>(MsgID));
    } else if (ReportedComment[SourceAndLine].count(static_cast<int>(MsgID)) !=
               0) {
      // Same comment has been inserted for this line.
      // Avoid inserting duplicated comment for the same line.
      return;
    }
    DiagnosticsUtils::report<Comments>(SL, MsgID, getCompilerInstance(),
                                       TransformSet, std::forward<Ts>(Vals)...);
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

  const std::string &getItemName() {
    const static std::string ItemName =
        "item_" + getHashAsString(TM->InRoot).substr(0, 6);
    return ItemName;
  }

  const std::string &getHashID() {
    const static std::string HashID = getHashAsString(TM->InRoot).substr(0, 6);
    return HashID;
  }
  // Get node from match result map. And also check if the node's host file is
  // in the InRoot path.
  template <typename NodeType>
  const NodeType *
  getNodeAsType(const ast_matchers::MatchFinder::MatchResult &Result,
                const char *Name, bool CheckNode = true) {
    if (auto Node = Result.Nodes.getNodeAs<NodeType>(Name)) {
      if (checkNode(Result.SourceManager, Node->getBeginLoc(), CheckNode))
        return Node;
    }
    return nullptr;
  }

private:
  bool checkNode(SourceManager *SM, const SourceLocation &Begin,
                 bool CheckNode) {
    if (CheckNode)
      return isInRoot(SM, Begin) && !isReplaced(Begin.getRawEncoding());
    return true;
  }

  // Check if the node's host file is in the InRoot path.
  bool isInRoot(SourceManager *SM, const SourceLocation LS) {
    std::string FilePath = SM->getFilename(SM->getExpansionLoc(LS));
    makeCanonical(FilePath);
    return isChildPath(TM->InRoot, FilePath);
  }

  // Check if the location has been replaced by the same rule.
  bool isReplaced(unsigned LocationID) {
    for (auto ReplacedID : Replaced) {
      if (LocationID == ReplacedID)
        return true;
    }
    Replaced.push_back(LocationID);
    return false;
  }
  std::vector<unsigned> Replaced;

public:
  bool isTranslationRule() const override { return true; }
  static bool classof(const ASTTraversal *T) { return T->isTranslationRule(); }

  // @RulesDependent : rules are separated by ","
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
      std::string Rule = RulesDependent.substr(Previous, Current - Previous);
      Rule.erase(std::remove(Rule.begin(), Rule.end(), ' '),
                 Rule.end()); // Remove space if exists
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

/// Translation rule for CUDA class attributes.
///
/// This rule replace __align__ class attributes to __sycl_align__.
class AlignAttrsRule : public NamedTranslationRule<AlignAttrsRule> {
public:
  AlignAttrsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
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
};

/// Translation rule for types replacements in var. declarations.
class SyclStyleVectorRule : public NamedTranslationRule<SyclStyleVectorRule> {
public:
  SyclStyleVectorRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  static const std::map<std::string, std::string> MemberNamesMap;
};

/// rules replace the int2 to cuda sytle.
class SyclStyleVectorCtorRule
    : public NamedTranslationRule<SyclStyleVectorCtorRule> {
public:
  SyclStyleVectorCtorRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
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

/// Translation rule for dim3 types member fields replacements.
class Dim3MemberFieldsRule : public NamedTranslationRule<Dim3MemberFieldsRule> {
public:
  Dim3MemberFieldsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
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
  KernelCallRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile,
                    "SharedMemVarRule, ConstantMemVarRule, DeviceMemVarRule");
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for shared memory variables.
class SharedMemVarRule : public NamedTranslationRule<SharedMemVarRule> {
public:
  SharedMemVarRule() { SetRuleProperty(ApplyToCudaFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for constant memory variables.
class ConstantMemVarRule : public NamedTranslationRule<ConstantMemVarRule> {

  llvm::APInt Size;
  std::string TypeName;
  std::string ConstantVarName;
  bool IsArray;
  std::map<std::string, unsigned int> CntOfCVarPerKelfun;
  std::map<std::string, std::string> SizeOfConstMemVar;
  std::map<std::string, bool> CVarIsArray;

public:
  ConstantMemVarRule() { SetRuleProperty(ApplyToCudaFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation rule for device memory variables.
class DeviceMemVarRule : public NamedTranslationRule<DeviceMemVarRule> {
public:
  DeviceMemVarRule() { SetRuleProperty(ApplyToCudaFile); }
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
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile, "KernelCallRule");
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Name all unnamed types.
class UnnamedTypesRule : public NamedTranslationRule<UnnamedTypesRule> {
public:
  UnnamedTypesRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translation for math functions
class MathFunctionsRule : public NamedTranslationRule<MathFunctionsRule> {
public:
  MathFunctionsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  static const std::map<std::string, std::string> FunctionNamesMap;
};

/// Translation rule for replacing __syncthreads() function call.
///
/// This rule replace __syncthreads() with item.barrier()
class SyncThreadsRule : public NamedTranslationRule<SyncThreadsRule> {
public:
  SyncThreadsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Translate cudaFunctionAttributes to Sycl kernel info, defined in
/// runtime headers.
// TODO: only maxThreadsPerBlock is supported.
class KernelFunctionInfoRule
    : public NamedTranslationRule<KernelFunctionInfoRule> {
public:
  KernelFunctionInfoRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  static const std::map<std::string, std::string> AttributesNamesMap;
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
