//===--- ASTTraversal.h ---------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_AST_TRAVERSAL_H
#define DPCT_AST_TRAVERSAL_H

#include "AnalysisInfo.h"
#include "Debug.h"
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
namespace dpct {

// TODO: implement one of this for each source language.
enum RuleType {
  ApplyToCudaFile = 1,
  ApplyToCppFile = 2,
};

typedef struct {
  int RType;
  std::vector<std::string> RulesDependon;
} CommonRuleProperty;

class ASTTraversalManager;

/// Migraiton rules at the pre-processing stages, e.g. macro rewriting and
/// including directives rewriting.
class IncludesCallbacks : public PPCallbacks {
  TransformSetTy &TransformSet;
  IncludeMapSetTy &IncludeMapSet;
  SourceManager &SM;

  std::unordered_set<std::string> SeenFiles;
  bool DpstdHeaderInserted;
  ASTTraversalManager &ATM;

public:
  IncludesCallbacks(TransformSetTy &TransformSet,
                    IncludeMapSetTy &IncludeMapSet, SourceManager &SM,
                    ASTTraversalManager &ATM)
      : TransformSet(TransformSet), IncludeMapSet(IncludeMapSet), SM(SM),
        DpstdHeaderInserted(false), ATM(ATM) {}
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;
  /// Hook called whenever a macro definition is seen.
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override;
  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override;
  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MD) override;
  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override;
  // TODO: implement one of this for each source language.
  void ReplaceCuMacro(const Token &MacroNameTok);
  void ReplaceCuMacro(SourceRange ConditionRange);
  void Defined(const Token &MacroNameTok, const MacroDefinition &MD,
               SourceRange Range) override;
  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID = FileID()) override;
  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind ConditionValue) override;
  void Elif(SourceLocation Loc, SourceRange ConditionRange,
            ConditionValueKind ConditionValue, SourceLocation IfLoc) override;
  bool ShouldEnter(StringRef FileName, bool IsAngled) override;

private:
  /// e.g. "__launch_bounds(32, 32)  void foo()"
  /// Result is "void foo()"
  TextModification *removeMacroInvocationAndTrailingSpaces(SourceRange Range);
};

class ASTTraversal;
using ASTTraversalConstructor = std::function<ASTTraversal *()>;
static constexpr size_t NUM_OF_TRANSFORMATIONS = 3;
using EmittedTransformationsTy =
    llvm::SmallVector<TextModification *, NUM_OF_TRANSFORMATIONS>;

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

  static std::unordered_map<const char *, EmittedTransformationsTy> &
  getEmittedTransformations() {
    static std::unordered_map<const char *, EmittedTransformationsTy>
        EmittedTransformations;
    return EmittedTransformations;
  }

  static void registerRule(const char *ID, const std::string &Name,
                           ASTTraversalConstructor Factory) {
    getConstructorTable()[ID] = Factory;
    getIDTable()[Name] = ID;
    getNameTable()[ID] = Name;
    getEmittedTransformations()[ID] = EmittedTransformationsTy();
  }
};

/// Base class for all compatibility tool-related AST traversals.
class ASTTraversal : public ast_matchers::MatchFinder::MatchCallback {
public:
  /// Specify what nodes need to be matched by this ASTTraversal.
  virtual void registerMatcher(ast_matchers::MatchFinder &MF) = 0;

  /// Specify what needs to be done for each matched node.
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override = 0;

  virtual bool isMigrationRule() const { return false; }
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
  void emplaceMigrationRule(const char *ID) {
    assert(ASTTraversalMetaInfo::getConstructorTable().find(ID) !=
           ASTTraversalMetaInfo::getConstructorTable().end());
    Storage.emplace_back(std::unique_ptr<ASTTraversal>(
        ASTTraversalMetaInfo::getConstructorTable()[ID]()));
  }

  void emplaceAllRules(int SourceFileFlag);

  /// Run all emplaced ASTTraversal's over the given AST and populate \a TS.
  void matchAST(ASTContext &Context, TransformSetTy &TS, StmtStringMap &SSM);
};

/// Base class for migration rules.
///
/// The purpose of a MigrationRule is to populate TransformSet with
/// SourceTransformation's.
class MigrationRule : public ASTTraversal {
  friend class ASTTraversalManager;
  ASTTraversalManager *TM;

  TransformSetTy *TransformSet = nullptr;
  void setTransformSet(TransformSetTy &TS) { TransformSet = &TS; }

  static unsigned PairID;

protected:
  /// Add \a TM to the set of transformations.
  ///
  /// The ownership of the TM is transferred to the TransformSet.
  void emplaceTransformation(const char *RuleID, TextModification *TM);

  inline static unsigned incPairID() { return ++PairID; }

  const CompilerInstance &getCompilerInstance();

  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  void report(SourceLocation SL, IDTy MsgID, bool UseTextBegin, Ts &&... Vals) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    if (SL.isMacroID() && !SM.isMacroArgExpansion(SL)) {
      auto ItMatch = dpct::DpctGlobalInfo::getMacroTokenToMacroDefineLoc().find(
        SM.getCharacterData(SM.getImmediateSpellingLoc(SL)));
      if (ItMatch != dpct::DpctGlobalInfo::getMacroTokenToMacroDefineLoc().end()) {
        if (ItMatch->second->IsInRoot){
          SL = ItMatch->second->NameTokenLoc;
        }
      }
    }
    DiagnosticsUtils::report<IDTy, Ts...>(SL, MsgID, getCompilerInstance(),
                                          TransformSet, UseTextBegin,
                                          std::forward<Ts>(Vals)...);
  }

  // Extend version of report()
  // Pass Stmt to process macro more precisely.
  // The location should be consistent with the result of ReplaceStmt::getReplacement
  template <typename IDTy, typename... Ts>
  void report(const Stmt *S, IDTy MsgID, bool UseTextBegin, Ts &&... Vals) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    SourceLocation Begin(S->getBeginLoc());
    if (Begin.isMacroID() && !isOuterMostMacro(S)) {
      if (SM.isMacroArgExpansion(Begin)) {
        Begin = SM.getSpellingLoc(SM.getImmediateExpansionRange(Begin).getBegin());
      }
      else {
        Begin = SM.getSpellingLoc(Begin);
      }
    }
    else {
      Begin = SM.getExpansionLoc(Begin);
    }

    DiagnosticsUtils::report<IDTy, Ts...>(Begin, MsgID, getCompilerInstance(),
      TransformSet, UseTextBegin,
      std::forward<Ts>(Vals)...);
  }

  /// Dereference.
  /// returns "deviceProp" for exression `&deviceProp`
  std::string DereferenceArg(const clang::Expr *E, const ASTContext &Context) {
    if (auto arg = dyn_cast<clang::UnaryOperator>(E)) {
      if (arg->getOpcode() == UO_AddrOf) {
        return getStmtSpelling(arg->getSubExpr());
      }
    }
    /// TODO implement dereference for the general case, not only for foo(&a).
    /// TODO for now, report "can't compile".
    return "";
  }

  const std::string &getItemName() { return DpctGlobalInfo::getItemName(); }

  // Get node from match result map. And also check if the node's host file is
  // in the InRoot path and if the node has been processed by the same rule.
  template <typename NodeType>
  inline const NodeType *
  getNodeAsType(const ast_matchers::MatchFinder::MatchResult &Result,
                const char *Name, bool CheckNode = true) {
    return getNode<NodeType>(Result, Name, CheckNode, CheckNode);
  }
  template <typename NodeType>
  inline const NodeType *
  getAssistNodeAsType(const ast_matchers::MatchFinder::MatchResult &Result,
                      const char *Name, bool CheckInRoot = true) {
    return getNode<NodeType>(Result, Name, false, CheckInRoot);
  }

  const VarDecl *getVarDecl(const Expr *E) {
    if (!E)
      return nullptr;
    if (auto DeclRef = dyn_cast<DeclRefExpr>(E->IgnoreImpCasts()))
      return dyn_cast<VarDecl>(DeclRef->getDecl());
    return nullptr;
  }

private:
  template <typename NodeType>
  const NodeType *getNode(const ast_matchers::MatchFinder::MatchResult &Result,
                          const char *Name, bool CheckReplaced,
                          bool CheckInRoot) {
    if (auto Node = Result.Nodes.getNodeAs<NodeType>(Name))
      if (checkNode(Node->getSourceRange(), CheckReplaced, CheckInRoot))
        return Node;
    return nullptr;
  }
  bool checkNode(SourceRange &&SR, bool CheckReplaced, bool CheckInRoot) {
    if (CheckInRoot && !isInRoot(SR.getBegin()))
      return false;
    if (CheckReplaced && isReplaced(SR))
      return false;
    return true;
  }

  // Check if the node's host file is in the InRoot path.
  inline bool isInRoot(SourceLocation &&SL) {
    return DpctGlobalInfo::isInRoot(SL);
  }

  // Check if the location has been replaced by the same rule.
  bool isReplaced(SourceRange &SR) {
    for (auto RR : Replaced) {
      if (SR == RR)
        return true;
    }
    Replaced.push_back(SR);
    return false;
  }
  std::vector<SourceRange> Replaced;

public:
  bool isMigrationRule() const override { return true; }
  static bool classof(const ASTTraversal *T) { return T->isMigrationRule(); }

  virtual const std::string getName() const { return ""; }
  virtual const EmittedTransformationsTy getEmittedTransformations() const {
    return EmittedTransformationsTy();
  }

  void print(llvm::raw_ostream &OS);
  void printStatistics(llvm::raw_ostream &OS);

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

/// Migration rules with names
template <typename T> class NamedMigrationRule : public MigrationRule {
public:
  static const char ID;

  const std::string getName() const override final {
    return ASTTraversalMetaInfo::getNameTable()[&ID];
  }

  const EmittedTransformationsTy
  getEmittedTransformations() const override final {
    return ASTTraversalMetaInfo::getEmittedTransformations()[&ID];
  }

  void insertIncludeFile(SourceLocation SL, std::set<std::string> &HeaderFilter,
                         std::string &&InsertText);

protected:
  void emplaceTransformation(TextModification *TM) {
    if (TM) {
      TM->setParentRuleID(&ID);
      MigrationRule::emplaceTransformation(&ID, TM);
    }
  }

  void insertAroundStmt(const Stmt *S, std::string &&Prefix,
                        std::string &&Suffix, bool DoMacroExpansion = false) {
    auto P = incPairID();
    emplaceTransformation(
        new InsertBeforeStmt(S, std::move(Prefix), P, DoMacroExpansion));
    emplaceTransformation(
        new InsertAfterStmt(S, std::move(Suffix), P, DoMacroExpansion));
  }
  void insertAroundRange(const SourceLocation &PrefixSL,
                         const SourceLocation &SuffixSL, std::string &&Prefix,
                         std::string &&Suffix) {
    auto P = incPairID();
    emplaceTransformation(new InsertText(PrefixSL, std::move(Prefix), P));
    emplaceTransformation(new InsertText(SuffixSL, std::move(Suffix), P));
  }
};

template <typename T> const char NamedMigrationRule<T>::ID(0);

/// As follow define the migration rules which target to migration source
/// lanuage features to DPC++. The rules inherit from NamedMigrationRule
/// TODO: implement similar rules for each source language.
///

/// Migration rule for iteration space builtin variables (threadIdx, etc).
class IterationSpaceBuiltinRule
    : public NamedMigrationRule<IterationSpaceBuiltinRule> {
public:
  IterationSpaceBuiltinRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for class attributes.
/// This rule replace __align__ class attributes to __dpct_align__.
class AlignAttrsRule : public NamedMigrationRule<AlignAttrsRule> {
public:
  AlignAttrsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for function attributes.
/// This rule replace __forceinline__ class attributes to __dpct_inline__.
class FuncAttrsRule : public NamedMigrationRule<FuncAttrsRule> {
public:
  FuncAttrsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for atomic functions.
class AtomicFunctionRule : public NamedMigrationRule<AtomicFunctionRule> {
public:
  AtomicFunctionRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  static const std::unordered_map<std::string, std::string> AtomicFuncNamesMap;

private:
  void ReportUnsupportedAtomicFunc(const CallExpr *CE);
  void MigrateAtomicFunc(const CallExpr *CE,
                         const ast_matchers::MatchFinder::MatchResult &Result);
  void GetShareAttrRecursive(const Expr *Expr, bool &HasSharedAttr, bool &NeedReport);
  bool IsStmtInStatement(const clang::Stmt *S, const clang::Decl *Root);

};

/// Migration rule for thrust functions
class ThrustFunctionRule : public NamedMigrationRule<ThrustFunctionRule> {
public:
  ThrustFunctionRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
};

/// Migration rule for thrust constructor expressions
class ThrustCtorExprRule : public NamedMigrationRule<ThrustCtorExprRule> {
public:
  ThrustCtorExprRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
};

/// Migration rule for types replacements in var. declarations.
class TypeInDeclRule : public NamedMigrationRule<TypeInDeclRule> {
public:
  TypeInDeclRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void processCudaStreamType(const DeclaratorDecl *DD, const SourceManager *SM,
                             bool &SpecialCaseHappened);
  void reportForNcclAndCudnn(const TypeLoc *TL, const SourceLocation BeginLoc);
  bool replaceTemplateSpecialization(SourceManager *SM, LangOptions &LOpts,
                                     SourceLocation BeginLoc,
                                     const TemplateSpecializationTypeLoc TSL);
  bool replaceDependentNameTypeLoc(SourceManager *SM, LangOptions &LOpts,
                                   const TypeLoc *TL);
  bool isDeviceRandomStateType(const TypeLoc *TL, const SourceLocation &SL);
};

/// Migration rule for inserting namespace for vector types
class VectorTypeNamespaceRule
    : public NamedMigrationRule<VectorTypeNamespaceRule> {
public:
  VectorTypeNamespaceRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for vector type member access
class VectorTypeMemberAccessRule
    : public NamedMigrationRule<VectorTypeMemberAccessRule> {
public:
  VectorTypeMemberAccessRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

public:
  void renameMemberField(const MemberExpr *ME);
  static const std::map<std::string, std::string> MemberNamesMap;
};

/// Migration rule for vector type operator
class VectorTypeOperatorRule
    : public NamedMigrationRule<VectorTypeOperatorRule> {
public:
  VectorTypeOperatorRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void MigrateOverloadedOperatorDecl(
      const ast_matchers::MatchFinder::MatchResult &Result,
      const FunctionDecl *FD);
  void MigrateOverloadedOperatorCall(
      const ast_matchers::MatchFinder::MatchResult &Result,
      const CXXOperatorCallExpr *CE);

private:
  static const char NamespaceName[];
};

/// Migration rule for vector type constructor and make_<vector type>()
class VectorTypeCtorRule : public NamedMigrationRule<VectorTypeCtorRule> {
public:
  VectorTypeCtorRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  std::string getReplaceTypeName(const std::string &TypeName);
};

class ReplaceDim3CtorRule : public NamedMigrationRule<ReplaceDim3CtorRule> {
  ReplaceDim3Ctor *getReplaceDim3Modification(
      const ast_matchers::MatchFinder::MatchResult &Result);

public:
  ReplaceDim3CtorRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile, "Dim3MemberFieldsRule");
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for dim3 types member fields replacements.
class Dim3MemberFieldsRule : public NamedMigrationRule<Dim3MemberFieldsRule> {
public:
  Dim3MemberFieldsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void FieldsRename(const ast_matchers::MatchFinder::MatchResult &Result,
                    std::string Str, const MemberExpr *ME);
};

/// Migration rule for return types replacements.
class ReturnTypeRule : public NamedMigrationRule<ReturnTypeRule> {
public:
  ReturnTypeRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for removing of error hanlding if-stmt
class ErrorHandlingIfStmtRule
    : public NamedMigrationRule<ErrorHandlingIfStmtRule> {
public:
  ErrorHandlingIfStmtRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for adding try-catch for host APIs calls
class ErrorHandlingHostAPIRule
    : public NamedMigrationRule<ErrorHandlingHostAPIRule> {
public:
  ErrorHandlingHostAPIRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void insertTryCatch(const FunctionDecl *FD);
};

/// Migration rule for Device Property variables.
class DevicePropVarRule : public NamedMigrationRule<DevicePropVarRule> {
public:
  DevicePropVarRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

public:
  static const std::map<std::string, std::string> PropNamesMap;
};

/// Migration rule for enums constants.
class EnumConstantRule : public NamedMigrationRule<EnumConstantRule> {
public:
  EnumConstantRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void handleComputeMode(std::string EnumName, const DeclRefExpr *E);

  static std::map<std::string, std::string> EnumNamesMap;
};

/// Migration rule for Error enums constants.
class ErrorConstantsRule : public NamedMigrationRule<ErrorConstantsRule> {
public:
  ErrorConstantsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

class ManualMigrateEnumsRule
    : public NamedMigrationRule<ManualMigrateEnumsRule> {
public:
  ManualMigrateEnumsRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for BLAS enums.
class BLASEnumsRule : public NamedMigrationRule<BLASEnumsRule> {
public:
  BLASEnumsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for RANDOM enums.
class RandomEnumsRule : public NamedMigrationRule<RandomEnumsRule> {
public:
  RandomEnumsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for spBLAS enums.
class SPBLASEnumsRule : public NamedMigrationRule<SPBLASEnumsRule> {
public:
  SPBLASEnumsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for BLAS function calls.
class BLASFunctionCallRule : public NamedMigrationRule<BLASFunctionCallRule> {
public:
  BLASFunctionCallRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  bool isReplIndex(int i, const std::vector<int> &IndexInfo, int &IndexTemp);

  std::vector<std::string> getParamsAsStrs(const CallExpr *CE,
                                           const ASTContext &Context);
  const clang::VarDecl *getAncestralVarDecl(const clang::CallExpr *CE);

  struct BLASEnumInfo {
    std::vector<int> OperationIndexInfo;
    int FillModeIndexInfo;
    int SideModeIndexInfo;
    int DiagTypeIndexInfo;

    BLASEnumInfo() {}
    BLASEnumInfo(const std::vector<int> OperationIndexInfo,
                 const int FillModeIndexInfo, const int SideModeIndexInfo,
                 const int DiagTypeIndexInfo)
        : OperationIndexInfo(OperationIndexInfo),
          FillModeIndexInfo(FillModeIndexInfo),
          SideModeIndexInfo(SideModeIndexInfo),
          DiagTypeIndexInfo(DiagTypeIndexInfo) {}
  };

  std::string processParamIntCastToBLASEnum(
      const Expr *E, const CStyleCastExpr *CSCE, const int DistinctionID,
      const std::string IndentStr, const BLASEnumInfo &EnumInfo,
      std::string &PrefixInsertStr, std::string &CurrentArgumentRepl);
  bool isCEOrUETTEOrAnIdentifierOrLiteral(const Expr *E);
  std::string getExprString(const Expr *E,
                            bool AddparenthesisIfNecessary = false);

  std::vector<std::string> CallExprArguReplVec;
  std::string CallExprReplStr;

  std::string getFinalCallExprStr(std::string& FuncName) {
    std::string ResultStr;
    if (!CallExprArguReplVec.empty())
      ResultStr = CallExprArguReplVec[0];
    for (unsigned int i = 1; i < CallExprArguReplVec.size(); i++) {
      ResultStr = ResultStr + ", " + CallExprArguReplVec[i];
    }

    return FuncName + "(*" + ResultStr + ")";
  }

  void addWait(const std::string &FuncName, const CallExpr *CE,
               std::string &SuffixInsertStr, const std::string &IndentStr) {
    auto I = MapNames::SyncBLASFunc.find(FuncName);
    if (I != MapNames::SyncBLASFunc.end()) {
      ExprAnalysis EA(CE->getArg(I->second));
      SuffixInsertStr = getNL() + IndentStr + "if(" +
                        MapNames::getClNamespace() + "::get_pointer_type(" +
                        EA.getReplacedString() + ", " + CallExprArguReplVec[0] +
                        "->get_context())!=" + MapNames::getClNamespace() +
                        "::usm::alloc::device && " +
                        MapNames::getClNamespace() + "::get_pointer_type(" +
                        EA.getReplacedString() + ", " + CallExprArguReplVec[0] +
                        "->get_context())!=" + MapNames::getClNamespace() +
                        "::usm::alloc::shared) " + CallExprArguReplVec[0] +
                        "->wait();" + SuffixInsertStr;
    }
  }

  void
  applyMigrationText(bool NeedUseLambda, bool IsMacroArg, bool CanAvoidBrace,
                     bool CanAvoidUsingLambda, std::string OriginStmtType,
                     bool IsAssigned, SourceLocation OuterInsertLoc,
                     SourceLocation PrefixInsertLoc,
                     SourceLocation SuffixInsertLoc,
                     SourceLocation FuncNameBegin, SourceLocation FuncCallEnd,
                     unsigned int FuncCallLength, std::string IndentStr,
                     std::string PrefixInsertStr, std::string SuffixInsertStr,
                     bool IsHelperFunction = false, std::string FuncName = "") {
    if (NeedUseLambda) {
      if (CanAvoidUsingLambda && !IsMacroArg) {
        std::string InsertStr;
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none && !CanAvoidBrace)
          InsertStr = std::string("{") + getNL() + IndentStr + PrefixInsertStr +
                      CallExprReplStr + ";" + SuffixInsertStr + getNL() +
                      IndentStr + "}" + getNL() + IndentStr;
        else
          InsertStr = PrefixInsertStr + CallExprReplStr + ";" +
                      SuffixInsertStr + getNL() + IndentStr;
        emplaceTransformation(
            new InsertText(OuterInsertLoc, std::move(InsertStr)));
        report(OuterInsertLoc, Diagnostics::CODE_LOGIC_CHANGED, true,
               OriginStmtType == "if" ? "an " + OriginStmtType
                                      : "a " + OriginStmtType);
        if (IsHelperFunction)
          emplaceTransformation(new ReplaceText(FuncNameBegin, FuncCallLength,
                                                "0", true, FuncName));
        else
          emplaceTransformation(
              new ReplaceText(FuncNameBegin, FuncCallLength, "0"));
      } else {
        if (IsAssigned) {
          report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_LAMBDA, false);
          insertAroundRange(
              PrefixInsertLoc, SuffixInsertLoc,
              std::string("[&](){") + getNL() + IndentStr + PrefixInsertStr,
              std::string(";") + SuffixInsertStr + getNL() + IndentStr +
                  "return 0;" + getNL() + IndentStr + std::string("}()"));
        } else {
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            std::string("[&](){") + getNL() + IndentStr +
                                PrefixInsertStr,
                            std::string(";") + SuffixInsertStr + getNL() +
                                IndentStr + std::string("}()"));
        }
        if (IsHelperFunction)
          emplaceTransformation(new ReplaceText(FuncNameBegin, FuncCallLength,
                                                std::move(CallExprReplStr),
                                                true, FuncName));
        else
          emplaceTransformation(new ReplaceText(FuncNameBegin, FuncCallLength,
                                                std::move(CallExprReplStr)));
      }
    } else {
      if (!PrefixInsertStr.empty() || !SuffixInsertStr.empty()) {
        if (dpct::DpctGlobalInfo::getUsmLevel() == UsmLevel::none &&
            !CanAvoidBrace)
          insertAroundRange(
              PrefixInsertLoc, SuffixInsertLoc,
              std::string("{") + getNL() + IndentStr + PrefixInsertStr,
              SuffixInsertStr + getNL() + IndentStr + std::string("}"));
        else
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            std::move(PrefixInsertStr),
                            std::move(SuffixInsertStr));
      }
      if (IsHelperFunction)
        emplaceTransformation(new ReplaceText(FuncNameBegin, FuncCallLength,
                                              std::move(CallExprReplStr), true,
                                              FuncName));
      else
        emplaceTransformation(new ReplaceText(FuncNameBegin, FuncCallLength,
                                              std::move(CallExprReplStr)));
      if (IsAssigned) {
        insertAroundRange(FuncNameBegin, FuncCallEnd, "(", ", 0)");
        report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_COMMA_OP, true);
      }
    }
  }
};

/// Migration rule for Random function calls.
class RandomFunctionCallRule
    : public NamedMigrationRule<RandomFunctionCallRule> {
public:
  RandomFunctionCallRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for device Random function calls.
class DeviceRandomFunctionCallRule
    : public NamedMigrationRule<DeviceRandomFunctionCallRule> {
public:
  DeviceRandomFunctionCallRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for spBLAS function calls.
class SPBLASFunctionCallRule
    : public NamedMigrationRule<SPBLASFunctionCallRule> {
public:
  SPBLASFunctionCallRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for SOLVER enums.
class SOLVEREnumsRule : public NamedMigrationRule<SOLVEREnumsRule> {
public:
  SOLVEREnumsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for SOLVER function calls.
class SOLVERFunctionCallRule
    : public NamedMigrationRule<SOLVERFunctionCallRule> {
public:
  SOLVERFunctionCallRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  bool isReplIndex(int i, std::vector<int> &IndexInfo, int &IndexTemp);

  std::string getBufferNameAndDeclStr(const Expr *Arg, const ASTContext &AC,
                                      const std::string &TypeAsStr,
                                      SourceLocation SL,
                                      std::string &BufferDecl,
                                      int DistinctionID);
  void getParameterEnd(const SourceLocation &ParameterEnd,
                       SourceLocation &ParameterEndAfterComma,
                       const ast_matchers::MatchFinder::MatchResult &Result);
  const clang::VarDecl *getAncestralVarDecl(const clang::CallExpr *CE);
};

/// Migration rule for general function calls.
class FunctionCallRule : public NamedMigrationRule<FunctionCallRule> {
public:
  FunctionCallRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for event API calls
class EventAPICallRule : public NamedMigrationRule<EventAPICallRule> {
public:
  EventAPICallRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void handleEventRecord(const CallExpr *CE,
                         const ast_matchers::MatchFinder::MatchResult &Result,
                         bool IsAssigned);
  void
  handleEventElapsedTime(const CallExpr *CE,
                         const ast_matchers::MatchFinder::MatchResult &Result,
                         bool IsAssigned);
  void
  handleTimeMeasurement(const CallExpr *CE,
                        const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for stream API calls
class StreamAPICallRule : public NamedMigrationRule<StreamAPICallRule> {
public:
  StreamAPICallRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for kernel API calls
class KernelCallRule : public NamedMigrationRule<KernelCallRule> {
  std::unordered_set<unsigned> Insertions;

public:
  KernelCallRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile,
                    "SharedMemVarRule, ConstantMemVarRule, DeviceMemVarRule");
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void
  removeTrailingSemicolon(const CallExpr *KCall,
                          const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for device function calls
class DeviceFunctionCallRule
    : public NamedMigrationRule<DeviceFunctionCallRule> {
public:
  DeviceFunctionCallRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for __constant__/__shared__/__device__ memory variables.
class MemVarRule : public NamedMigrationRule<MemVarRule> {
public:
  MemVarRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void processDeref(const Stmt *S, ASTContext &Context);
};

/// Migration rule for memory management routine.
/// Current implementation is intentionally simplistic. The following things
/// need a more detailed design:
///   - interplay with error handling (possible solution is that we keep
///   function
///     signature as close to original as possible, so return error codes when
///     original functions return them);
///   - SYCL memory buffers are typed. Using a "char" type is definitely a
///   hack.
///     Using better type information requires some kind of global analysis and
///     heuristics, as well as a mechnism for user hint (like "treat all
///     buffers as float-typed")'
///   - interplay with streams need to be designed.
///   - transformation rules are currently unordered, which create potential
///     ambiguity, so need to understand how to handle function call arguments,
///     which are modified by other rules.
///
/// TODO:
///   - trigger include of runtime library.
class MemoryMigrationRule : public NamedMigrationRule<MemoryMigrationRule> {

public:
  MemoryMigrationRule();
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void mallocMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                       const CallExpr *C,
                       const UnresolvedLookupExpr *ULExpr = NULL,
                       bool IsAssigned = false);
  void memcpyMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                       const CallExpr *C,
                       const UnresolvedLookupExpr *ULExpr = NULL,
                       bool IsAssigned = false);
  void freeMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                     const CallExpr *C,
                     const UnresolvedLookupExpr *ULExpr = NULL,
                     bool IsAssigned = false);
  void memsetMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                       const CallExpr *C,
                       const UnresolvedLookupExpr *ULExpr = NULL,
                       bool IsAssigned = false);
  void arrayMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                       const CallExpr *C,
                       const UnresolvedLookupExpr *ULExpr = NULL,
                       bool IsAssigned = false);
  void getSymbolAddressMigration(
      const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
      const UnresolvedLookupExpr *ULExpr = NULL, bool IsAssigned = false);
  void getSymbolSizeMigration(
      const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
      const UnresolvedLookupExpr *ULExpr = NULL, bool IsAssigned = false);
  void prefetchMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                         const CallExpr *C,
                         const UnresolvedLookupExpr *ULExpr = NULL,
                         bool IsAssigned = false);
  void miscMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                     const CallExpr *C,
                     const UnresolvedLookupExpr *ULExpr = NULL,
                     bool IsAssigned = false);
  void cudaArrayGetInfo(const ast_matchers::MatchFinder::MatchResult &Result,
                        const CallExpr *C,
                        const UnresolvedLookupExpr *ULExpr = NULL,
                        bool IsAssigned = false);
  void cudaHostGetFlags(const ast_matchers::MatchFinder::MatchResult &Result,
                        const CallExpr *C,
                        const UnresolvedLookupExpr *ULExpr = NULL,
                        bool IsAssigned = false);
  void cudaMemAdvise(const ast_matchers::MatchFinder::MatchResult &Result,
                     const CallExpr *C,
                     const UnresolvedLookupExpr *ULExpr = NULL,
                     bool IsAssigned = false);
  void handleAsync(const CallExpr *C, unsigned i,
                          const ast_matchers::MatchFinder::MatchResult &Result);
  void handleDirection(const CallExpr *C, unsigned i);
  void replaceMemAPIArg(
      const Expr *E, const ast_matchers::MatchFinder::MatchResult &Result,
      std::string OffsetFromBaseStr = "");
  const ArraySubscriptExpr *getArraySubscriptExpr(const Expr *E);
  const Expr *getUnaryOperatorExpr(const Expr *E);
  void memcpySymbolMigration(
      const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
      const UnresolvedLookupExpr *ULExpr = NULL, bool IsAssigned = false);
  std::unordered_map<
      std::string,
      std::function<void(const ast_matchers::MatchFinder::MatchResult &Result,
                         const CallExpr *C, const UnresolvedLookupExpr *ULExpr,
                         bool IsAssigned)>>
      MigrationDispatcher;
  std::string getTypeStrRemovedAddrOf(const Expr *E, bool isCOCE = false);
  std::string getAssignedStr(const Expr *E, const std::string &Arg0Str);
  void mallocArrayMigration(const CallExpr *C, const std::string &Name,
                            size_t FlagIndex, SourceManager &SM);
  void mallocMigrationWithTransformation(SourceManager &SM, const CallExpr *C,
                                         const std::string &CallName,
                                         std::string &&ReplaceName,
                                         const std::string &PaddingArgs = "",
                                         bool NeedTypeCast = true,
                                         size_t AllocatedArgIndex = 0,
                                         size_t SizeArgIndel = 1);
  bool canUseTemplateStyleMigration(const Expr *AllocatedExpr,
                                    const Expr *SizeExpr, std::string &ReplType,
                                    std::string &ReplSize);
  std::string getTransformedMallocPrefixStr(const Expr *MallocOutArg,
                                            bool NeedTypeCast,
                                            bool TemplateStyle = false);
  void aggregatePitchedData(const CallExpr *C, size_t DataArgIndex,
                            size_t SizeArgIndex, SourceManager &SM,
                            bool ExcludeSizeArg = false);
  void aggregate3DVectorClassCtor(const CallExpr *C, StringRef ClassName,
                                  size_t ArgStartIndex, StringRef DefaultValue,
                                  SourceManager &SM, size_t ArgsNum = 2);
  void aggregateArgsToCtor(const CallExpr *C, const std::string &ClassName,
                           size_t StartArgIndex, size_t EndArgIndex,
                           const std::string &PaddingArgs, SourceManager &SM);
  void insertToPitchedData(const CallExpr *C, size_t ArgIndex) {
    if (C->getNumArgs() > ArgIndex)
      emplaceTransformation(
          new InsertAfterStmt(C->getArg(ArgIndex), "->to_pitched_data()"));
  }
  void insertZeroOffset(const CallExpr *C, size_t InsertArgIndex) {
    static std::string InsertedText =
        buildString(MapNames::getClNamespace(),
                    "::", DpctGlobalInfo::getCtadClass("id", 3), "(0, 0, 0), ");
    if (C->getNumArgs() > InsertArgIndex)
      emplaceTransformation(new InsertBeforeStmt(C->getArg(InsertArgIndex),
                                                 std::string(InsertedText)));
  }
};

class MemoryDataTypeRule : public NamedMigrationRule<MemoryDataTypeRule> {
  static inline std::string getCtadType(StringRef BaseTypeName) {
    return buildString(DpctGlobalInfo::getCtadClass(
        buildString(MapNames::getClNamespace(), "::", BaseTypeName), 3));
  }
  template <class... Args>
  void emplaceParamDecl(const VarDecl *VD, StringRef ParamType,
                               bool HasInitialZeroCtor, Args &&... ParamNames) {
    std::string ParamDecl;
    llvm::raw_string_ostream OS(ParamDecl);
    OS << ParamType << " ";
    unsigned Index = 0;
    std::initializer_list<int>{
        (printParamNameWithInitArgs(OS, VD->getName(), HasInitialZeroCtor,
                                    ParamNames, Index),
         0)...};
    OS << ";";
    emplaceTransformation(
        ReplaceVarDecl::getVarDeclReplacement(VD, std::move(OS.str())));
  }
  static inline llvm::raw_ostream &printParamName(llvm::raw_ostream &OS,
                                                  StringRef BaseName,
                                                  StringRef ParamName) {
    return OS << BaseName << "_" << ParamName << getCTFixedSuffix();
  }
  static inline llvm::raw_ostream &
  printParamNameWithInitArgs(llvm::raw_ostream &OS, StringRef BaseName,
                             bool HasInitialZeroCtor, StringRef ParamName,
                             unsigned &Index) {
    if (Index++)
      OS << ", ";
    printParamName(OS, BaseName, ParamName);
    if (HasInitialZeroCtor)
      OS << "(0, 0, 0)";
    return OS;
  }

  const static MapNames::MapTy Parms3DMemberNames;
  const static MapNames::MapTy ExtentMemberNames;
  const static MapNames::MapTy PitchMemberNames;

public:
  void emplaceMemcpy3DDeclarations(const VarDecl *VD);
  static std::string getMemcpy3DArguments(StringRef BaseName);

  static std::string getMemcpy3DMemberName(StringRef BaseName,
                                           const std::string &Member) {
    auto Itr = Parms3DMemberNames.find(Member);
    if (Itr != Parms3DMemberNames.end()) {
      std::string ReplacedName;
      llvm::raw_string_ostream OS(ReplacedName);
      printParamName(OS, BaseName, Itr->second);
      return OS.str();
    }
    return Member;
  }
  MemoryDataTypeRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Name all unnamed types.
class UnnamedTypesRule : public NamedMigrationRule<UnnamedTypesRule> {
public:
  UnnamedTypesRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Guess original code indent width.
class GuessIndentWidthRule : public NamedMigrationRule<GuessIndentWidthRule> {
public:
  GuessIndentWidthRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration for math functions
class MathFunctionsRule : public NamedMigrationRule<MathFunctionsRule> {
public:
  MathFunctionsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void handleExceptionalFunctions(
      const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result);
  void
  handleHalfFunctions(const CallExpr *CE,
                      const ast_matchers::MatchFinder::MatchResult &Result);
  void handleSingleDoubleFunctions(
      const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result);
  void
  handleTypecastFunctions(const CallExpr *CE,
                          const ast_matchers::MatchFinder::MatchResult &Result);
  void
  handleMiscFunctions(const CallExpr *CE,
                      const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration for warp functions
class WarpFunctionsRule : public NamedMigrationRule<WarpFunctionsRule> {
public:
  WarpFunctionsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migration rule for replacing __syncthreads() function call.
///
/// This rule replace __syncthreads() with item.barrier()
class SyncThreadsRule : public NamedMigrationRule<SyncThreadsRule> {
public:
  SyncThreadsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Migrate Function Attributes to Sycl kernel info, defined in
/// runtime headers.
// TODO: only maxThreadsPerBlock is supported.
class KernelFunctionInfoRule
    : public NamedMigrationRule<KernelFunctionInfoRule> {
public:
  KernelFunctionInfoRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  static const std::map<std::string, std::string> AttributesNamesMap;
};

/// Migration rule for type cast issue
class TypeCastRule : public NamedMigrationRule<TypeCastRule> {
public:
  TypeCastRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// RecognizeAPINameRule to give comments for the api not in the record table
class RecognizeAPINameRule : public NamedMigrationRule<RecognizeAPINameRule> {
public:
  RecognizeAPINameRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  const std::string GetFunctionSignature(const FunctionDecl *Func);
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

/// Texture migration rule
class TextureRule : public NamedMigrationRule<TextureRule> {
  // Get the binary operator if E is lhs of an assign experssion.
  const BinaryOperator *getAssignedBO(const Expr *E, ASTContext &Context);
  const BinaryOperator *getParentAsAssignedBO(const Expr *E,
                                              ASTContext &Context);
  void replaceResourceDataExpr(const MemberExpr *ME, const ASTContext &Context);
  inline const MemberExpr *getParentMemberExpr(const Stmt *S) {
    return DpctGlobalInfo::findAncestor<MemberExpr>(S);
  }
  static MapNames::MapTy LinearResourceTypeNames;
  static MapNames::MapTy Pitched2DResourceTypeNames;

public:
  TextureRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  static const MapNames::MapTy TextureMemberNames;
};

template <typename T> class RuleRegister {
public:
  RuleRegister(const char *ID, const std::string &Name) {
    ASTTraversalMetaInfo::registerRule(ID, Name, [] { return new T; });
  }
};

/// CXXNewExprRule is to migrate types in C++ new expressions, e.g.
/// "new cudaStream_t[10]" => "new queue_p[10]"
/// "new cudaStream_t" => "new queue_p"
class CXXNewExprRule : public NamedMigrationRule<CXXNewExprRule> {
public:
  CXXNewExprRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

class NamespaceRule : public NamedMigrationRule<NamespaceRule> {
public:
  NamespaceRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

class RemoveBaseClassRule : public NamedMigrationRule<RemoveBaseClassRule> {
public:
  RemoveBaseClassRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

class ThrustVarRule : public NamedMigrationRule<ThrustVarRule> {
public:
  ThrustVarRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
};


#define REGISTER_RULE(TYPE_NAME)                                               \
  RuleRegister<TYPE_NAME> g_##TYPE_NAME(&TYPE_NAME::ID, #TYPE_NAME);

} // namespace dpct
} // namespace clang
#endif // DPCT_AST_TRAVERSAL_H
