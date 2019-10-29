//===--- ASTTraversal.h ---------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
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
  SourceManager &SM;

  std::unordered_set<std::string> SeenFiles;
  bool DpstdHeaderInserted;
  ASTTraversalManager &ATM;

public:
  IncludesCallbacks(TransformSetTy &TransformSet, SourceManager &SM,
                    ASTTraversalManager &ATM)
      : TransformSet(TransformSet), SM(SM), DpstdHeaderInserted(false),
        ATM(ATM) {}
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

    // Concatenate source file and line number (eg: x.cpp:4) and parameter.
    std::string SourceAndLine =
        buildString(SM.getBufferName(SL), ":", SM.getPresumedLineNumber(SL),
                    ":", std::forward<Ts>(Vals)...);

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
  std::string DereferenceArg(const clang::Expr *E, const ASTContext &Context) {
    if (auto arg = dyn_cast<clang::UnaryOperator>(E)) {
      if (arg->getOpcode() == UO_AddrOf) {
        return getStmtSpelling(arg->getSubExpr(), Context);
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
    emplaceTransformation(new InsertBeforeStmt(S, std::move(Prefix), P, DoMacroExpansion));
    emplaceTransformation(new InsertAfterStmt(S, std::move(Suffix), P, DoMacroExpansion));
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
  void
  MigrateAtomicFunc(const CallExpr *CE,
                      const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for thrust functions
class ThrustFunctionRule : public NamedMigrationRule<ThrustFunctionRule> {
public:
  ThrustFunctionRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
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
  // A duplicate filter that filters out redundant replacements when
  // serveral variable are declared in one statement
  std::unordered_set<unsigned> DupFilter;
};

/// Migration rule for types replacements in template var. declarations
class TemplateTypeInDeclRule
    : public NamedMigrationRule<TemplateTypeInDeclRule> {
public:
  TemplateTypeInDeclRule() {
    SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  // A duplicate filter that filters out redundant replacements when
  // serveral variable are declared in one statement
  std::unordered_set<unsigned> DupFilter;
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
  bool isNamespaceInserted(SourceLocation SL);

private:
  void replaceTypeName(const QualType &QT, SourceLocation BeginLoc,
                       bool isDeclType = false);
  std::unordered_set<unsigned int> DupFilter;
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

  static const std::map<std::string, std::string> EnumNamesMap;
};

/// Migration rule for Error enums constants.
class ErrorConstantsRule : public NamedMigrationRule<ErrorConstantsRule> {
public:
  ErrorConstantsRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
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

/// Migration rule for BLAS function calls.
class BLASFunctionCallRule : public NamedMigrationRule<BLASFunctionCallRule> {
public:
  BLASFunctionCallRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  std::string getBufferNameAndDeclStr(const Expr *Arg, const ASTContext &AC,
                                      const std::string &TypeAsStr,
                                      SourceLocation SL,
                                      std::string &BufferDecl,
                                      int DistinctionID);
  std::string
  getBufferNameAndDeclStr(const std::string &PointerName, const ASTContext &AC,
                          const std::string &TypeAsStr, SourceLocation SL,
                          std::string &BufferDecl, int DistinctionID);

  bool isReplIndex(int i, const std::vector<int> &IndexInfo, int &IndexTemp);

  std::vector<std::string> getParamsAsStrs(const CallExpr *CE,
                                           const ASTContext &Context);
  const clang::VarDecl *getAncestralVarDecl(const clang::CallExpr *CE);
  void processParamIntCastToBLASEnum(const Expr *E, const CStyleCastExpr *CSCE,
                                     const ASTContext &Context,
                                     const int DistinctionID,
                                     const std::string IndentStr,
                                     const std::vector<int> &OperationIndexInfo,
                                     const int FillModeIndexInfo,
                                     std::string &PrefixInsertStr);
  void processTrmmCall(const CallExpr *CE, std::string &PrefixInsertStr,
                       const std::string IndentStr);
  void processTrmmParams(const CallExpr *CE, std::string &PrefixInsertStr,
                         std::string &BufferName, std::string &BufferDecl,
                         int &IndexTemp, int DistinctionID,
                         const std::string IndentStr,
                         const std::vector<std::string> &BufferTypeInfo,
                         const SourceLocation &StmtBegin);
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
  removeTrailingSemicolon(const CUDAKernelCallExpr *KCall,
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
  void insertExplicitCast(const ImplicitCastExpr *Impl, const QualType &Type);
  template <class T>
  void replaceMemberOperator(const ast_type_traits::DynTypedNode &Node) {
    if (auto M = Node.get<T>())
      emplaceTransformation(new ReplaceToken(M->getOperatorLoc(), "->"));
  }
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
  void getSymbolAddressMigration(
      const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
      const UnresolvedLookupExpr *ULExpr = NULL, bool IsAssigned = false);
  void miscMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                       const CallExpr *C,
                       const UnresolvedLookupExpr *ULExpr = NULL,
                       bool IsAssigned = false);
  void handleAsync(const CallExpr *C, unsigned i,
                   const ast_matchers::MatchFinder::MatchResult &Result);
  void replaceMemAPIArg(const Expr *E,
                        const ast_matchers::MatchFinder::MatchResult &Result,
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
};

/// Name all unnamed types.
class UnnamedTypesRule : public NamedMigrationRule<UnnamedTypesRule> {
public:
  UnnamedTypesRule() { SetRuleProperty(ApplyToCudaFile | ApplyToCppFile); }
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

#define REGISTER_RULE(TYPE_NAME)                                               \
  RuleRegister<TYPE_NAME> g_##TYPE_NAME(&TYPE_NAME::ID, #TYPE_NAME);

} // namespace dpct
} // namespace clang

#endif // DPCT_AST_TRAVERSAL_H
