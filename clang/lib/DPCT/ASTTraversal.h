//===--------------- ASTTraversal.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_AST_TRAVERSAL_H
#define DPCT_AST_TRAVERSAL_H

#include "AnalysisInfo.h"
#include "CrashRecovery.h"
#include "Diagnostics.h"
#include "FFTAPIMigration.h"
#include "InclusionHeaders.h"
#include "MapNames.h"
#include "TextModification.h"
#include "Utility.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Frontend/CompilerInstance.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace clang {
namespace dpct {

enum class PassKind : unsigned { PK_Analysis = 0, PK_Migration, PK_End };

/// Migration rules at the pre-processing stages, e.g. macro rewriting and
/// including directives rewriting.
class IncludesCallbacks : public PPCallbacks {
  TransformSetTy &TransformSet;
  IncludeMapSetTy &IncludeMapSet;
  SourceManager &SM;
  RuleGroups &Groups;

  std::unordered_set<std::string> SeenFiles;
  bool IsFileInCmd = true;

public:
  IncludesCallbacks(TransformSetTy &TransformSet,
                    IncludeMapSetTy &IncludeMapSet, SourceManager &SM,
                    RuleGroups &G)
      : TransformSet(TransformSet), IncludeMapSet(IncludeMapSet), SM(SM),
        Groups(G) {}
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
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
  bool ReplaceCuMacro(const Token &MacroNameTok);
  void ReplaceCuMacro(SourceRange ConditionRange, IfType IT,
                      SourceLocation IfLoc, SourceLocation ElifLoc);
  void Defined(const Token &MacroNameTok, const MacroDefinition &MD,
               SourceRange Range) override;
  void Endif(SourceLocation Loc, SourceLocation IfLoc) override;
  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID = FileID()) override;
  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind ConditionValue) override;
  void Else(SourceLocation Loc, SourceLocation IfLoc) override;
  void Elif(SourceLocation Loc, SourceRange ConditionRange,
            ConditionValueKind ConditionValue, SourceLocation IfLoc) override;
  bool ShouldEnter(StringRef FileName, bool IsAngled) override;
  bool isInAnalysisScope(SourceLocation Loc);
  // Find the "#" before a preprocessing directive, return -1 if have some false
  int findPoundSign(SourceLocation DirectiveStart);
  void insertCudaArchRepl(std::shared_ptr<clang::dpct::ExtReplacement> Repl);

private:
  /// e.g. "__launch_bounds(32, 32)  void foo()"
  /// Result is "void foo()"
  std::shared_ptr<TextModification>
  removeMacroInvocationAndTrailingSpaces(SourceRange Range);
};

/// Base class for all tool-related AST traversals.
class ASTTraversal : public ast_matchers::MatchFinder::MatchCallback {
public:
  /// Specify what nodes need to be matched by this ASTTraversal.
  virtual void registerMatcher(ast_matchers::MatchFinder &MF) = 0;

  /// Specify what needs to be done for each matched node.
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override = 0;

  virtual bool isMigrationRule() const { return false; }
};

/// Base class for migration rules.
///
/// The purpose of a MigrationRule is to populate TransformSet with
/// SourceTransformation's.
class MigrationRule : public ASTTraversal {
  friend class MigrationRuleManager;

  void setTransformSet(TransformSetTy &TS) { TransformSet = &TS; }
  void setName(StringRef N) { Name = N; }

  static unsigned PairID;

protected:
  TransformSetTy *TransformSet = nullptr;
  /// Add \a TM to the set of transformations.
  ///
  /// The ownership of the TM is transferred to the TransformSet.
  void emplaceTransformation(TextModification *TM);

  inline static unsigned incPairID() { return ++PairID; }

  const CompilerInstance &getCompilerInstance();

  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  bool report(SourceLocation SL, IDTy MsgID, bool UseTextBegin, Ts &&...Vals) {
    return DiagnosticsUtils::report<IDTy, Ts...>(
        SL, MsgID, TransformSet, UseTextBegin,
        std::forward<Ts>(Vals)...);
  }
  // Extend version of report()
  // Pass Stmt to process macro more precisely.
  // The location should be consistent with the result of
  // ReplaceStmt::getReplacement
  template <typename IDTy, typename... Ts>
  void report(const Stmt *S, IDTy MsgID, bool UseTextBegin, Ts &&...Vals) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    SourceLocation Begin(S->getBeginLoc());
    if (Begin.isMacroID() && !isOuterMostMacro(S)) {
      if (SM.isMacroArgExpansion(Begin)) {
        Begin =
            SM.getSpellingLoc(SM.getImmediateExpansionRange(Begin).getBegin());
      } else {
        Begin = SM.getSpellingLoc(Begin);
      }
    } else {
      Begin = SM.getExpansionLoc(Begin);
    }

    DiagnosticsUtils::report<IDTy, Ts...>(Begin, MsgID, TransformSet,
                                          UseTextBegin,
                                          std::forward<Ts>(Vals)...);
  }

  // Get node from match result map. And also check if the node's host file is
  // in the InRoot path and if the node has been processed by the same rule.
  template <typename NodeType>
  inline const NodeType *
  getNodeAsType(const ast_matchers::MatchFinder::MatchResult &Result,
                const char *Name) {
    if (auto Node = Result.Nodes.getNodeAs<NodeType>(Name))
      if (!isReplaced(Node->getSourceRange()))
        return Node;
    return nullptr;
  }
  template <typename NodeType>
  inline const NodeType *
  getAssistNodeAsType(const ast_matchers::MatchFinder::MatchResult &Result,
                      const char *Name) {
    return Result.Nodes.getNodeAs<NodeType>(Name);
  }

  const VarDecl *getVarDecl(const Expr *E) {
    if (!E)
      return nullptr;
    if (auto DeclRef = dyn_cast<DeclRefExpr>(E->IgnoreImpCasts()))
      return dyn_cast<VarDecl>(DeclRef->getDecl());
    return nullptr;
  }

private:
  // Check if the location has been replaced by the same rule.
  bool isReplaced(SourceRange SR) {
    for (const auto &RR : Replaced) {
      if (SR == RR)
        return true;
    }
    Replaced.push_back(SR);
    return false;
  }

  std::vector<SourceRange> Replaced;
  TransformSetTy Transformations;
  StringRef Name;

public:
  bool isMigrationRule() const override { return true; }
  static bool classof(const ASTTraversal *T) { return T->isMigrationRule(); }

  StringRef getName() const { return Name; }
  const TransformSetTy &getEmittedTransformations() const {
    return Transformations;
  }

  void print(llvm::raw_ostream &OS);
  void printStatistics(llvm::raw_ostream &OS);
};

/// Migration rules with names
template <typename T> class NamedMigrationRule : public MigrationRule {
public:
  static const char ID;

  void insertIncludeFile(SourceLocation SL, std::set<std::string> &HeaderFilter,
                         std::string &&InsertText);

  void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    runWithCrashGuard([=]() { static_cast<T *>(this)->runRule(Result); },
                      "Error: dpct internal error. Migration rule causing the "
                      "error skipped. Migration continues.\n");
    return;
  }

protected:
  void emplaceTransformation(TextModification *TM) {
    if (TM) {
      TM->setParentRuleName(getName());
      MigrationRule::emplaceTransformation(TM);
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
                         std::string &&Suffix,
                         bool BlockLevelFormatFlag = false) {
    auto P = incPairID();
    auto PIT = new InsertText(PrefixSL, std::move(Prefix), P);
    auto SIT = new InsertText(SuffixSL, std::move(Suffix), P);
    if (BlockLevelFormatFlag) {
      PIT->setBlockLevelFormatFlag();
      SIT->setBlockLevelFormatFlag();
    }
    emplaceTransformation(std::move(PIT));
    emplaceTransformation(std::move(SIT));
  }

  void addReplacementForLibraryAPI(LibraryMigrationFlags Flags,
                                   LibraryMigrationStrings &Strings,
                                   LibraryMigrationLocations Locations,
                                   std::string FuncName, const CallExpr *CE) {
    if (Flags.NeedUseLambda) {
      if (Strings.PrefixInsertStr.empty() && Strings.SuffixInsertStr.empty()) {
        // If there is one API call in the migrated code, it is unnecessary to
        // use a lambda expression
        Flags.NeedUseLambda = false;
      }
    }

    if (Flags.NeedUseLambda) {
      if ((Flags.MoveOutOfMacro && Flags.IsMacroArg) ||
          (Flags.CanAvoidUsingLambda && !Flags.IsMacroArg)) {
        std::string InsertString;
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
            !Flags.CanAvoidBrace)
          InsertString = std::string("{") + getNL() + Strings.IndentStr +
                         Strings.PrefixInsertStr + Strings.Repl + ";" +
                         Strings.SuffixInsertStr + getNL() + Strings.IndentStr +
                         "}" + getNL() + Strings.IndentStr;
        else
          InsertString = Strings.PrefixInsertStr + Strings.Repl + ";" +
                         Strings.SuffixInsertStr + getNL() + Strings.IndentStr;

        if (Flags.MoveOutOfMacro && Flags.IsMacroArg) {
          auto IT = new InsertText(Locations.OutOfMacroInsertLoc,
                                   std::move(InsertString));
          IT->setBlockLevelFormatFlag();
          emplaceTransformation(std::move(IT));
          report(Locations.OutOfMacroInsertLoc, Diagnostics::CODE_LOGIC_CHANGED,
                 true, "function-like macro");
        } else {
          auto IT =
              new InsertText(Locations.OuterInsertLoc, std::move(InsertString));
          IT->setBlockLevelFormatFlag();
          emplaceTransformation(std::move(IT));
          report(Locations.OuterInsertLoc, Diagnostics::CODE_LOGIC_CHANGED,
                 true,
                 Flags.OriginStmtType == "if" ? "an " + Flags.OriginStmtType
                                              : "a " + Flags.OriginStmtType);
        }
        emplaceTransformation(
            new ReplaceText(Locations.PrefixInsertLoc, Locations.Len, "0"));
      } else {
        if (Flags.IsAssigned) {
          report(Locations.PrefixInsertLoc, Diagnostics::NOERROR_RETURN_LAMBDA,
                 false);
          insertAroundRange(
              Locations.PrefixInsertLoc, Locations.SuffixInsertLoc,
              std::string("[&](){") + getNL() + Strings.IndentStr +
                  Strings.PrefixInsertStr,
              std::string(";") + Strings.SuffixInsertStr + getNL() +
                  Strings.IndentStr + "return 0;" + getNL() +
                  Strings.IndentStr + std::string("}()"),
              true);
        } else {
          insertAroundRange(
              Locations.PrefixInsertLoc, Locations.SuffixInsertLoc,
              std::string("[&](){") + getNL() + Strings.IndentStr +
                  Strings.PrefixInsertStr,
              std::string(";") + Strings.SuffixInsertStr + getNL() +
                  Strings.IndentStr + std::string("}()"),
              true);
        }
        emplaceTransformation(new ReplaceText(
            Locations.PrefixInsertLoc, Locations.Len, std::move(Strings.Repl)));
      }
    } else {
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
          !Flags.CanAvoidBrace) {
        if (!Strings.PrefixInsertStr.empty() ||
            !Strings.SuffixInsertStr.empty()) {
          insertAroundRange(
              Locations.PrefixInsertLoc, Locations.SuffixInsertLoc,
              Strings.PrePrefixInsertStr + std::string("{") + getNL() +
                  Strings.IndentStr + Strings.PrefixInsertStr,
              Strings.SuffixInsertStr + getNL() + Strings.IndentStr +
                  std::string("}"),
              true);
        }
      } else {
        insertAroundRange(Locations.PrefixInsertLoc, Locations.SuffixInsertLoc,
                          Strings.PrePrefixInsertStr + Strings.PrefixInsertStr,
                          std::move(Strings.SuffixInsertStr), true);
      }
      if (Flags.IsAssigned) {
        insertAroundRange(Locations.FuncNameBegin, Locations.FuncCallEnd,
                          "DPCT_CHECK_ERROR(", ")");
        requestFeature(HelperFeatureEnum::device_ext);
      }

      emplaceTransformation(new ReplaceStmt(CE, true, Strings.Repl));
    }
  }

  std::string makeDevicePolicy(const Stmt *S) {
    auto UniqueName = [](const Stmt *S) {
      auto &SM = DpctGlobalInfo::getSourceManager();
      SourceLocation Loc = S->getBeginLoc();
      return getHashAsString(Loc.printToString(SM)).substr(0, 6);
    };
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, S, HelperFuncType::HFT_DefaultQueue);
    std::string TemplateArg = "";
    if (DpctGlobalInfo::isSyclNamedLambda())
      TemplateArg = std::string("<class Policy_") + UniqueName(S) + ">";
    std::string Policy = "oneapi::dpl::execution::make_device_policy" +
                         TemplateArg + "({{NEEDREPLACEQ" +
                         std::to_string(Index) + "}})";
    return Policy;
  }
};

template <typename T> const char NamedMigrationRule<T>::ID(0);

/// As follow define the migration rules which target to migration source
/// language features to SYCL. The rules inherit from NamedMigrationRule
/// TODO: implement similar rules for each source language.
///

/// Migration rule for iteration space built-in variables (threadIdx, etc).
class IterationSpaceBuiltinRule
    : public NamedMigrationRule<IterationSpaceBuiltinRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  bool renameBuiltinName(const DeclRefExpr *DRE, std::string &NewName);
};

/// Migration rule for atomic functions.
class AtomicFunctionRule : public NamedMigrationRule<AtomicFunctionRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  void ReportUnsupportedAtomicFunc(const CallExpr *CE);
  void MigrateAtomicFunc(const CallExpr *CE,
                         const ast_matchers::MatchFinder::MatchResult &Result);
};

class ZeroLengthArrayRule
    : public NamedMigrationRule<ZeroLengthArrayRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class MiscAPIRule : public NamedMigrationRule<MiscAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for types replacements in var. declarations.
class TypeInDeclRule : public NamedMigrationRule<TypeInDeclRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  struct TypeLocHash {
    std::size_t operator()(TypeLoc const &TL) const noexcept {
      return std::hash<unsigned>{}(TL.getBeginLoc().getRawEncoding());
    }
  };
  struct TypeLocEqual {
    bool operator()(TypeLoc const &TL1, TypeLoc const &TL2) const {
      return (TL1.getBeginLoc() == TL2.getBeginLoc()) &&
             (TL1.getEndLoc() == TL2.getEndLoc());
    }
  };
  // Holds the set of TypeLocs that have been processed.
  // Used to prevent them from being processed multiple times
  std::unordered_set<TypeLoc, TypeLocHash, TypeLocEqual> ProcessedTypeLocs;

  void processConstFFTHandleType(const DeclaratorDecl *DD,
                                 SourceLocation BeginLoc,
                                 SourceLocation EndLoc,
                                 bool HasGlobalNSPrefix);
  void processCudaStreamType(const DeclaratorDecl *DD);
  bool replaceTemplateSpecialization(SourceManager *SM, LangOptions &LOpts,
                                     SourceLocation BeginLoc,
                                     const TemplateSpecializationTypeLoc TSL);
  bool replaceDependentNameTypeLoc(SourceManager *SM, LangOptions &LOpts,
                                   const TypeLoc *TL);
  bool replaceTransformIterator(SourceManager *SM, LangOptions &LOpts,
                                const TypeLoc *TL);
};

class TemplateSpecializationTypeLocRule
    : public clang::dpct::NamedMigrationRule<
          TemplateSpecializationTypeLocRule> {

public:
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};

class UserDefinedAPIRule
    : public clang::dpct::NamedMigrationRule<UserDefinedAPIRule> {
  std::string APIName;
  bool HasExplicitTemplateArgs;

public:
  UserDefinedAPIRule(std::string APIName, bool HasExplicitTemplateArguments)
      : APIName(std::move(APIName)),
        HasExplicitTemplateArgs(HasExplicitTemplateArguments){};
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};

class UserDefinedTypeRule
    : public clang::dpct::NamedMigrationRule<UserDefinedTypeRule> {
  std::string TypeName;

public:
  UserDefinedTypeRule(std::string TypeName) : TypeName(TypeName){};
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};

class UserDefinedClassFieldRule
    : public clang::dpct::NamedMigrationRule<UserDefinedClassFieldRule> {
  std::string BaseName;
  std::string FieldName;

public:
  UserDefinedClassFieldRule(std::string BaseName, std::string FieldName)
      : BaseName(BaseName), FieldName(FieldName){};
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};

class UserDefinedClassMethodRule
    : public clang::dpct::NamedMigrationRule<UserDefinedClassMethodRule> {
  std::string BaseName;
  std::string MethodName;

public:
  UserDefinedClassMethodRule(std::string BaseName, std::string MethodName)
      : BaseName(BaseName), MethodName(MethodName){};
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};

class UserDefinedEnumRule
    : public clang::dpct::NamedMigrationRule<UserDefinedEnumRule> {
  std::string EnumName;

public:
  UserDefinedEnumRule(std::string EnumName) : EnumName(EnumName){};
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};


/// Migration rule for inserting namespace for vector types
class VectorTypeNamespaceRule
    : public NamedMigrationRule<VectorTypeNamespaceRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for vector type member access
class VectorTypeMemberAccessRule
    : public NamedMigrationRule<VectorTypeMemberAccessRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

public:
  void renameMemberField(const MemberExpr *ME);
  static const std::map<std::string, std::string> MemberNamesMap;
};

/// Migration rule for vector type operator
class VectorTypeOperatorRule
    : public NamedMigrationRule<VectorTypeOperatorRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

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

class ReplaceDim3CtorRule : public NamedMigrationRule<ReplaceDim3CtorRule> {
  ReplaceDim3Ctor *getReplaceDim3Modification(
      const ast_matchers::MatchFinder::MatchResult &Result);

public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for dim3 types member fields replacements.
class Dim3MemberFieldsRule : public NamedMigrationRule<Dim3MemberFieldsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CudaExtentRule : public NamedMigrationRule<CudaExtentRule> {
  CharSourceRange getConstructorRange(const CXXConstructExpr *Ctor);
  void replaceConstructor(const CXXConstructExpr *Ctor);
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }
};

class CudaUuidRule : public NamedMigrationRule<CudaUuidRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for return types replacements.
class ReturnTypeRule : public NamedMigrationRule<ReturnTypeRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for removing of error handling if-stmt
class ErrorHandlingIfStmtRule
    : public NamedMigrationRule<ErrorHandlingIfStmtRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for adding try-catch for host APIs calls
class ErrorHandlingHostAPIRule
    : public NamedMigrationRule<ErrorHandlingHostAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  void insertTryCatch(const FunctionDecl *FD);
};

/// Migration rule for CUDA device property and attribute.
/// E.g. cudaDeviceProp, cudaPointerAttributes.
class DeviceInfoVarRule : public NamedMigrationRule<DeviceInfoVarRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

public:
  static const std::map<std::string, std::string> PropNamesMap;
};

/// Migration rule for enums constants.
class EnumConstantRule : public NamedMigrationRule<EnumConstantRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  void handleComputeMode(std::string EnumName, const DeclRefExpr *E);

  static std::unordered_map<std::string, std::shared_ptr<EnumNameRule>>
      EnumNamesMap;
};

/// Migration rule for Error enums constants.
class ErrorConstantsRule : public NamedMigrationRule<ErrorConstantsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class LinkageSpecDeclRule : public NamedMigrationRule<LinkageSpecDeclRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class ManualMigrateEnumsRule
    : public NamedMigrationRule<ManualMigrateEnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for FFT enums.
class FFTEnumsRule : public NamedMigrationRule<FFTEnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for CU_JIT enums.
class CU_JITEnumsRule : public NamedMigrationRule<CU_JITEnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for BLAS enums.
class BLASEnumsRule : public NamedMigrationRule<BLASEnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for RANDOM enums.
class RandomEnumsRule : public NamedMigrationRule<RandomEnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for spBLAS enums.
class SPBLASEnumsRule : public NamedMigrationRule<SPBLASEnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for BLAS function calls.
class BLASFunctionCallRule : public NamedMigrationRule<BLASFunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

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
  bool NeedWaitAPICall = false;

  std::string getFinalCallExprStr(std::string &FuncName) {
    std::string ResultStr;
    if (!CallExprArguReplVec.empty())
      ResultStr = CallExprArguReplVec[0];
    for (unsigned int i = 1; i < CallExprArguReplVec.size(); i++) {
      ResultStr = ResultStr + ", " + CallExprArguReplVec[i];
    }

    return FuncName + "(*" + ResultStr + ")" +
           (NeedWaitAPICall ? ".wait()" : "");
  }

  void addWait(const std::string &FuncName, const CallExpr *CE,
               std::string &PrefixInsertStr, std::string &SuffixInsertStr,
               const std::string &IndentStr) {
    auto getIfStmtStr = [=](const std::string Ptr) -> std::string {
      return "if(" + MapNames::getClNamespace() + "get_pointer_type(" + Ptr +
             ", " + CallExprArguReplVec[0] +
             "->get_context())!=" + MapNames::getClNamespace() +
             "usm::alloc::device && " + MapNames::getClNamespace() +
             "get_pointer_type(" + Ptr + ", " + CallExprArguReplVec[0] +
             "->get_context())!=" + MapNames::getClNamespace() +
             "usm::alloc::shared) {";
    };

    auto DefaultQueue = DpctGlobalInfo::getDefaultQueue(CE);

    if (FuncName == "cublasSrotg_v2" || FuncName == "cublasDrotg_v2" ||
        FuncName == "cublasCrotg_v2" || FuncName == "cublasZrotg_v2" ||
        FuncName == "cublasSrotmg_v2" || FuncName == "cublasDrotmg_v2") {
      std::string Prefix, Suffix;
      // Use the second argument to check pointer type
      ExprAnalysis EAPointer(CE->getArg(1));
      std::string IfStmtStr = getIfStmtStr(EAPointer.getReplacedString());

      auto declareTempArgs = [&](std::string &Var,
                                 const std::string &ArgNamePrefix,
                                 const std::string &Type, const int Idx) {
        Var = ArgNamePrefix +
              std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
        Prefix = Prefix + Type + "* " + Var + " = " +
                 ExprAnalysis::ref(CE->getArg(Idx)) + ";" + getNL() + IndentStr;
        if (Type == MapNames::getClNamespace() + "float2")
          CallExprArguReplVec[Idx] = "(std::complex<float>*)" + Var;
        else if (Type == MapNames::getClNamespace() + "double2")
          CallExprArguReplVec[Idx] = "(std::complex<double>*)" + Var;
        else
          CallExprArguReplVec[Idx] = Var;
      };
      auto copyBack = [&](const std::string &Var, const std::string &Size,
                          const std::string &Type, const int Idx) {
        if (Size == "1")
          Suffix = Suffix + getNL() + IndentStr + "  " +
                   getDrefName(CE->getArg(Idx)) + " = *" + Var + ";";
        else {
          Suffix = Suffix + getNL() + IndentStr + "  " + DefaultQueue +
                   ".memcpy(" + ExprAnalysis::ref(CE->getArg(Idx)) + ", " +
                   Var + ", sizeof(" + Type + ")*" + Size + ").wait();";
          requestFeature(HelperFeatureEnum::device_ext);
        }
      };

      if (FuncName == "cublasSrotmg_v2" || FuncName == "cublasDrotmg_v2") {
        std::string Type = (FuncName == "cublasSrotmg_v2") ? "float" : "double";
        std::string D1Ptr, D2Ptr, X1Ptr, Y1Ptr, ParamPtr;

        declareTempArgs(D1Ptr, "d1_ct", Type, 1);
        declareTempArgs(D2Ptr, "d2_ct", Type, 2);
        declareTempArgs(X1Ptr, "x1_ct", Type, 3);
        declareTempArgs(ParamPtr, "param_ct", Type, 5);

        requestFeature(HelperFeatureEnum::device_ext);
        Prefix = Prefix + IfStmtStr + getNL() + IndentStr;
        Prefix = Prefix + "  " + D1Ptr + " = " + MapNames::getClNamespace() +
                 "malloc_shared<" + Type + ">(8, " + DefaultQueue + ");" +
                 getNL() + IndentStr;
        Prefix = Prefix + "  " + D2Ptr + " = " + D1Ptr + " + 1;" + getNL() +
                 IndentStr;
        Prefix = Prefix + "  " + X1Ptr + " = " + D1Ptr + " + 2;" + getNL() +
                 IndentStr;
        Prefix = Prefix + "  " + ParamPtr + " = " + D1Ptr + " + 3;" + getNL() +
                 IndentStr;
        Prefix = Prefix + "  *" + D1Ptr + " = " + getDrefName(CE->getArg(1)) +
                 ";" + getNL() + IndentStr;
        Prefix = Prefix + "  *" + D2Ptr + " = " + getDrefName(CE->getArg(2)) +
                 ";" + getNL() + IndentStr;
        Prefix = Prefix + "  *" + X1Ptr + " = " + getDrefName(CE->getArg(3)) +
                 ";" + getNL() + IndentStr;

        Prefix = Prefix + std::string("}") + getNL() + IndentStr;
        Suffix = Suffix + getNL() + IndentStr + IfStmtStr + getNL() +
                 IndentStr + "  " + CallExprArguReplVec[0] + "->wait();";

        copyBack(D1Ptr, "1", Type, 1);
        copyBack(D2Ptr, "1", Type, 2);
        copyBack(X1Ptr, "1", Type, 3);
        copyBack(ParamPtr, "5", Type, 5);

        Suffix = Suffix + getNL() + IndentStr + "  " +
                 MapNames::getClNamespace() + "free(" + D1Ptr + ", " +
                 DefaultQueue + ");";
        Suffix = Suffix + getNL() + IndentStr + "}";
      } else {
        // cublasSrotg_v2, cublasDrotg_v2, cublasCrotg_v2 or cublasZrotg_v2
        std::string Type, RealType;
        if (FuncName == "cublasSrotg_v2") {
          Type = "float";
          RealType = "float";
        } else if (FuncName == "cublasDrotg_v2") {
          Type = "double";
          RealType = "double";
        } else if (FuncName == "cublasCrotg_v2") {
          Type = MapNames::getClNamespace() + "float2";
          RealType = "float";
        } else {
          Type = MapNames::getClNamespace() + "double2";
          RealType = "double";
        }

        std::string APtr, BPtr, CPtr, SPtr;
        declareTempArgs(APtr, "a_ct", Type, 1);
        declareTempArgs(BPtr, "b_ct", Type, 2);
        declareTempArgs(CPtr, "c_ct", RealType, 3);
        declareTempArgs(SPtr, "s_ct", Type, 4);

        requestFeature(HelperFeatureEnum::device_ext);
        Prefix = Prefix + IfStmtStr + getNL() + IndentStr;
        if (FuncName == "cublasSrotg_v2" || FuncName == "cublasDrotg_v2") {
          Prefix = Prefix + "  " + APtr + " = " + MapNames::getClNamespace() +
                   "malloc_shared<" + Type + ">(4, " +
                   DefaultQueue + ");" +
                   getNL() + IndentStr;
          Prefix = Prefix + "  " + BPtr + " = " + APtr + " + 1;" + getNL() +
                   IndentStr;
          Prefix = Prefix + "  " + CPtr + " = " + APtr + " + 2;" + getNL() +
                   IndentStr;
          Prefix = Prefix + "  " + SPtr + " = " + APtr + " + 3;" + getNL() +
                   IndentStr;
        } else {
          Prefix = Prefix + "  " + APtr + " = " + MapNames::getClNamespace() +
                   "malloc_shared<" + Type + ">(3, " +
                   DefaultQueue + ");" +
                   getNL() + IndentStr;
          Prefix = Prefix + "  " + CPtr + " = " + MapNames::getClNamespace() +
                   "malloc_shared<" + RealType + ">(1, " +
                   DefaultQueue + ");" +
                   getNL() + IndentStr;
          Prefix = Prefix + "  " + BPtr + " = " + APtr + " + 1;" + getNL() +
                   IndentStr;
          Prefix = Prefix + "  " + SPtr + " = " + APtr + " + 2;" + getNL() +
                   IndentStr;
        }
        Prefix = Prefix + "  *" + APtr + " = " + getDrefName(CE->getArg(1)) +
                 ";" + getNL() + IndentStr;
        Prefix = Prefix + "  *" + BPtr + " = " + getDrefName(CE->getArg(2)) +
                 ";" + getNL() + IndentStr;
        Prefix = Prefix + "  *" + CPtr + " = " + getDrefName(CE->getArg(3)) +
                 ";" + getNL() + IndentStr;
        Prefix = Prefix + "  *" + SPtr + " = " + getDrefName(CE->getArg(4)) +
                 ";" + getNL() + IndentStr;

        Prefix = Prefix + std::string("}") + getNL() + IndentStr;
        Suffix = Suffix + getNL() + IndentStr + IfStmtStr + getNL() +
                 IndentStr + "  " + CallExprArguReplVec[0] + "->wait();";

        copyBack(APtr, "1", Type, 1);
        copyBack(BPtr, "1", Type, 2);
        copyBack(CPtr, "1", RealType, 3);
        copyBack(SPtr, "1", Type, 4);

        Suffix = Suffix + getNL() + IndentStr + "  " +
                 MapNames::getClNamespace() + "free(" + APtr + ", " +
                 DefaultQueue + ");";
        if (FuncName == "cublasCrotg_v2" || FuncName == "cublasZrotg_v2") {
          Suffix = Suffix + getNL() + IndentStr + "  " +
                   MapNames::getClNamespace() + "free(" + CPtr + ", " +
                   DefaultQueue + ");";
        }
        Suffix = Suffix + getNL() + IndentStr + "}";
      }
      PrefixInsertStr = Prefix + PrefixInsertStr;
      SuffixInsertStr = Suffix + SuffixInsertStr;
    } else if (MapNames::MaySyncBLASFunc.find(FuncName) !=
               MapNames::MaySyncBLASFunc.end()) {
      auto MaySyncI = MapNames::MaySyncBLASFunc.find(FuncName);
      int ArgIndex = MaySyncI->second.second;
      std::string Type = MaySyncI->second.first;
      ExprAnalysis EA(CE->getArg(ArgIndex));
      std::string ResultTempPtr =
          "res_temp_ptr_ct" +
          std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());

      std::string OriginType;
      if (Type == "std::complex<float>") {
        OriginType = MapNames::getClNamespace() + "float2";
        CallExprArguReplVec[ArgIndex] =
            "(std::complex<float>*)" + ResultTempPtr;
      } else if (Type == "std::complex<double>") {
        OriginType = MapNames::getClNamespace() + "double2";
        CallExprArguReplVec[ArgIndex] =
            "(std::complex<double>*)" + ResultTempPtr;
      } else {
        OriginType = Type;
        CallExprArguReplVec[ArgIndex] = ResultTempPtr;
      }

      std::string IfStmtStr = getIfStmtStr(EA.getReplacedString());

      requestFeature(HelperFeatureEnum::device_ext);
      PrefixInsertStr = OriginType + "* " + ResultTempPtr + " = " +
                        EA.getReplacedString() + ";" + getNL() + IndentStr +
                        IfStmtStr + getNL() + IndentStr + "  " + ResultTempPtr +
                        " = " + MapNames::getClNamespace() + "malloc_shared<" +
                        OriginType + ">(1, " + DefaultQueue + ");" + getNL() +
                        IndentStr + "}" + getNL() + IndentStr + PrefixInsertStr;
      SuffixInsertStr = getNL() + IndentStr + IfStmtStr + getNL() + IndentStr +
                        "  " + CallExprArguReplVec[0] + "->wait();" + getNL() +
                        IndentStr + "  " + getDrefName(CE->getArg(ArgIndex)) +
                        " = *" + ResultTempPtr + ";" + getNL() + IndentStr +
                        "  " + MapNames::getClNamespace() + "free(" +
                        ResultTempPtr + ", " + DefaultQueue + ");" + getNL() +
                        IndentStr + "}" + SuffixInsertStr;
    }
  }

  std::vector<std::string> SyncAPIBufferAssignmentInThenBlock;
  std::vector<std::string> SyncAPIBufferAssignmentInElseBlock;
  std::string processSyncAPIBufferArg(const std::string &FuncName,
                                      const CallExpr *CE,
                                      std::string &PrefixInsertStr,
                                      const std::string &IndentStr,
                                      const std::string &Type,
                                      const int &ArgIndex) {
    std::string BufferName =
        getTempNameForExpr(CE->getArg(ArgIndex), true, true) + "buf_ct" +
        std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());

    PrefixInsertStr = PrefixInsertStr + "auto " + BufferName + " = " +
                      MapNames::getClNamespace() + "buffer<" + Type + ">(" +
                      MapNames::getClNamespace() + "range<1>(1));" + getNL() +
                      IndentStr;
    std::string PointerStr = ExprAnalysis::ref(CE->getArg(ArgIndex));

    bool IsTheLastArgOfRotmg =
        (FuncName == "cublasSrotmg_v2" || FuncName == "cublasDrotmg_v2") &&
        (ArgIndex == 5);

    requestFeature(HelperFeatureEnum::device_ext);
    SyncAPIBufferAssignmentInThenBlock.emplace_back(
        BufferName + " = " + MapNames::getDpctNamespace() + "get_buffer<" +
        Type + ">(" + PointerStr + ");");
    SyncAPIBufferAssignmentInElseBlock.emplace_back(
        BufferName + " = " + MapNames::getClNamespace() + "buffer<" + Type +
        ">(" +
        ((Type == "std::complex<float>" || Type == "std::complex<double>")
             ? std::string("(" + Type + "*)")
             : "") +
        PointerStr + ", " + MapNames::getClNamespace() + "range<1>(" +
        (IsTheLastArgOfRotmg ? "5" : "1") + "));");

    return BufferName;
  }

  void printIfStmt(const std::string &FuncName, const CallExpr *CE,
                   std::string &PrefixInsertStr, const std::string &IndentStr) {
    std::string PointerStr;
    auto getBlockStr = [&](std::vector<std::string> Stmts) -> std::string {
      std::string Res;
      for (const auto &Stmt : Stmts) {
        Res = Res + "  " + Stmt + getNL() + IndentStr;
      }
      return Res;
    };

    auto assembleIfStmt = [&]() {
      requestFeature(HelperFeatureEnum::device_ext);
      std::string IfStmtStr = "if (" + MapNames::getDpctNamespace() +
                              "is_device_ptr(" + PointerStr + ")) {" + getNL() +
                              IndentStr +
                              getBlockStr(SyncAPIBufferAssignmentInThenBlock) +
                              "} else {" + getNL() + IndentStr +
                              getBlockStr(SyncAPIBufferAssignmentInElseBlock) +
                              "}" + getNL() + IndentStr;

      PrefixInsertStr = PrefixInsertStr + IfStmtStr;
    };

    if (MapNames::MaySyncBLASFunc.find(FuncName) !=
        MapNames::MaySyncBLASFunc.end()) {
      auto MaySyncAPIIter = MapNames::MaySyncBLASFunc.find(FuncName);
      PointerStr = ExprAnalysis::ref(CE->getArg(MaySyncAPIIter->second.second));
      assembleIfStmt();
    } else if (MapNames::MustSyncBLASFunc.find(FuncName) !=
               MapNames::MustSyncBLASFunc.end()) {
      PointerStr = ExprAnalysis::ref(CE->getArg(4));
      assembleIfStmt();
    } else if (MapNames::MaySyncBLASFuncWithMultiArgs.find(FuncName) !=
               MapNames::MaySyncBLASFuncWithMultiArgs.end()) {
      auto MaySyncAPIWithMultiArgsIter =
          MapNames::MaySyncBLASFuncWithMultiArgs.find(FuncName);
      PointerStr = ExprAnalysis::ref(
          CE->getArg(MaySyncAPIWithMultiArgsIter->second.begin()->first));
      assembleIfStmt();
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
                     std::string PrefixInsertStr, std::string SuffixInsertStr) {
    if (NeedUseLambda) {
      if (CanAvoidUsingLambda && !IsMacroArg) {
        std::string InsertStr;
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
            !CanAvoidBrace)
          InsertStr = std::string("{") + getNL() + IndentStr + PrefixInsertStr +
                      CallExprReplStr + ";" + SuffixInsertStr + getNL() +
                      IndentStr + "}" + getNL() + IndentStr;
        else
          InsertStr = PrefixInsertStr + CallExprReplStr + ";" +
                      SuffixInsertStr + getNL() + IndentStr;
        auto IT = new InsertText(OuterInsertLoc, std::move(InsertStr));
        IT->setBlockLevelFormatFlag();
        emplaceTransformation(std::move(IT));
        report(OuterInsertLoc, Diagnostics::CODE_LOGIC_CHANGED, true,
               OriginStmtType == "if" ? "an " + OriginStmtType
                                      : "a " + OriginStmtType);
        emplaceTransformation(
            new ReplaceText(FuncNameBegin, FuncCallLength, "0"));
      } else {
        if (IsAssigned) {
          report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_LAMBDA, false);
          insertAroundRange(
              PrefixInsertLoc, SuffixInsertLoc,
              std::string("[&](){") + getNL() + IndentStr + PrefixInsertStr,
              std::string(";") + SuffixInsertStr + getNL() + IndentStr +
                  "return 0;" + getNL() + IndentStr + std::string("}()"),
              true);
        } else {
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            std::string("[&](){") + getNL() + IndentStr +
                                PrefixInsertStr,
                            std::string(";") + SuffixInsertStr + getNL() +
                                IndentStr + std::string("}()"),
                            true);
        }

        emplaceTransformation(new ReplaceText(FuncNameBegin, FuncCallLength,
                                              std::move(CallExprReplStr)));
      }
    } else {
      if (!PrefixInsertStr.empty() || !SuffixInsertStr.empty()) {
        if (dpct::DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
            !CanAvoidBrace)
          insertAroundRange(
              PrefixInsertLoc, SuffixInsertLoc,
              std::string("{") + getNL() + IndentStr + PrefixInsertStr,
              SuffixInsertStr + getNL() + IndentStr + std::string("}"), true);
        else
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            std::move(PrefixInsertStr),
                            std::move(SuffixInsertStr), true);
      }

      emplaceTransformation(new ReplaceText(FuncNameBegin, FuncCallLength,
                                            std::move(CallExprReplStr)));
      if (IsAssigned) {
        insertAroundRange(FuncNameBegin, FuncCallEnd, "DPCT_CHECK_ERROR(", ")");
        requestFeature(HelperFeatureEnum::device_ext);
      }
    }
  }
};

/// Migration rule for Random function calls.
class RandomFunctionCallRule
    : public NamedMigrationRule<RandomFunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for device Random function calls.
class DeviceRandomFunctionCallRule
    : public NamedMigrationRule<DeviceRandomFunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for spBLAS function calls.
class SPBLASFunctionCallRule
    : public NamedMigrationRule<SPBLASFunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for SOLVER enums.
class SOLVEREnumsRule : public NamedMigrationRule<SOLVEREnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for SOLVER function calls.
class SOLVERFunctionCallRule
    : public NamedMigrationRule<SOLVERFunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

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
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  std::string findValueofAttrVar(const Expr *AttrArg, const CallExpr *CE);
};

class EventAPICallRule;
class EventQueryTraversal {
  EventAPICallRule *Rule;
  ASTContext &Context;

  bool QueryCallUsed = false;

  using ResultTy = std::vector<std::pair<const Stmt *, TextModification *>>;

  const VarDecl *getAssignTarget(const CallExpr *);

  bool checkVarDecl(const VarDecl *, const FunctionDecl *);
  bool isEventQuery(const CallExpr *);
  std::string getReplacedEnumValue(const DeclRefExpr *);

  TextModification *buildCallReplacement(const CallExpr *);

  bool traverseFunction(const FunctionDecl *, const VarDecl *);
  bool traverseStmt(const Stmt *, const VarDecl *, ResultTy &);
  bool traverseAssignRhs(const Expr *, ResultTy &);
  bool traverseEqualStmt(const Stmt *, const VarDecl *, ResultTy &);

  void handleDirectEqualStmt(const DeclRefExpr *, const CallExpr *);

  bool startFromStmt(const Stmt *, const std::function<const VarDecl *()> &);

public:
  EventQueryTraversal(EventAPICallRule *R)
      : Rule(R), Context(DpctGlobalInfo::getContext()) {}
  bool startFromQuery(const CallExpr *);
  bool startFromEnumRef(const DeclRefExpr *);
  bool startFromTypeLoc(TypeLoc TL);
};
/// Migration rule for event API calls
class EventAPICallRule : public NamedMigrationRule<EventAPICallRule> {
public:
  EventAPICallRule() { CurrentRule = this; }
  ~EventAPICallRule() {
    if (CurrentRule == this)
      CurrentRule = nullptr;
  }
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  void handleEventRecord(const CallExpr *CE,
                         const ast_matchers::MatchFinder::MatchResult &Result,
                         bool IsAssigned);
  void handleEventElapsedTime(bool IsAssigned);
  void handleTimeMeasurement();
  void handleTargetCalls(const Stmt *Parent, const Stmt *Last = nullptr);
  void handleKernelCalls(const Stmt *Parent, const CUDAKernelCallExpr *KCall);
  void handleOrdinaryCalls(const CallExpr *Call);
  bool IsEventArgArraySubscriptExpr(const Expr *E);
  const Expr *findNextRecordedEvent(const Stmt *Parent, unsigned KCallLoc);

  static EventQueryTraversal getEventQueryTraversal();

private:
  void handleEventRecordWithProfilingEnabled(
      const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result,
      bool IsAssigned);
  void handleEventRecordWithProfilingDisabled(
      const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result,
      bool IsAssigned);
  void findEventAPI(const Stmt *Node, const CallExpr *&Call,
                    const std::string EventAPIName);
  void processAsyncJob(const Stmt *Node);
  void updateAsyncRange(const Stmt *FuncBody, const std::string EventAPIName);
  void updateAsyncRangRecursive(const Stmt *Node, const CallExpr *AsyncCE,
                                const std::string EventAPIName);

  void findThreadSyncLocation(const Stmt *Node);
  const clang::Stmt *getRedundantParenExpr(const CallExpr *Call);
  bool isEventElapsedTimeFollowed(const CallExpr *Expr);
  // Since the state of a rule is shared between multiple matches, it is
  // necessary to clear the previous migration status.
  // The call is supposed to be called whenever a migration on time measurement
  // is triggered.
  void reset() {
    RecordBegin = nullptr;
    RecordEnd = nullptr;
    TimeElapsedCE = nullptr;
    RecordBeginLoc = 0;
    RecordEndLoc = 0;
    TimeElapsedLoc = 0;
    ThreadSyncLoc = 0;
    Events2Wait.clear();
    QueueCounter.clear();
    Queues2Wait.clear();
    DefaultQueueAdded = false;
    IsKernelInLoopStmt = false;
    IsKernelSync = false;
  }
  const Stmt *RecordBegin = nullptr, *RecordEnd = nullptr;
  const CallExpr *TimeElapsedCE = nullptr;
  unsigned RecordBeginLoc = 0;
  unsigned RecordEndLoc = 0;
  unsigned TimeElapsedLoc = 0;

  // To store the location of "cudaThreadSynchronize"
  unsigned ThreadSyncLoc = 0;
  std::vector<std::string> Events2Wait;
  std::map<std::string, int> QueueCounter;
  std::vector<std::pair<std::string, const CallExpr *>> Queues2Wait;
  bool DefaultQueueAdded = false;

  // To check whether kernel call is in loop stmt between RecordBeginLoc and
  // RecordEndLoc
  bool IsKernelInLoopStmt = false;

  // To check whether kernel call needs wait between RecordBeginLoc and
  // RecordEndLoc
  bool IsKernelSync = false;

  std::map<const Stmt *, bool> ExprCache;
  std::map<const VarDecl *, bool> VarDeclCache;

  friend class EventQueryTraversal;
  static EventAPICallRule *CurrentRule;
};

/// Migration rule for stream API calls
class StreamAPICallRule : public NamedMigrationRule<StreamAPICallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for kernel API calls
class KernelCallRule : public NamedMigrationRule<KernelCallRule> {
  std::unordered_set<unsigned> Insertions;

public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  void
  removeTrailingSemicolon(const CallExpr *KCall,
                          const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for device function calls
class DeviceFunctionDeclRule
    : public NamedMigrationRule<DeviceFunctionDeclRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for __constant__/__shared__/__device__ memory variables.
class MemVarRule : public NamedMigrationRule<MemVarRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  void previousHCurrentD(const VarDecl *VD, tooling::Replacement &R);
  void previousDCurrentH(const VarDecl *VD, tooling::Replacement &R);
  void removeHostConstantWarning(tooling::Replacement &R);
  void processTypeDeclaredLocal(const VarDecl *MemVar,
                                std::shared_ptr<MemVarInfo> Info);
  bool currentIsDevice(const VarDecl *MemVar, std::shared_ptr<MemVarInfo> Info);
  bool currentIsHost(const VarDecl *VD, std::string VarName);
};

/// Migration rule for memory management routine.
/// Current implementation is intentionally simplistic. The following things
/// need a more detailed design:
///   - interplay with error handling (possible solution is that we keep
///   function
///     signature as close to original as possible, so return error codes when
///     original functions return them);
///   - SYCL memory buffers are typed. Using a "char" type is definitely a
///   tradeoff.
///     Using better type information requires some kind of global analysis and
///     heuristics, as well as a mechanism for user hint (like "treat all
///     buffers as float-typed")'
///   - interplay with streams need to be designed.
///   - transformation rules are currently unordered, which create potential
///     ambiguity, so need to understand how to handle function call arguments,
///     which are modified by other rules.
///
class MemoryMigrationRule : public NamedMigrationRule<MemoryMigrationRule> {

public:
  MemoryMigrationRule();
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

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
  void replaceMemAPIArg(const Expr *E,
                        const ast_matchers::MatchFinder::MatchResult &Result,
                        const std::string &StreamStr,
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
    if (C->getNumArgs() > ArgIndex) {
      if (C->getArg(ArgIndex)->IgnoreImplicit()->getStmtClass() !=
          Stmt::StmtClass::DeclRefExprClass)
        insertAroundStmt(C->getArg(ArgIndex), "(", ")");
      requestFeature(HelperFeatureEnum::device_ext);
      emplaceTransformation(
          new InsertAfterStmt(C->getArg(ArgIndex), "->to_pitched_data()"));
    }
  }
  void insertZeroOffset(const CallExpr *C, size_t InsertArgIndex) {
    static std::string InsertedText =
        buildString(MapNames::getClNamespace(),
                    DpctGlobalInfo::getCtadClass("id", 3), "(0, 0, 0), ");
    if (C->getNumArgs() > InsertArgIndex)
      emplaceTransformation(new InsertBeforeStmt(C->getArg(InsertArgIndex),
                                                 std::string(InsertedText)));
  }
};

class MemoryDataTypeRule : public NamedMigrationRule<MemoryDataTypeRule> {
  static inline std::string getCtadType(StringRef BaseTypeName) {
    return buildString(DpctGlobalInfo::getCtadClass(
        buildString(MapNames::getClNamespace(), BaseTypeName), 3));
  }
  template <class... Args>
  void emplaceParamDecl(const VarDecl *VD, StringRef ParamType,
                        bool HasInitialZeroCtor, std::string InitValue = "0",
                        Args &&...ParamNames) {
    std::string ParamDecl;
    llvm::raw_string_ostream OS(ParamDecl);
    OS << ParamType << " ";
    unsigned Index = 0;
    printParamNameWithInitArgs(OS, VD->getName(), HasInitialZeroCtor, Index,
                               InitValue, std::forward<Args>(ParamNames)...);
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
                             bool HasInitialZeroCtor, unsigned &Index,
                             StringRef InitValue) {
    return OS;
  }
  template <class... RestNamesT>
  static inline llvm::raw_ostream &
  printParamNameWithInitArgs(llvm::raw_ostream &OS, StringRef BaseName,
                             bool HasInitialZeroCtor, unsigned &Index,
                             StringRef InitValue, StringRef FirstName,
                             RestNamesT &&...Rest) {
    if (Index++)
      OS << ", ";
    printParamName(OS, BaseName, FirstName);
    if (HasInitialZeroCtor)
      OS << "(" << InitValue << ", " << InitValue << ", " << InitValue << ")";
    return printParamNameWithInitArgs(OS, BaseName, HasInitialZeroCtor, Index,
                                      InitValue,
                                      std::forward<RestNamesT>(Rest)...);
  }

  const static MapNames::MapTy MemberNames;
  const static MapNames::MapTy ExtentMemberNames;
  const static MapNames::MapTy PitchMemberNames;
  const static MapNames::MapTy PitchMemberToSetter;
  const static std::map<std::string, HelperFeatureEnum> PitchMemberToFeature;
  const static MapNames::MapTy SizeOrPosToMember;
  const static std::vector<std::string> RemoveMember;

public:
  void emplaceCuArrayDescDeclarations(const VarDecl *VD);
  void emplaceMemcpy3DDeclarations(const VarDecl *VD, bool hasDirection);
  static std::string getMemcpy3DArguments(StringRef BaseName,
                                          bool hasDirection);

  static std::string getMemberName(StringRef BaseName,
                                   const std::string &Member) {
    auto Itr = MemberNames.find(Member);
    if (Itr != MemberNames.end()) {
      std::string ReplacedName;
      llvm::raw_string_ostream OS(ReplacedName);
      printParamName(OS, BaseName, Itr->second);
      return OS.str();
    }
    return Member;
  }

  static std::string getPitchMemberSetter(StringRef BaseName,
                                          const std::string &Member,
                                          const Expr *E);

  static std::string getSizeOrPosMember(StringRef BaseName,
                                        const std::string &Member);

  static bool isRemove(std::string Name) {
    return std::find(RemoveMember.begin(), RemoveMember.end(), Name) !=
           RemoveMember.end();
  }

  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CMemoryAPIRule : public NamedMigrationRule<CMemoryAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Name all unnamed types.
class UnnamedTypesRule : public NamedMigrationRule<UnnamedTypesRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Guess original code indent width.
class GuessIndentWidthRule : public NamedMigrationRule<GuessIndentWidthRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration for math functions
class MathFunctionsRule : public NamedMigrationRule<MathFunctionsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
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
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CooperativeGroupsFunctionRule
    : public NamedMigrationRule<CooperativeGroupsFunctionRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for replacing __syncthreads() function call.
///
/// This rule replace __syncthreads() with item.barrier()
class SyncThreadsRule : public NamedMigrationRule<SyncThreadsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class SyncThreadsMigrationRule
    : public NamedMigrationRule<SyncThreadsMigrationRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migrate Function Attributes to Sycl kernel info, defined in
/// runtime headers.
class KernelFunctionInfoRule
    : public NamedMigrationRule<KernelFunctionInfoRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

  static const std::map<std::string, std::string> AttributesNamesMap;
};

/// RecognizeAPINameRule to give comments for the API not in the record table
class RecognizeAPINameRule : public NamedMigrationRule<RecognizeAPINameRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  const std::string getFunctionSignature(const FunctionDecl *Func,
                                         std::string ObjName);
  std::vector<std::vector<std::string>>
  splitAPIName(std::vector<std::string> &AllAPINames);
  void processFuncCall(const CallExpr *CE);
};

/// RecognizeTypeRule to emit warning message for known unsupported type
class RecognizeTypeRule : public NamedMigrationRule<RecognizeTypeRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class TextureMemberSetRule : public NamedMigrationRule<TextureMemberSetRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
  void removeRange(SourceRange R);
};

/// Texture migration rule
class TextureRule : public NamedMigrationRule<TextureRule> {
  // Get the binary operator if E is lhs of an assign expression.
  const Expr *getAssignedBO(const Expr *E, ASTContext &Context);
  const Expr *getParentAsAssignedBO(const Expr *E, ASTContext &Context);
  bool removeExtraMemberAccess(const MemberExpr *ME);
  void replaceTextureMember(const MemberExpr *ME, ASTContext &Context,
                            SourceManager &SM);
  void replaceResourceDataExpr(const MemberExpr *ME, ASTContext &Context);
  inline const MemberExpr *getParentMemberExpr(const Stmt *S) {
    return DpctGlobalInfo::findParent<MemberExpr>(S);
  }
  std::string getTextureFlagsSetterInfo(const Expr *Flags,
                                        StringRef &SetterName);
  std::string getMemberAssignedValue(const Stmt *AssignStmt,
                                     StringRef MemberName,
                                     StringRef &SetMethodName);
  static MapNames::MapTy ResourceTypeNames;

public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

  static const MapNames::MapTy TextureMemberNames;

private:
  bool processTexVarDeclInDevice(const VarDecl *VD);

  bool tryMerge(const MemberExpr *ME, const Expr *BO);

  class SettersMerger {
    TextureRule *Rule;
    const std::vector<std::string> &MethodNames;
    std::map<const Stmt *, bool> &ProcessedBO;

    const Stmt *Target = nullptr;
    bool Stop = false;
    ValueDecl *D = nullptr;
    bool IsArrow = false;
    std::vector<std::pair<unsigned, const Stmt *>> Result;

    void traverse(const Stmt *);
    void traverseBinaryOperator(const Stmt *);
    bool applyResult();

  public:
    SettersMerger(const std::vector<std::string> &Names, TextureRule *TexRule)
        : Rule(TexRule), MethodNames(Names), ProcessedBO(TexRule->ProcessedBO) {
    }

    bool tryMerge(const Stmt *S);
  };

  std::map<const Stmt *, bool> ProcessedBO;
};

/// CXXNewExprRule is to migrate types in C++ new expressions, e.g.
/// "new cudaStream_t[10]" => "new queue_p[10]"
/// "new cudaStream_t" => "new queue_p"
class CXXNewExprRule : public NamedMigrationRule<CXXNewExprRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class NamespaceRule : public NamedMigrationRule<NamespaceRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class RemoveBaseClassRule : public NamedMigrationRule<RemoveBaseClassRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class PreDefinedStreamHandleRule
    : public NamedMigrationRule<PreDefinedStreamHandleRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class FFTFunctionCallRule : public NamedMigrationRule<FFTFunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class DriverModuleAPIRule : public NamedMigrationRule<DriverModuleAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class DriverDeviceAPIRule : public NamedMigrationRule<DriverDeviceAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class DriverContextAPIRule : public NamedMigrationRule<DriverContextAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CudaArchMacroRule : public NamedMigrationRule<CudaArchMacroRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};
class ComplexAPIRule : public NamedMigrationRule<ComplexAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CudaStreamCastRule : public NamedMigrationRule<CudaStreamCastRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class TypeRemoveRule : public NamedMigrationRule<TypeRemoveRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class TypeMmberRule : public NamedMigrationRule<TypeMmberRule> {
  std::optional<SourceLocation>
  findTokenEndBeforeColonColon(SourceLocation TokStart,
                               const SourceManager &SM);

public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class CompatWithClangRule : public NamedMigrationRule<CompatWithClangRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

TextModification *replaceText(SourceLocation Begin, SourceLocation End,
                              std::string &&Str, const SourceManager &SM);

TextModification *removeArg(const CallExpr *C, unsigned n,
                            const SourceManager &SM) ;
} // namespace dpct
} // namespace clang
#endif // DPCT_AST_TRAVERSAL_H
