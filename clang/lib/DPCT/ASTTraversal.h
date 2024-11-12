//===--------------------------- ASTTraversal.h ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_AST_TRAVERSAL_H
#define DPCT_AST_TRAVERSAL_H

#include "AnalysisInfo.h"
#include "ErrorHandle/CrashRecovery.h"
#include "Diagnostics/Diagnostics.h"
#include "TextModification.h"
#include "Utility.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"

namespace clang {
namespace dpct {

enum class PassKind : unsigned { PK_Analysis = 0, PK_Migration, PK_End };

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

  inline static unsigned incPairID() { return ++PairID; }

  const CompilerInstance &getCompilerInstance();

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

  /// Add \a TM to the set of transformations.
  ///
  /// The ownership of the TM is transferred to the TransformSet.
  void emplaceTransformation(TextModification *TM);

  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  bool report(SourceLocation SL, IDTy MsgID, bool UseTextBegin, Ts &&...Vals) {
    return DiagnosticsUtils::report<IDTy, Ts...>(
        SL, MsgID, TransformSet, UseTextBegin, std::forward<Ts>(Vals)...);
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

    DiagnosticsUtils::report<IDTy, Ts...>(
        Begin, MsgID, TransformSet, UseTextBegin, std::forward<Ts>(Vals)...);
  }
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

  /// @brief If necessary, initialize an argument or emit warning.
  /// @param Call Function CallExpr
  /// @param Arg An argument (may be an expression) of \p Call .
  void analyzeUninitializedDeviceVar(const clang::Expr *Call,
                                     const clang::Expr *Arg) {
    if (!Call || !Arg)
      return;
    std::vector<const clang::VarDecl *> DeclsRequireInit;
    int Res = isArgumentInitialized(Arg, DeclsRequireInit);
    if (Res == 0) {
      for (const auto D : DeclsRequireInit) {
        emplaceTransformation(new InsertText(
            D->getEndLoc().getLocWithOffset(Lexer::MeasureTokenLength(
                D->getEndLoc(), DpctGlobalInfo::getSourceManager(),
                DpctGlobalInfo::getContext().getLangOpts())),
            " = 0"));
      }
    } else if (Res == -1) {
      report(Call->getBeginLoc(), Diagnostics::UNINITIALIZED_DEVICE_VAR, false,
             ExprAnalysis::ref(Arg));
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


} // namespace dpct
} // namespace clang
#endif // DPCT_AST_TRAVERSAL_H
