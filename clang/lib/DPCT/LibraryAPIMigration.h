//===--- LibraryAPIMigration.h ---------------------------*- C++ -*---===//
//
// Copyright (C) 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_LIBRARY_API_MIGRAION_H
#define DPCT_LIBRARY_API_MIGRAION_H

#include "Diagnostics.h"
#include "ExprAnalysis.h"
#include "Utility.h"

namespace clang {
namespace dpct {
struct LibraryMigrationFlags {
  bool NeedUseLambda = false;
  bool CanAvoidUsingLambda = false;
  bool IsMacroArg = false;
  bool CanAvoidBrace = false;
  bool IsAssigned = false;
  bool MoveOutOfMacro = false;
  std::string OriginStmtType;
};
struct LibraryMigrationLocations {
  SourceLocation PrefixInsertLoc;
  SourceLocation SuffixInsertLoc;
  SourceLocation OuterInsertLoc;
  SourceLocation FuncNameBegin;
  SourceLocation FuncCallEnd;
  SourceLocation OutOfMacroInsertLoc;
  unsigned int Len = 0;
};
struct LibraryMigrationStrings {
  std::string PrePrefixInsertStr;
  std::string PrefixInsertStr;
  std::string Repl;
  std::string SuffixInsertStr;
  std::string IndentStr;
};

class FFTFunctionCallBuilder {
public:
  FFTFunctionCallBuilder(const clang::CallExpr *TheCallExpr,
                         std::string IndentStr, StringRef FuncName,
                         LibraryMigrationLocations Locations, LibraryMigrationFlags Flags)
      : TheCallExpr(TheCallExpr), IndentStr(IndentStr), FuncName(FuncName),
        Locations(Locations), Flags(Flags) {
    for (size_t i = 0; i < TheCallExpr->getNumArgs(); ++i) {
      ArgsList.emplace_back(ExprAnalysis::ref(TheCallExpr->getArg(i)));
    }
  }


  void update1D2D3DCommitCallExpr(unsigned int DescIdx,
                                  std::vector<int> DimsIdx,
                                  unsigned int PrecDomainIdx, int QueueIndex);
  void updateManyCommitCallExpr(int QueueIndex);
  void setValueFor1DBatched(unsigned int DescIdx, unsigned int SizeIdx,
                            unsigned int BatchIdx);
  void assembleExecCallExpr(const Expr *DirExpr, int Index);
  void assembleExecCallExpr(int64_t Dir, int Index);
  std::string getPrefixString();
  std::string getSuffixString();
  std::string getCallExprReplString();
  std::string getPrePrefixString();
  bool moveDeclOutOfBracesIfNeeds(const LibraryMigrationFlags Flags,
                                  SourceLocation &TypeBegin, int &TypeLength);

private:
  void updateCommitCallExpr(unsigned int DescIdx, std::vector<std::string> Dims,
                            unsigned int PrecDomainIdx, int QueueIndex);
  void addDescriptorTypeInfo(unsigned int DescIdx, std::string PrecAndDomain);
  std::string getPrecAndDomainStr(unsigned int PrecDomainIdx);
  void updateBufferArgs(unsigned int Idx, const std::string &TypeStr,
                        std::string ExtraIndent = "");
  const clang::CallExpr *TheCallExpr;
  std::string PrePrefixStmt;
  std::vector<std::string> PrefixStmts;
  std::vector<std::string> SuffixStmts;
  std::vector<std::string> ArgsList;
  std::string IndentStr;
  StringRef FuncName;
  std::string CallExprRepl;
  LibraryMigrationLocations Locations;
  LibraryMigrationFlags Flags;

  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  inline void report(SourceLocation SL, IDTy MsgID, bool UseTextBegin, Ts &&... Vals) {
    TransformSetTy TS;
    auto &SM = DpctGlobalInfo::getSourceManager();
    if (SL.isMacroID() && !SM.isMacroArgExpansion(SL)) {
      auto ItMatch = dpct::DpctGlobalInfo::getMacroTokenToMacroDefineLoc().find(
          getHashStrFromLoc(SM.getImmediateSpellingLoc(SL)));
      if (ItMatch !=
          dpct::DpctGlobalInfo::getMacroTokenToMacroDefineLoc().end()) {
        if (ItMatch->second->IsInRoot) {
          SL = ItMatch->second->NameTokenLoc;
        }
      }
    }
    DiagnosticsUtils::report<IDTy, Ts...>(
        SL, MsgID, DpctGlobalInfo::getCompilerInstance(), &TS, UseTextBegin,
        std::forward<Ts>(Vals)...);
    for (auto &T : TS)
      DpctGlobalInfo::getInstance().addReplacement(
          T->getReplacement(DpctGlobalInfo::getContext()));
  }

  std::string getDescr() {
    std::string Descr;
    if (FuncName.startswith("cufftMake") || FuncName.startswith("cufftExec"))
      Descr = ArgsList[0];
    else
      Descr = getDrefName(TheCallExpr->getArg(0));
    return Descr;
  }

  // Prefix has included the array since all Descr are migrated to shared_ptr
  std::string getDescrMemberCallPrefix() {
    std::string MemberCallPrefix;
    std::string Descr = getDescr();
    if ('*' == *Descr.begin()) {
      MemberCallPrefix = "(" + Descr + ")->";
    } else {
      MemberCallPrefix = Descr + "->";
    }
    return MemberCallPrefix;
  }

};

void initVars(const CallExpr *CE, LibraryMigrationFlags &Flags,
              LibraryMigrationStrings &ReplaceStrs,
              LibraryMigrationLocations &Locations);


} // namespace dpct
} // namespace clang

#endif // !DPCT_LIBRARY_API_MIGRAION_H
