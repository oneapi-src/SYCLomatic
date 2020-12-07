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

class FFTFunctionCallBuilder {
public:
  FFTFunctionCallBuilder(const clang::CallExpr *TheCallExpr,
                         std::string IndentStr, std::string FuncName)
      : TheCallExpr(TheCallExpr), IndentStr(IndentStr), FuncName(FuncName) {
    for (size_t i = 0; i < TheCallExpr->getNumArgs(); ++i) {
      ArgsList.emplace_back(ExprAnalysis::ref(TheCallExpr->getArg(i)));
    }
  }

  std::string getPrefixString();
  std::string getSuffixString();
  std::string getCallExprReplString();
  std::string getPrePrefixString();
  bool moveDeclOutOfBracesIfNeeds(const LibraryMigrationFlags Flags,
                                  SourceLocation &TypeBegin, int &TypeLength);
  void updateFFTPlanAPIInfo(FFTPlanAPIInfo &FPAInfo,
                            LibraryMigrationFlags &Flags, int Index);
  void updateFFTExecAPIInfo(std::string FFTExecAPIInfoKey);

private:
  void updateBufferArgs(unsigned int Idx, const std::string &TypeStr,
                        std::string ExtraIndent = "");
  bool isInplace(const Expr *Ptr1, const Expr *Ptr2);
  void addDescriptorTypeInfo(std::string PrecAndDomain);
  std::string getDescrMemberCallPrefix();
  std::string getDescr();
  FFTTypeEnum getFFTType(unsigned int PrecDomainIdx);
  std::string getPrecAndDomainStr(unsigned int PrecDomainIdx);
  void assembleExecCallExpr(int64_t Dir);

  const clang::CallExpr *TheCallExpr;
  std::string PrePrefixStmt;
  std::vector<std::string> PrefixStmts;
  std::vector<std::string> SuffixStmts;
  std::vector<std::string> ArgsList;
  std::string IndentStr;
  std::string FuncName;
  std::string CallExprRepl;

  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  inline void report(SourceLocation SL, IDTy MsgID, bool UseTextBegin,
                     Ts &&... Vals) {
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
};

void initVars(const CallExpr *CE, LibraryMigrationFlags &Flags,
              LibraryMigrationStrings &ReplaceStrs,
              LibraryMigrationLocations &Locations);

} // namespace dpct
} // namespace clang

#endif // !DPCT_LIBRARY_API_MIGRAION_H
