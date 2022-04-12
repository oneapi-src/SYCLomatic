//===--- Statics.h ---------------------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef C2S_DEBUG_H
#define C2S_DEBUG_H

#include "TextModification.h"

#include <map>
#include <memory>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "llvm/Support/Debug.h"

namespace clang {
namespace c2s {

class ASTTraversal;

#ifdef NDEBUG
#undef C2S_DEBUG_BUILD
#else
#undef C2S_DEBUG_BUILD
#define C2S_DEBUG_BUILD 1
#endif

class StaticsInfo {
public:
  static void
  printMigrationRules(const std::vector<std::unique_ptr<ASTTraversal>> &TRs);
  static void printMatchedRules(
      const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules);
  static void printReplacements(const TransformSetTy &TS, ASTContext &Context);
};

llvm::raw_ostream &C2SLog();
llvm::raw_ostream &C2SStats();
llvm::raw_ostream &C2SDiags();
llvm::raw_ostream &C2STerm();
std::string getC2SStatsStr();
std::string getC2SDiagsStr();
std::string getC2STermStr();
std::string getC2SLogStr();
extern int VerboseLevel;

extern std::unordered_map<std::string, std::array<unsigned int, 3>>
    LOCStaticsMap;
extern std::map<std::string, unsigned int> SrcAPIStaticsMap;

enum VerboseLevel {
  VL_NonVerbose = 0,
  VL_VerboseLow = 1,
  VL_VerboseHigh = 2,
};

void PrintMsg(const std::string &Msg, bool IsPrintOnNormal = true);
} // namespace c2s
} // namespace clang
#endif // C2S_DEBUG_H
