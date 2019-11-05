//===--- Debug.h ---------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef DPCT_DEBUG_H
#define DPCT_DEBUG_H

#include "TextModification.h"

#include <map>
#include <memory>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "llvm/Support/Debug.h"

namespace clang {
namespace dpct {

class ASTTraversal;

#ifdef NDEBUG
#undef DPCT_DEBUG_BUILD
#else
#undef DPCT_DEBUG_BUILD
#define DPCT_DEBUG_BUILD 1
#endif

class DebugInfo {
public:
  static void ShowStatus(int status = 0);
  static void
  printMigrationRules(const std::vector<std::unique_ptr<ASTTraversal>> &TRs);
  static void printMatchedRules(
      const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules);
  static void printReplacements(const TransformSetTy &TS, ASTContext &Context);
};

llvm::raw_ostream &DpctLog();
llvm::raw_ostream &DpctStats();
llvm::raw_ostream &DpctDiags();
llvm::raw_ostream &DpctTerm();
std::string getDpctStatsStr();
std::string getDpctDiagsStr();
std::string getDpctTermStr();
extern int VerboseLevel;

extern std::unordered_map<std::string, std::array<unsigned int, 3>>
    LOCStaticsMap;
extern std::map<std::string, unsigned int> SrcAPIStaticsMap;

enum VerboseLevel {
  NonVerbose = 0,
  VerboseLow = 1,
  VerboseHigh = 2,
};

#ifdef DPCT_DEBUG_BUILD // Debug build
#define dpct_unreachable(message)                                            \
  do {                                                                         \
    DebugInfo::ShowStatus();                                                   \
    llvm::dbgs() << message << "\n";                                           \
    llvm::dbgs() << "Abortion at " << __FILE__ << ":" << __LINE__ << "\n";     \
    abort();                                                                   \
  } while (false)
#else // Release build
#define dpct_unreachable(message)                                            \
  do {                                                                         \
  } while (false)
#endif // Release build

void PrintMsg(const std::string &Msg, bool IsPrintOnNormal = true);
} // namespace dpct
} // namespace clang
#endif // DPCT_DEBUG_H
