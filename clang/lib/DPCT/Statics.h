//===--------------- Statics.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

class MigrationRule;

#ifdef NDEBUG
#undef DPCT_DEBUG_BUILD
#else
#undef DPCT_DEBUG_BUILD
#define DPCT_DEBUG_BUILD 1
#endif

class StaticsInfo {
public:
  static void
  printMigrationRules(const std::vector<std::unique_ptr<MigrationRule>> &TRs);
  static void printMatchedRules(
      const std::vector<std::unique_ptr<MigrationRule>> &MatchedRules);
  static void printReplacements(const TransformSetTy &TS, ASTContext &Context);
};

llvm::raw_ostream &DpctLog();
llvm::raw_ostream &DpctStats();
llvm::raw_ostream &DpctDiags();
llvm::raw_ostream &DpctTerm();
llvm::raw_ostream &DpctDebugs();
std::string getDpctStatsStr();
std::string getDpctDiagsStr();
std::string getDpctTermStr();
std::string getDpctLogStr();
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
} // namespace dpct
} // namespace clang
#endif // DPCT_DEBUG_H
