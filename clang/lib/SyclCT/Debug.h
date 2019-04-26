//===--- Debug.h ---------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef SYCLCT_DEBUG_H
#define SYCLCT_DEBUG_H

#include "TextModification.h"

#include <map>
#include <memory>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "llvm/Support/Debug.h"

namespace clang {
namespace syclct {

class ASTTraversal;

#ifdef NDEBUG
#undef SYCLCT_DEBUG_BUILD
#else
#undef SYCLCT_DEBUG_BUILD
#define SYCLCT_DEBUG_BUILD true
#endif

#ifdef SYCLCT_DEBUG_BUILD
// Stat of debug build

#define SYCLCT_DEBUG_WITH_TYPE(TYPE, X)                                        \
  DebugTypeRegister(TYPE);                                                     \
  DEBUG_WITH_TYPE(TYPE, X)

// General debug information with TYPE = "syclct"
#define SYCLCT_DEBUG(X) SYCLCT_DEBUG_WITH_TYPE("syclct", X)

class DebugTypeRegister {
public:
  DebugTypeRegister(const std::string &type);
};

// End of debug Build
#else
// Start of release build

#define SYCLCT_DEBUG_WITH_TYPE(TYPE, X)                                        \
  do {                                                                         \
  } while (false)

#define SYCLCT_DEBUG(X)                                                        \
  do {                                                                         \
  } while (false)

// End of release build
#endif

class DebugInfo {
public:
  static void ShowStatus(int status = 0);
  static void
  printTranslationRules(const std::vector<std::unique_ptr<ASTTraversal>> &TRs);
  static void printMatchedRules(
      const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules);
  static void printReplacements(ReplacementFilter &ReplFilter,
                                clang::ASTContext &Context);
};

llvm::raw_ostream &SyclctLog();
llvm::raw_ostream &SyclctStats();
llvm::raw_ostream &SyclctDiags();
llvm::raw_ostream &SyclctTerm();
std::string getSyclctStatsStr();
std::string getSyclctDiagsStr();
std::string getSyclctTermStr();
extern int VerboseLevel;

extern std::unordered_map<std::string, std::array<unsigned int, 3>>
    LOCStaticsMap;
extern std::unordered_map<std::string, unsigned int> SrcAPIStaticsMap;

enum VerboseLevel {
  NonVerbose = 0,
  VerboseLow = 1,
  VerboseHigh = 2,
};

#ifdef SYCLCT_DEBUG_BUILD // Debug build
#define syclct_unreachable(message)                                            \
  do {                                                                         \
    DebugInfo::ShowStatus();                                                   \
    llvm::dbgs() << message << "\n";                                           \
    llvm::dbgs() << "Abortion at " << __FILE__ << ":" << __LINE__ << "\n";     \
    abort();                                                                   \
  } while (false)
#else // Release build
#define syclct_unreachable(message)                                            \
  do {                                                                         \
  } while (false)
#endif // Release build
} // namespace syclct
} // namespace clang

#endif // SYCLCT_DEBUG_H
