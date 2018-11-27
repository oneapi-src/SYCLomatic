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

#ifndef NDEBUG // Debug build

static constexpr bool IsReleaseBuild = false;
static constexpr bool IsDebugBuild = true;

#define SYCLCT_DEBUG_WITH_TYPE(TYPE, X)                                        \
  DebugTypeRegister(TYPE);                                                     \
  DEBUG_WITH_TYPE(TYPE, X)

// General debug information with TYPE = "syclct"
#define SYCLCT_DEBUG(X) SYCLCT_DEBUG_WITH_TYPE("syclct", X)

// End of Debug Build

#else // Release build

static constexpr bool IsReleaseBuild = true;
static constexpr bool IsDebugBuild = false;

#define SYCLCT_DEBUG_WITH_TYPE(TYPE, X)                                        \
  do {                                                                         \
  } while (false)

#define SYCLCT_DEBUG(X)                                                        \
  do {                                                                         \
  } while (false)

#endif // End of Release build

class DebugTypeRegister {
public:
  DebugTypeRegister(const std::string &type);
};

class DebugInfo {
public:
  static void ShowStatistics(int status = 0);
  static void
  printTranslationRules(const std::vector<std::unique_ptr<ASTTraversal>> &TRs);
  static void printMatchedRules(
      const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules);
  static void printReplacements(ReplacementFilter &ReplFilter,
                                clang::ASTContext &Context);
};

} // namespace syclct
} // namespace clang

#endif // SYCLCT_DEBUG_H
