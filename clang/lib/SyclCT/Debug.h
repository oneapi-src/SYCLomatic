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

constexpr bool IsDebugBuild = true;

// General debug information with TYPE = "syclct"
#define SYCLCT_DEBUG(X) DEBUG_WITH_TYPE("syclct", X)

// End of Debug Build

#else // Release build

constexpr bool IsDebugBuild = false;

#define SYCLCT_DEBUG(X)                                                        \
  do {                                                                         \
  } while (false)

#endif // End of Release build

class DebugInfo {
public:
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
