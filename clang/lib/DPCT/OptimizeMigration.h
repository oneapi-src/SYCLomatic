//===---OptimizeMigration.h ------------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_OPTIMIZE_MIGRAION_H
#define DPCT_OPTIMIZE_MIGRAION_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace dpct {

class ForLoopUnrollRule : public NamedMigrationRule<ForLoopUnrollRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class DeviceConstantVarOptimizeRule
    : public NamedMigrationRule<DeviceConstantVarOptimizeRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

} // namespace dpct
} // namespace clang

#endif // !DPCT_OPTIMIZE_MIGRAION_H