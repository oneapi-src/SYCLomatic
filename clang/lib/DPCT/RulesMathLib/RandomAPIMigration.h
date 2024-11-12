//===------------------ RandomAPIMigration.h-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_RANDOM_API_MIGRATION_H
#define DPCT_RANDOM_API_MIGRATION_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace dpct {


/// Migration rule for RANDOM enums.
class RandomEnumsRule : public NamedMigrationRule<RandomEnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for Random function calls.
class RandomFunctionCallRule
    : public NamedMigrationRule<RandomFunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for device Random function calls.
class DeviceRandomFunctionCallRule
    : public NamedMigrationRule<DeviceRandomFunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};



} // namespace dpct
} // namespace clang

#endif // DPCT_RANDOM_API_MIGRATION_H