//===--------------- ThrustAPIMigration.h-----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_THRUST_API_MIGRATION_H
#define DPCT_THRUST_API_MIGRATION_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace dpct {

class ThrustAPIRule : public NamedMigrationRule<ThrustAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  std::set<SourceLocation> SortULExpr;
  void thrustFuncMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                           const CallExpr *C,
                           const UnresolvedLookupExpr *ULExpr = NULL);
};

class ThrustTypeRule : public NamedMigrationRule<ThrustTypeRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  void thrustCtorMigration(const CXXConstructExpr *CE);
  void replacePlaceHolderExpr(const CXXConstructExpr *CE);
};

} // namespace dpct
} // namespace clang

#endif // DPCT_THRUST_API_MIGRATION_H