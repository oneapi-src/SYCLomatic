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

class ThrustRule : public NamedMigrationRule<ThrustRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  std::set<SourceLocation> SortULExpr;
  void thrustFuncMigration(const ast_matchers::MatchFinder::MatchResult &Result,
                           const CallExpr *C,
                           const UnresolvedLookupExpr *ULExpr = NULL);
  void thrustCtorMigration(const CXXConstructExpr *CE);
  void replacePlaceHolderExpr(const CXXConstructExpr *CE);
};
} // namespace dpct
} // namespace clang

#endif // DPCT_THRUST_API_MIGRATION_H