//===------------------ SpBLASAPIMigration.h ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_SPBLAS_API_MIGRATION_H
#define DPCT_SPBLAS_API_MIGRATION_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace dpct {

class SpBLASTypeLocRule
    : public clang::dpct::NamedMigrationRule<SpBLASTypeLocRule> {
public:
  void registerMatcher(clang::ast_matchers::MatchFinder &MF) override;
  void runRule(const clang::ast_matchers::MatchFinder::MatchResult &Result);
};

class SPBLASEnumsRule : public NamedMigrationRule<SPBLASEnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for spBLAS function calls.
class SPBLASFunctionCallRule
    : public NamedMigrationRule<SPBLASFunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};


} // namespace dpct
} // namespace clang

#endif // DPCT_SPBLAS_API_MIGRATION_H