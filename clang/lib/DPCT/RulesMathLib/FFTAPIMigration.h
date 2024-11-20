//===--------------- FFTAPIMigration.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_FFT_API_MIGRATION_H
#define DPCT_FFT_API_MIGRATION_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/AST/Expr.h"

namespace clang {
namespace dpct {

TextModification *processFunctionPointer(const UnaryOperator *UO);

class FFTFunctionCallRule : public NamedMigrationRule<FFTFunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};


/// Migration rule for FFT enums.
class FFTEnumsRule : public NamedMigrationRule<FFTEnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};


} // namespace dpct
} // namespace clang

#endif // !DPCT_FFT_API_MIGRATION_H
