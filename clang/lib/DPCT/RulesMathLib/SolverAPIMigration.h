//===------------------ SolverAPIMigration.h ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_SOLVER_API_MIGRATION_H
#define DPCT_SOLVER_API_MIGRATION_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace dpct {
/// Migration rule for SOLVER enums.
class SOLVEREnumsRule : public NamedMigrationRule<SOLVEREnumsRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

/// Migration rule for SOLVER function calls.
class SOLVERFunctionCallRule
    : public NamedMigrationRule<SOLVERFunctionCallRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

  bool isReplIndex(int i, std::vector<int> &IndexInfo, int &IndexTemp);

  std::string getBufferNameAndDeclStr(const Expr *Arg, const ASTContext &AC,
                                      const std::string &TypeAsStr,
                                      SourceLocation SL,
                                      std::string &BufferDecl,
                                      int DistinctionID);
  void getParameterEnd(const SourceLocation &ParameterEnd,
                       SourceLocation &ParameterEndAfterComma,
                       const ast_matchers::MatchFinder::MatchResult &Result);
  const clang::VarDecl *getAncestralVarDecl(const clang::CallExpr *CE);
};

} // namespace dpct
} // namespace clang

#endif // DPCT_SOLVER_API_MIGRATION_H