//===---------------LIBCUAPIMigration.h---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

#ifndef DPCT_LIBCU_API_MIGRATION_H
#define DPCT_LIBCU_API_MIGRATION_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace dpct {

class LIBCURule : public NamedMigrationRule<LIBCURule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  void processLIBCUFuncCall(const CallExpr *CE);
  void processLIBCUMemberCall(const CXXMemberCallExpr *MC);
  void processLIBCUTypeLoc(const TypeLoc *TL);
  void processCubTypeDef(const TypedefDecl *TD);
  void processLIBCUUsingDirectiveDecl(const UsingDirectiveDecl *UDD);
};
} // namespace dpct
} // namespace clang

#endif // DPCT_LIBCU_API_MIGRATION_H