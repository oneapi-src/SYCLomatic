//===--------------- LIBCUAPIMigration.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_LIBCU_API_MIGRATION_H
#define DPCT_LIBCU_API_MIGRATION_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang{
namespace dpct{

class LIBCUAPIRule : public NamedMigrationRule<LIBCUAPIRule>{
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class LIBCUTypeRule : public NamedMigrationRule<LIBCUTypeRule>{
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

class LIBCUMemberFuncRule : public NamedMigrationRule<LIBCUMemberFuncRule>{
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};


} // dpct
} // clang

#endif // DPCT_LIBCU_API_MIGRATION_H