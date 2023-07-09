//===--------------- WMMAAPIMigration.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef WMMA_API_MIGRATION_H
#define WMMA_API_MIGRATION_H

#include "ASTTraversal.h"

using namespace clang::ast_matchers;

namespace clang {
namespace dpct {

class WMMARule : public NamedMigrationRule<WMMARule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

} // namespace dpct
} // namespace clang

#endif
