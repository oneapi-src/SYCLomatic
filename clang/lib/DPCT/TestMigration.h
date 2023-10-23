//===-------------------------- TestMigration.h ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file only used on test.
// If feature is ready
// this file should be removed
#ifndef TEST_MIGRATION_H
#define TEST_MIGRATION_H

#include "ASTTraversal.h"

using namespace clang::ast_matchers;

namespace clang {
namespace dpct {

class TESTRule : public NamedMigrationRule<TESTRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

}
}

#endif