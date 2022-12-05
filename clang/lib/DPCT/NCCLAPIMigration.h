//===--------------- NCCLAPIMigration.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NCCL_API_MIGRATION_H
#define NCCL_API_MIGRATION_H

#include "ASTTraversal.h"

using namespace clang::ast_matchers;

namespace clang {
namespace dpct {

class NCCLRule : public NamedMigrationRule<NCCLRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

}
}

#endif