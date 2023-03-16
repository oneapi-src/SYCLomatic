//===--------------------- InlineAsmMigration.h----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_INLINE_ASM_MIGRATION
#define CLANG_DPCT_INLINE_ASM_MIGRATION

#include "ASTTraversal.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace dpct {

class AsmRule : public NamedMigrationRule<AsmRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

} // namespace dpct
} // namespace clang

#endif // CLANG_DPCT_INLINE_ASM_MIGRATION
