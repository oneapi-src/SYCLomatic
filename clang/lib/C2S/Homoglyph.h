//===--- Homoglyph.h - clang-tidy -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOMOGLYPH_H
#define HOMOGLYPH_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"

namespace clang {
namespace c2s {

class ConfusableIdentifierDetectionRule
    : public NamedMigrationRule<ConfusableIdentifierDetectionRule> {
  llvm::StringMap<llvm::SmallVector<NamedDecl const *>> Mapper;
  std::string skeleton(StringRef Name);
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

} // namespace c2s
} // namespace clang
#endif // HOMOGLYPH_H
