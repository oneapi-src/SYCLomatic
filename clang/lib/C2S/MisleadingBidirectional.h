//===--- MisleadingBidirectionalCheck.h - clang-tidy ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MISLEADING_BIDIRECTIONAL_H
#define MISLEADING_BIDIRECTIONAL_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"

namespace clang {
namespace c2s {

class MisleadingBidirectionalHandler : public CommentHandler {
  TransformSetTy &TransformSet;

public:
  MisleadingBidirectionalHandler(TransformSetTy &TransformSet)
      : TransformSet(TransformSet){};
  bool HandleComment(Preprocessor &PP, SourceRange Range) override;
};

class MisleadingBidirectionalRule
    : public NamedMigrationRule<MisleadingBidirectionalRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

} // namespace c2s
} // namespace clang
#endif // MISLEADING_BIDIRECTIONAL_H