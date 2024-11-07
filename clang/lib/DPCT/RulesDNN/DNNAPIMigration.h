//===---DNNAPIMigration.h -------------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_DNN_API_MIGRATION_H
#define DPCT_DNN_API_MIGRATION_H

#include "ASTTraversal.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace dpct {

class CuDNNTypeRule : public NamedMigrationRule<CuDNNTypeRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);

  static std::map<std::string, std::string> CuDNNEnumNamesMap;
  static std::map<std::string, HelperFeatureEnum>
      CuDNNEnumNamesHelperFeaturesMap;
};

class CuDNNAPIRule : public NamedMigrationRule<CuDNNAPIRule> {
public:
  void registerMatcher(ast_matchers::MatchFinder &MF) override;
  void runRule(const ast_matchers::MatchFinder::MatchResult &Result);
};

} // namespace dpct
} // namespace clang

#endif // !DPCT_DNN_API_MIGRATION_H