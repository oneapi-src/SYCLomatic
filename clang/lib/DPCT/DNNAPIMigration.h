//===---DNNAPIMigration.h -------------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_DNN_API_MIGRAION_H
#define DPCT_DNN_API_MIGRAION_H

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

#endif // !DPCT_DNN_API_MIGRAION_H