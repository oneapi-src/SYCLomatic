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



} // dpct
} // clang

#endif // DPCT_LIBCU_API_MIGRATION_H