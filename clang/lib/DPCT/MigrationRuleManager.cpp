//===--------------- MigrationRuleManager.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MigrationRuleManager.h"

namespace clang {
namespace dpct {

MigrationRuleManager::MigrationRuleManager(PassKind Kind, TransformSetTy &TS)
    : PK(Kind), Transformers(TS) {}

void MigrationRuleManager::emplaceMigrationRule(const RuleFactoryMapType::value_type &RuleFactory) {
  auto Rule = RuleFactory.second->createMigrationRule();
  Rule->setName(RuleFactory.first);
  Rule->setTransformSet(Transformers);
  Rule->registerMatcher(Matchers);
  Storage.push_back(std::move(Rule));
}

void MigrationRuleManager::emplaceAllRules() {
  for (const auto &F : getFactoryMap(PK))
    emplaceMigrationRule(F);
}

void MigrationRuleManager::emplaceRules(
    const std::vector<std::string> &RuleNames) {
  auto &FactoryMap = getFactoryMap(PK);
  for (const auto &Name : RuleNames) {
    auto FactoryIter = FactoryMap.find(Name);
    if (FactoryIter != FactoryMap.end()) {
      emplaceMigrationRule(*FactoryIter);
    }
  }
}

void MigrationRuleManager::matchAST(ASTContext &Context,
                                    const std::vector<std::string> &RuleNames) {
  if (RuleNames.empty()) {
    emplaceAllRules();
  } else {
    emplaceRules(RuleNames);
  }

  StaticsInfo::printMigrationRules(Storage);

  Matchers.matchAST(Context);

  StaticsInfo::printMatchedRules(Storage);
}

} // namespace dpct
} // namespace clang