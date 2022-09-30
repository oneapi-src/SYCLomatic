//===--------------- MigrationRuleManager.h -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_MIGRATION_RULE_MANAGER_H
#define DPCT_MIGRATION_RULE_MANAGER_H

#include "ASTTraversal.h"

#include <memory>
#include <unordered_map>

namespace clang {
namespace dpct {

class MigrationRule;

struct MigrationRuleFactoryBase {
  virtual std::unique_ptr<MigrationRule> createMigrationRule() const = 0;
  virtual ~MigrationRuleFactoryBase() {}
};

/// Pass manager for Migration instances.
class MigrationRuleManager {
  using RuleFactoryMapType =
      std::unordered_map<std::string,
                         std::shared_ptr<MigrationRuleFactoryBase>>;

  const PassKind PK;
  std::vector<std::unique_ptr<MigrationRule>> Storage;
  ast_matchers::MatchFinder Matchers;
  TransformSetTy &Transformers;

  static RuleFactoryMapType &getFactoryMap(PassKind Kind) {
    static RuleFactoryMapType Map[static_cast<unsigned>(PassKind::PK_End)];
    return Map[static_cast<unsigned>(Kind)];
  }

  void emplaceMigrationRule(const RuleFactoryMapType::value_type &RuleFactory);

  void emplaceAllRules();
  void emplaceRules(const std::vector<std::string> &RuleNames);

public:
  MigrationRuleManager(PassKind PK, TransformSetTy &Transformers);

  static void registerRule(PassKind Kind, const std::string &Name,
                           std::shared_ptr<MigrationRuleFactoryBase> F) {
    getFactoryMap(Kind)[Name] = F;
  }

  void matchAST(ASTContext &Context, const std::vector<std::string> &RuleNames);
};

template <class RuleTy> class RuleRegister {
  struct RuleFactory : public MigrationRuleFactoryBase {
    std::unique_ptr<MigrationRule> createMigrationRule() const override {
      return std::make_unique<RuleTy>();
    }
  };

public:
  RuleRegister(const std::string &Name, std::initializer_list<PassKind> Kinds) {
    static std::shared_ptr<RuleFactory> F = std::make_shared<RuleFactory>();
    for (auto Kind : Kinds)
      MigrationRuleManager::registerRule(Kind, Name, F);
  }
};

#define REGISTER_RULE(TYPE_NAME, ...)                                          \
  RuleRegister<TYPE_NAME> g_##TYPE_NAME(#TYPE_NAME, {__VA_ARGS__});

} // namespace dpct
} // namespace clang

#endif // DPCT_MIGRATION_RULE_MANAGER_H
