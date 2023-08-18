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

  RuleGroupKind Group = RuleGroupKind::RK_Common;
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
  const RuleGroups Groups;

  static RuleFactoryMapType &getFactoryMap(PassKind Kind) {
    static RuleFactoryMapType Map[static_cast<unsigned>(PassKind::PK_End)];
    return Map[static_cast<unsigned>(Kind)];
  }

  void emplaceMigrationRule(const RuleFactoryMapType::value_type &RuleFactory);

  void emplaceAllRules();
  void emplaceRules(const std::vector<std::string> &RuleNames);

public:
  MigrationRuleManager(PassKind PK, TransformSetTy &Transformers, RuleGroups G);

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

  template <class... Args>
  void registRule(std::shared_ptr<RuleFactory> Factory, const std::string &Name,
                  PassKind Kind, Args... As) {
    MigrationRuleManager::registerRule(Kind, Name, Factory);
    registRule(Factory, Name, As...);
  }
  template <class... Args>
  void registRule(std::shared_ptr<RuleFactory> Factory, const std::string &Name,
                  RuleGroupKind Group, Args... As) {
    Factory->Group = Group;
    registRule(Factory, Name, As...);
  }
  void registRule(std::shared_ptr<RuleFactory> Factory,
                  const std::string &Name) {
    return;
  }
  RuleRegister(const std::string &Name, std::initializer_list<PassKind> Kinds) {
    static std::shared_ptr<RuleFactory> F = std::make_shared<RuleFactory>();
    for (auto Kind : Kinds)
      MigrationRuleManager::registerRule(Kind, Name, F);
  }

public:
  template <class... Args> RuleRegister(const std::string &Name, Args... As) {
    auto F = std::make_shared<RuleFactory>();
    registRule(F, Name, As...);
  }
};

#define REGISTER_RULE(TYPE_NAME, ...)                                          \
  RuleRegister<TYPE_NAME> g_##TYPE_NAME(#TYPE_NAME, __VA_ARGS__);

} // namespace dpct
} // namespace clang

#endif // DPCT_MIGRATION_RULE_MANAGER_H
