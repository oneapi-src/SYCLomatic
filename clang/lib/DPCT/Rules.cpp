//===--- Rules.cpp -------------------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//
#include "Rules.h"
#include "Utility.h"
#include "Error.h"
#include "MapNames.h"
#include "llvm/Support/raw_os_ostream.h"

std::vector<std::string> MetaRuleObject::RuleFiles;

void RegisterMacroRule(MetaRuleObject &R) {
  auto It = MapNames::MacroRuleMap.find(R.In);
  if (It != MapNames::MacroRuleMap.end()) {
    if (It->second.Priority > R.Priority) {
      It->second.Id = R.RuleId;
      It->second.Priority = R.Priority;
      It->second.In = R.In;
      It->second.Out = R.Out;
      It->second.HelperFeature =
        clang::dpct::HelperFeatureEnum::no_feature_helper;
      It->second.Includes = R.Includes;
    }
  } else {
    MapNames::MacroRuleMap.emplace(
        R.In,
        MacroMigrationRule(R.RuleId, R.Priority, R.In, R.Out,
                           clang::dpct::HelperFeatureEnum::no_feature_helper,
                           R.Includes));
  }
}

void ImportRules(llvm::cl::list<std::string> &RuleFiles) {
  for (auto &RuleFile : RuleFiles) {
    makeCanonical(RuleFile);

    // open the yaml file
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(RuleFile);
    if (!Buffer) {
      llvm::errs() << "error: failed to read " << RuleFile << ": "
        << Buffer.getError().message() << "\n";
      clang::dpct::ShowStatus(MigrationErrorInvalidRuleFilePath);
      dpctExit(MigrationErrorInvalidRuleFilePath);
    }

    // load rules
    std::vector<MetaRuleObject> rules;
    llvm::yaml::Input YAMLIn(Buffer.get()->getBuffer());
    YAMLIn >> rules;

    if (YAMLIn.error()) {
      // yaml parsing fail
      clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
      dpctExit(MigrationErrorCannotParseRuleFile);
    }

    MetaRuleObject::setRuleFiles(RuleFile);

    //Register Rules
    for (MetaRuleObject &r : rules) {
      switch (r.Kind) {
      case (RuleKind::Macro):
        RegisterMacroRule(r);
        break;
      default:
        break;
      }
    }
  }
}


