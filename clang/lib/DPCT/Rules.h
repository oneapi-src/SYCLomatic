//===--- Rules.h---------------------- -----------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef DPCT_RULES_H
#define DPCT_RULES_H
#include <string>
#include <vector>
#include "llvm/Support/YAMLTraits.h"
#include "CustomHelperFiles.h"

enum RuleKind { API, DataType, Macro };

enum RulePriority { Takeover, Default, Fallback };

class MetaRuleObject {
public:
  static std::string RuleFile;
  std::string RuleId;
  RulePriority Priority;
  RuleKind Kind;
  std::string In;
  std::string Out;
  std::vector<std::string> Includes;
  bool ReturnErrorCode;
  MetaRuleObject() {}
  MetaRuleObject(std::string id,
    RulePriority priority, RuleKind kind)
      : RuleId(id), 
    Priority(priority), Kind(kind) {}
};

template <> struct llvm::yaml::SequenceTraits<std::vector<MetaRuleObject>> {
  static size_t size(llvm::yaml::IO &Io, std::vector<MetaRuleObject> &Seq) {
    return Seq.size();
  }
  static MetaRuleObject &element(IO &, std::vector<MetaRuleObject> &Seq,
                                 size_t Index) {
    if (Index >= Seq.size())
      Seq.resize(Index + 1);
    return Seq[Index];
  }
};

template<> struct llvm::yaml::ScalarEnumerationTraits<RulePriority> {
  static void enumeration(llvm::yaml::IO &Io, RulePriority &Value) {
    Io.enumCase(Value, "Takeover", RulePriority::Takeover);
    Io.enumCase(Value, "Default", RulePriority::Default);
    Io.enumCase(Value, "Fallback", RulePriority::Fallback);
  }
};

template<> struct llvm::yaml::ScalarEnumerationTraits<RuleKind> {
  static void enumeration(llvm::yaml::IO &Io, RuleKind &Value) {
    Io.enumCase(Value, "API", RuleKind::API);
    Io.enumCase(Value, "DataType", RuleKind::DataType);
    Io.enumCase(Value, "Macro", RuleKind::Macro);
  }
};

template <> struct llvm::yaml::MappingTraits<MetaRuleObject> {
  static void mapping(llvm::yaml::IO &Io, MetaRuleObject &Doc) {
    Io.mapRequired("Rule", Doc.RuleId);
    Io.mapRequired("Kind", Doc.Kind);
    Io.mapRequired("Priority", Doc.Priority);
    Io.mapRequired("In", Doc.In);
    Io.mapRequired("Out", Doc.Out);
    Io.mapRequired("Includes", Doc.Includes);
  }
};

class RuleBase {
public:
  std::string Id;
  RulePriority Priority;
  RuleKind Kind;
  std::string In;
  std::string Out;
  clang::dpct::HelperFeatureEnum HelperFeature;
  std::vector<std::string> Includes;
  RuleBase(
      std::string Id, RulePriority Priority, RuleKind Kind, std::string In,
      std::string Out, clang::dpct::HelperFeatureEnum HelperFeature,
      const std::vector<std::string> &Includes = std::vector<std::string>())
      : Id(Id), Priority(Priority), Kind(Kind), In(In), Out(Out),
        HelperFeature(HelperFeature), Includes(Includes) {}
};


class MacroMigrationRule : public RuleBase {
public:
  MacroMigrationRule(
      std::string Id, RulePriority Priority, std::string InStr,
      std::string OutStr,
      clang::dpct::HelperFeatureEnum Helper =
          clang::dpct::HelperFeatureEnum::no_feature_helper,
      const std::vector<std::string> &Includes = std::vector<std::string>())
      : RuleBase(Id, Priority, RuleKind::Macro, InStr, OutStr, Helper,
                 Includes) {}
};

int ImportRules(std::string RuleFile);

#endif // DPCT_RULES_H