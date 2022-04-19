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

#ifndef C2S_RULES_H
#define C2S_RULES_H
#include <string>
#include <vector>
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/CommandLine.h"
#include "CustomHelperFiles.h"
enum RuleKind { API, DataType, Macro, Header, TypeRule };

enum RulePriority { Takeover, Default, Fallback };

// Record all information of imported rules
class MetaRuleObject {
public:
  static std::vector<std::string> RuleFiles;
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
  static void setRuleFiles(std::string File) {
    RuleFiles.push_back(File);
  }
};

template <>
struct llvm::yaml::SequenceTraits<
    std::vector<std::shared_ptr<MetaRuleObject>>> {
  static size_t size(llvm::yaml::IO &Io,
                     std::vector<std::shared_ptr<MetaRuleObject>> &Seq) {
    return Seq.size();
  }
  static std::shared_ptr<MetaRuleObject> &
  element(IO &, std::vector<std::shared_ptr<MetaRuleObject>> &Seq,
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
    Io.enumCase(Value, "Header", RuleKind::Header);
    Io.enumCase(Value, "Type", RuleKind::TypeRule);
  }
};

template <> struct llvm::yaml::MappingTraits<std::shared_ptr<MetaRuleObject>> {
  static void mapping(llvm::yaml::IO &Io, std::shared_ptr<MetaRuleObject> &Doc) {
    Doc = std::make_shared<MetaRuleObject>();
    Io.mapRequired("Rule", Doc->RuleId);
    Io.mapRequired("Kind", Doc->Kind);
    Io.mapRequired("Priority", Doc->Priority);
    Io.mapRequired("In", Doc->In);
    Io.mapRequired("Out", Doc->Out);
    Io.mapRequired("Includes", Doc->Includes);
  }
};

class RuleBase {
public:
  std::string Id;
  RulePriority Priority;
  RuleKind Kind;
  std::string In;
  std::string Out;
  clang::c2s::HelperFeatureEnum HelperFeature;
  std::vector<std::string> Includes;
  RuleBase(
      std::string Id, RulePriority Priority, RuleKind Kind, std::string In,
      std::string Out, clang::c2s::HelperFeatureEnum HelperFeature,
      const std::vector<std::string> &Includes = std::vector<std::string>())
      : Id(Id), Priority(Priority), Kind(Kind), In(In), Out(Out),
        HelperFeature(HelperFeature), Includes(Includes) {}
};


class MacroMigrationRule : public RuleBase {
public:
  MacroMigrationRule(
      std::string Id, RulePriority Priority, std::string InStr,
      std::string OutStr,
      clang::c2s::HelperFeatureEnum Helper =
          clang::c2s::HelperFeatureEnum::no_feature_helper,
      const std::vector<std::string> &Includes = std::vector<std::string>())
      : RuleBase(Id, Priority, RuleKind::Macro, InStr, OutStr, Helper,
                 Includes) {}
};

// The parsing result of the "Out" attribute of a API rule
// Kind::Top labels the root node.
// For example, if the input "Out" string is:
// foo($1, $deref($2))
// The SubBuilders of the "Top" OutputBuilder will be:
// 1. OutputBuilder: Kind="String", Str="foo("
// 2. OutputBuilder: Kind = "Arg", ArgIndex=1
// 3. OutputBuilder: Kind = "Deref", ArgIndex=2
// 4. OutputBuilder: Kind = "String", Str=")"
class OutputBuilder {
public:
  enum Kind {
    String,
    Top,
    Arg,
    Queue,
    Context,
    Device,
    Deref,
    TypeName,
    AddrOf,
    DerefedTypeName
  };
  std::string RuleName;
  Kind Kind;
  size_t ArgIndex;
  std::string Str;
  std::vector<std::shared_ptr<OutputBuilder>> SubBuilders;
  void parse(std::string&);
private:
  // /OutStr is the string specified in rule's "Out" session
  std::shared_ptr<OutputBuilder> consumeKeyword(std::string &OutStr,
                                                size_t &Idx);
  int consumeArgIndex(std::string &OutStr, size_t &Idx);
  void ignoreWhitespaces(std::string &OutStr, size_t &Idx);
  void consumeRParen(std::string &OutStr, size_t &Idx);
  void consumeLParen(std::string &OutStr, size_t &Idx);
};

void importRules(llvm::cl::list<std::string> &RuleFiles);

#endif // C2S_RULES_H