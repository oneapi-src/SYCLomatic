//===--------------- MapNames.h -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_MAPNAMES_H
#define DPCT_MAPNAMES_H

#include "UserDefinedRules/UserDefinedRules.h"
#include "Utility.h"
#include "CommandOption/ValidateArguments.h"
#include <map>
#include <set>

namespace clang {
namespace dpct {

enum class HelperFuncCatalog {
  GetDefaultQueue,
  GetOutOfOrderQueue,
  GetInOrderQueue,
};
/// Record mapping between names
class MapNames {
  static std::vector<std::string> ClNamespace;
  static std::vector<std::string> DpctNamespace;

public:
  static void setExplicitNamespaceMap(
      const std::set<ExplicitNamespace> &ExplicitNamespaces);
  // KeepNamespace = true for function or type that need avoid ambiguous.
  // Example: sycl::exception <--> std::exception
  // IsMathFunc = true for namespace before math functions.
  // Example: sycl::exp
  static std::string getClNamespace(bool KeepNamespace = false,
                                    bool IsMathFunc = false);
  static std::string getExpNamespace(bool KeepNamespace = false);
  static std::string getDpctNamespace(bool KeepNamespace = false);
  static const std::string &getLibraryHelperNamespace();
  static const std::string &getCheckErrorMacroName();

  using MapTy = std::map<std::string, std::string>;
  using SetTy = std::set<std::string>;

  static std::unordered_map<std::string, std::shared_ptr<EnumNameRule>>
      EnumNamesMap;
  static std::unordered_map<std::string, std::shared_ptr<TypeNameRule>>
      TypeNamesMap;
  static std::unordered_set<std::string> SYCLcompatUnsupportTypes;
  static std::unordered_map<std::string, MacroMigrationRule> MacroRuleMap;
  static std::unordered_map<std::string, MetaRuleObject &> HeaderRuleMap;
  static MapTy ITFName;
  static const MapTy RemovedAPIWarningMessage;
  static std::vector<MetaRuleObject::PatternRewriter> PatternRewriters;
  static std::map<clang::dpct::HelperFuncCatalog, std::string>
      CustomHelperFunctionMap;
  static std::unordered_map<std::string, std::shared_ptr<ClassFieldRule>>
      ClassFieldMap;

  template<class T>
  inline static const std::string &findReplacedName(
      const std::unordered_map<std::string, std::shared_ptr<T>> &Map,
      const std::string &Name) {
    static const std::string EmptyString;

    auto Itr = Map.find(Name);
    if (Itr == Map.end())
      return EmptyString;
    return Itr->second->NewName;
  }
  inline static const std::string &findReplacedName(const MapTy &Map,
                                                    const std::string &Name) {
    static const std::string EmptyString;

    auto Itr = Map.find(Name);
    if (Itr == Map.end())
      return EmptyString;
    return Itr->second;
  }
  template<class T>
  static bool replaceName(
      const std::unordered_map<std::string, std::shared_ptr<T>> &Map,
      std::string &Name) {
    auto &Result = findReplacedName(Map, Name);
    if (Result.empty())
      return false;
    Name = Result;
    return true;
  }
  static bool replaceName(const MapTy &Map, std::string &Name) {
    auto &Result = findReplacedName(Map, Name);
    if (Result.empty())
      return false;
    Name = Result;
    return true;
  }
  static bool isInSet(const SetTy &Set, std::string &Name) {
    return Set.find(Name) != Set.end();
  }
};

} // namespace dpct
} // namespace clang
#endif
