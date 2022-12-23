//===--------------- IncrementalMigrationUtility.h ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_INCREMENTAL_MIGRATION_UTILITY_H
#define DPCT_INCREMENTAL_MIGRATION_UTILITY_H

#include "AnalysisInfo.h"
#include <set>
#include <string>

enum class ExplicitNamespace : unsigned int;

namespace clang {
namespace dpct {

const std::string OPTION_AsyncHandler = "AsyncHandler";
const std::string OPTION_NDRangeDim = "NDRangeDim";
const std::string OPTION_CommentsEnabled = "CommentsEnabled";
const std::string OPTION_CustomHelperFileName = "CustomHelperFileName";
const std::string OPTION_CtadEnabled = "CtadEnabled";
const std::string OPTION_ExplicitClNamespace = "ExplicitClNamespace";
const std::string OPTION_ExtensionDEFlag = "ExtensionDEFlag";
const std::string OPTION_ExtensionDDFlag = "ExtensionDDFlag";
const std::string OPTION_NoDRYPattern = "NoDRYPattern";
const std::string OPTION_NoUseGenericSpace = "NoUseGenericSpace";
const std::string OPTION_CompilationsDir = "CompilationsDir";
#ifdef _WIN32
const std::string OPTION_VcxprojFile = "VcxprojFile";
#endif
const std::string OPTION_ProcessAll = "ProcessAll";
const std::string OPTION_SyclNamedLambda = "SyclNamedLambda";
const std::string OPTION_ExperimentalFlag = "ExperimentalFlag";
const std::string OPTION_ExplicitNamespace = "ExplicitNamespace";
const std::string OPTION_UsmLevel = "UsmLevel";
const std::string OPTION_OptimizeMigration = "OptimizeMigration";
const std::string OPTION_RuleFile = "RuleFile";
const std::string OPTION_AnalysisScopePath = "AnalysisScopePath";

bool isOnlyContainDigit(const std::string &Str);
bool convertToIntVersion(std::string VersionStr, unsigned int &Result);
enum class VersionCmpResult {
  VCR_CURRENT_IS_NEWER,
  VCR_CURRENT_IS_OLDER,
  VCR_VERSION_SAME,
  VCR_CMP_FAILED
};
VersionCmpResult compareToolVersion(std::string VersionInYaml);

template <typename T>
inline void setValueToOptMap(std::string Key, T Value, bool Specified) {
  assert(0 && "Unknown value type");
}
template <>
inline void setValueToOptMap(std::string Key, std::string Value,
                             bool Specified) {
  DpctGlobalInfo::getCurrentOptMap()[Key] =
      clang::tooling::OptionInfo(Value, Specified);
}
template <>
inline void setValueToOptMap(std::string Key, bool Value, bool Specified) {
  if (Value)
    setValueToOptMap(Key, std::string("true"), Specified);
  else
    setValueToOptMap(Key, std::string("false"), Specified);
}
template <>
inline void setValueToOptMap(std::string Key, unsigned int Value,
                             bool Specified) {
  setValueToOptMap(Key, std::to_string(Value), Specified);
}
template <>
inline void setValueToOptMap(std::string Key, std::set<ExplicitNamespace> Set,
                             bool Specified) {
  unsigned int Value = 0;
  for (auto &Item : Set) {
    Value |= (1 << static_cast<unsigned int>(Item));
  }
  setValueToOptMap(Key, Value, Specified);
}
template <>
inline void setValueToOptMap(std::string Key, std::vector<std::string> StrVec,
                             bool Specified) {
  std::sort(StrVec.begin(), StrVec.end());
  DpctGlobalInfo::getCurrentOptMap()[Key] =
      clang::tooling::OptionInfo(StrVec, Specified);
}

bool canContinueMigration(std::string &Msg);
} // namespace dpct
} // namespace clang

#endif // DPCT_INCREMENTAL_MIGRATION_UTILITY_H
