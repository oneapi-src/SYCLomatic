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
enum class KernelArgType;
enum class HelperFuncCatalog {
  GetDefaultQueue,
  GetOutOfOrderQueue,
  GetInOrderQueue,
};

const std::string StringLiteralUnsupported{"UNSUPPORTED"};

#define SUPPORTEDVECTORTYPENAMES                                               \
  "char1", "uchar1", "char2", "uchar2", "char3", "uchar3", "char4", "uchar4",  \
      "short1", "ushort1", "short2", "ushort2", "short3", "ushort3", "short4", \
      "ushort4", "int1", "uint1", "int2", "uint2", "int3", "uint3", "int4",    \
      "uint4", "long1", "ulong1", "long2", "ulong2", "long3", "ulong3",        \
      "long4", "ulong4", "float1", "float2", "float3", "float4", "longlong1",  \
      "ulonglong1", "longlong2", "ulonglong2", "longlong3", "ulonglong3",      \
      "longlong4", "ulonglong4", "double1", "double2", "double3", "double4",   \
      "__half", "__half2", "half", "half2", "__nv_bfloat16", "nv_bfloat16",    \
      "__nv_bfloat162", "nv_bfloat162", "__half_raw"
#define VECTORTYPE2MARRAYNAMES "__nv_bfloat162", "nv_bfloat162"

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

  struct ThrustFuncReplInfo {
    std::string ReplName;
    std::string ExtraParam;
  };

  using MapTy = std::map<std::string, std::string>;
  using SetTy = std::set<std::string>;
  using ThrustMapTy = std::map<std::string, ThrustFuncReplInfo>;

  
  static std::unordered_map<std::string, std::shared_ptr<EnumNameRule>>
      EnumNamesMap;
  static const SetTy SupportedVectorTypes;
  static const SetTy VectorTypes2MArray;
  static const std::map<std::string, int> VectorTypeMigratedTypeSizeMap;
  static const std::map<clang::dpct::KernelArgType, int> KernelArgTypeSizeMap;
  static int getArrayTypeSize(const int Dim);
  static const MapTy RemovedAPIWarningMessage;
  static std::unordered_set<std::string> SYCLcompatUnsupportTypes;
  static std::unordered_map<std::string, std::shared_ptr<TypeNameRule>>
      TypeNamesMap;
  static std::unordered_map<std::string, std::shared_ptr<ClassFieldRule>>
      ClassFieldMap;
  static std::unordered_map<std::string, std::shared_ptr<TypeNameRule>>
      CuDNNTypeNamesMap;
  static const MapTy Dim3MemberNamesMap;
  static const std::map<unsigned, std::string> ArrayFlagMap;
  static std::unordered_map<std::string, MacroMigrationRule> MacroRuleMap;
  static std::unordered_map<std::string, MetaRuleObject &> HeaderRuleMap;
  static MapTy CUBEnumsMap;
  static const SetTy ThrustFileExcludeSet;
  static ThrustMapTy ThrustFuncNamesMap;
  static std::map<std::string, clang::dpct::HelperFeatureEnum>
      ThrustFuncNamesHelperFeaturesMap;

  static const MapTy DriverEnumsMap;

  static MapTy ITFName;
  static MapTy RandomEngineTypeMap;
  static MapTy RandomOrderingTypeMap;
  static const std::map<std::string, std::string> RandomGenerateFuncMap;

  static MapTy DeviceRandomGeneratorTypeMap;

  static const std::map<std::string, std::vector<unsigned int>>
      FFTPlanAPINeedParenIdxMap;

  static const std::unordered_set<std::string> CooperativeGroupsAPISet;

  static const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
      SamplingInfoToSetFeatureMap;
  static const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
      SamplingInfoToGetFeatureMap;
  static const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
      ImageWrapperBaseToSetFeatureMap;
  static const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
      ImageWrapperBaseToGetFeatureMap;

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

  static const MapNames::MapTy MemberNamesMap;
  static const MapNames::MapTy MArrayMemberNamesMap;
  static const MapNames::MapTy FunctionAttrMap;
  static const MapNames::SetTy HostAllocSet;

  static std::unordered_map<std::string, std::string> AtomicFuncNamesMap;

  static std::vector<MetaRuleObject::PatternRewriter> PatternRewriters;
  /// {Original API, {ToType, FromType}}
  static std::unordered_map<std::string, std::pair<std::string, std::string>>
      MathTypeCastingMap;

  static std::map<clang::dpct::HelperFuncCatalog, std::string>
      CustomHelperFunctionMap;
};

class MigrationStatistics {
private:
  static std::map<std::string /*API Name*/, bool /*Is Migrated*/>
      MigrationTable;
  static std::map<std::string /*Type Name*/, bool /*Is Migrated*/>
      TypeMigrationTable;

public:
  static bool IsMigrated(const std::string &APIName);
  static std::vector<std::string> GetAllAPINames(void);
  static std::map<std::string, bool> &GetTypeTable(void);
};

} // namespace dpct
} // namespace clang
#endif
