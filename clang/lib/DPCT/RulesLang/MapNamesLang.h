//===--------------- MapNamesLang.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_RULESLANG_MAPNAMESLANG_H
#define DPCT_RULESLANG_MAPNAMESLANG_H

#include "CommandOption/ValidateArguments.h"
#include "UserDefinedRules/UserDefinedRules.h"
#include "Utility.h"
#include <map>
#include <set>

namespace clang {
namespace dpct {
enum class KernelArgType;

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
class MapNamesLang {

public:
  static void setExplicitNamespaceMap(
      const std::set<ExplicitNamespace> &ExplicitNamespaces);

  using MapTy = std::map<std::string, std::string>;
  using SetTy = std::set<std::string>;

  static const SetTy SupportedVectorTypes;
  static const SetTy VectorTypes2MArray;
  static const std::map<std::string, int> VectorTypeMigratedTypeSizeMap;
  static const std::map<clang::dpct::KernelArgType, int> KernelArgTypeSizeMap;
  static int getArrayTypeSize(const int Dim);

  static const MapTy Dim3MemberNamesMap;
  static const std::map<unsigned, std::string> ArrayFlagMap;

  static const std::unordered_set<std::string> CooperativeGroupsAPISet;

  static const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
      SamplingInfoToSetFeatureMap;
  static const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
      SamplingInfoToGetFeatureMap;
  static const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
      ImageWrapperBaseToSetFeatureMap;
  static const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
      ImageWrapperBaseToGetFeatureMap;

  static const MapNamesLang::MapTy MemberNamesMap;
  static const MapNamesLang::MapTy MArrayMemberNamesMap;
  static const MapNamesLang::MapTy FunctionAttrMap;
  static const MapNamesLang::SetTy HostAllocSet;

  static std::unordered_map<std::string, std::string> AtomicFuncNamesMap;

  /// {Original API, {ToType, FromType}}
  static std::unordered_map<std::string, std::pair<std::string, std::string>>
      MathTypeCastingMap;
};

} // namespace dpct
} // namespace clang
#endif //! DPCT_RULESLANG_MAPNAMESLANG_H
