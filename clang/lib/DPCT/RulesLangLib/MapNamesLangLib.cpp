//===--------------- MapNamesLangLib.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MapNamesLangLib.h"
#include "ASTTraversal.h"
#include "FileGenerator/GenFiles.h"
#include "RuleInfra/CallExprRewriter.h"
#include "RuleInfra/MapNames.h"
#include "RulesDNN/DNNAPIMigration.h"
#include "RulesDNN/MapNamesDNN.h"
#include "RulesLang/RulesLang.h"
#include <map>

using namespace clang;
using namespace clang::dpct;

namespace clang {
namespace dpct {

MapNamesLangLib::MapTy MapNamesLangLib::CUBEnumsMap;

MapNamesLangLib::ThrustMapTy MapNamesLangLib::ThrustFuncNamesMap;
std::map<std::string /*Original API*/, HelperFeatureEnum>
    MapNamesLangLib::ThrustFuncNamesHelperFeaturesMap;

void MapNamesLangLib::setExplicitNamespaceMap(
    const std::set<ExplicitNamespace> &ExplicitNamespaces) {

  // CUB enums mapping
  // clang-format off
  CUBEnumsMap = {
    {"BLOCK_STORE_DIRECT", MapNames::getDpctNamespace() + "group::group_store_algorithm::blocked"},
    {"BLOCK_STORE_STRIPED", MapNames::getDpctNamespace() + "group::group_store_algorithm::striped"},
    {"BLOCK_LOAD_DIRECT", MapNames::getDpctNamespace() + "group::group_load_algorithm::blocked"},
    {"BLOCK_LOAD_STRIPED", MapNames::getDpctNamespace() + "group::group_load_algorithm::striped"}
  };
  // clang-format on

  // Thrust function name mapping
  ThrustFuncNamesMap = {
#define ENTRY(from, to, policy) {from, {to, policy}},
#define ENTRY_HOST(from, to, policy) ENTRY(from, to, policy)
#define ENTRY_DEVICE(from, to, policy) ENTRY(from, to, policy)
#define ENTRY_BOTH(from, to, policy) ENTRY(from, to, policy)
#include "RulesLangLib/APINamesMapThrust.inc"
#undef ENTRY
#undef ENTRY_HOST
#undef ENTRY_DEVICE
#undef ENTRY_BOTH
  };

  ThrustFuncNamesHelperFeaturesMap = {
      {"thrust::sequence", HelperFeatureEnum::device_ext},
      {"thrust::stable_sort_by_key", HelperFeatureEnum::device_ext},
      {"thrust::transform_if", HelperFeatureEnum::device_ext},
      {"thrust::device_free", HelperFeatureEnum::device_ext},
      {"thrust::device_malloc", HelperFeatureEnum::device_ext},
      {"thrust::raw_pointer_cast", HelperFeatureEnum::device_ext},
      {"thrust::make_counting_iterator", HelperFeatureEnum::device_ext},
      {"thrust::device_pointer_cast", HelperFeatureEnum::device_ext},
      {"thrust::make_constant_iterator", HelperFeatureEnum::device_ext},
      {"thrust::partition_point", HelperFeatureEnum::device_ext}};
}

// Files to not preprocess, i.e. ignore #include <file>
const MapNamesLangLib::SetTy MapNamesLangLib::ThrustFileExcludeSet{
    "thrust/detail/adjacent_difference.inl",
    "thrust/detail/binary_search.inl",
    "thrust/detail/complex/complex.inl",
    "thrust/detail/copy_if.h",
    "thrust/detail/count.inl",
    "thrust/detail/equal.inl",
    "thrust/detail/pair.inl",
    "thrust/detail/pointer.inl",
    "thrust/detail/sequence.inl",
    "thrust/detail/sort.inl",
    "thrust/detail/temporary_buffer.h"};

} // namespace dpct
} // namespace clang