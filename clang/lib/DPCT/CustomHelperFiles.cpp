//===--------------- CustomHelperFiles.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "ASTTraversal.h"
#include "Config.h"
#include "CustomHelperFiles.h"
#include "DNNAPIMigration.h"
#include "llvm/Support/raw_os_ostream.h"

namespace clang {
namespace dpct {

void requestFeature(HelperFeatureEnum Feature) {
  if (Feature == HelperFeatureEnum::no_feature_helper) {
    return;
  }
  DpctGlobalInfo::setNeedDpctDeviceExt();
}

std::string getDpctVersionStr() {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << DPCT_VERSION_MAJOR << "." << DPCT_VERSION_MINOR << "."
     << DPCT_VERSION_PATCH;
  return OS.str();
}

void requestHelperFeatureForEnumNames(const std::string Name) {
  auto HelperFeatureIter =
      clang::dpct::EnumConstantRule::EnumNamesMap.find(Name);
  if (HelperFeatureIter != clang::dpct::EnumConstantRule::EnumNamesMap.end()) {
    requestFeature(HelperFeatureIter->second->RequestFeature);
    return;
  }
  auto CuDNNHelperFeatureIter =
      clang::dpct::CuDNNTypeRule::CuDNNEnumNamesHelperFeaturesMap.find(Name);
  if (CuDNNHelperFeatureIter !=
      clang::dpct::CuDNNTypeRule::CuDNNEnumNamesHelperFeaturesMap.end()) {
    requestFeature(CuDNNHelperFeatureIter->second);
  }
}
void requestHelperFeatureForTypeNames(const std::string Name) {
  auto HelperFeatureIter = MapNames::TypeNamesMap.find(Name);
  if (HelperFeatureIter != MapNames::TypeNamesMap.end()) {
    requestFeature(HelperFeatureIter->second->RequestFeature);
    return;
  }
  auto CuDNNHelperFeatureIter = MapNames::CuDNNTypeNamesMap.find(Name);
  if (CuDNNHelperFeatureIter != MapNames::CuDNNTypeNamesMap.end()) {
    requestFeature(CuDNNHelperFeatureIter->second->RequestFeature);
  }
}

} // namespace dpct
} // namespace clang