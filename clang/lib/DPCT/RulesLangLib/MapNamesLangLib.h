//===--------------- MapNamesLangLib.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_RULESLANGLIB_MAPNAMES_LANGLIB_H
#define DPCT_RULESLANGLIB_MAPNAMES_LANGLIB_H
#include "CommandOption/ValidateArguments.h"
#include "UserDefinedRules/UserDefinedRules.h"
#include "Utility.h"
#include <map>
#include <set>

namespace clang {
namespace dpct {

class MapNamesLangLib {

  using MapTy = std::map<std::string, std::string>;
  using SetTy = std::set<std::string>;

public:
  static void setExplicitNamespaceMap(
      const std::set<ExplicitNamespace> &ExplicitNamespaces);

  static MapTy CUBEnumsMap;

  struct ThrustFuncReplInfo {
    std::string ReplName;
    std::string ExtraParam;
  };
  using ThrustMapTy = std::map<std::string, ThrustFuncReplInfo>;

  static const SetTy ThrustFileExcludeSet;
  static ThrustMapTy ThrustFuncNamesMap;
  static std::map<std::string, clang::dpct::HelperFeatureEnum>
      ThrustFuncNamesHelperFeaturesMap;

}; // class MapNamesLangLib

} // namespace dpct
} // namespace clang

#endif //! DPCT_RULESLANGLIB_MAPNAMES_LANGLIB_H