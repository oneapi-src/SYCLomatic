//===--------------- MapNamesRandom.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_RULESMATHLIB_MAPNAMES_RANDOM_H
#define DPCT_RULESMATHLIB_MAPNAMES_RANDOM_H
#include "CommandOption/ValidateArguments.h"
#include "UserDefinedRules/UserDefinedRules.h"
#include "Utility.h"
#include <map>
#include <set>

namespace clang {
namespace dpct {

class MapNamesRandom {
  using MapTy = std::map<std::string, std::string>;

public:
  static MapTy RandomEngineTypeMap;
  static MapTy RandomOrderingTypeMap;
  static const std::map<std::string, std::string> RandomGenerateFuncMap;

  static MapTy DeviceRandomGeneratorTypeMap;

  static void setExplicitNamespaceMap(
      const std::set<ExplicitNamespace> &ExplicitNamespaces);

}; // class MapNamesRandom

} // namespace dpct
} // namespace clang

#endif //! DPCT_RULESMATHLIB_MAPNAMES_RANDOM_H