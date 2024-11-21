//===--------------- MapNamesDNN.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_RULESDNN_MAPNAMES_DNN_H
#define DPCT_RULESDNN_MAPNAMES_DNN_H
#include "CommandOption/ValidateArguments.h"
#include "UserDefinedRules/UserDefinedRules.h"
#include "Utility.h"
#include <map>
#include <set>

namespace clang {
namespace dpct {

class MapNamesDNN {

  // using MapTy = std::map<std::string, std::string>;

public:
  static void setExplicitNamespaceMap(
      const std::set<ExplicitNamespace> &ExplicitNamespaces);

  static std::unordered_map<std::string, std::shared_ptr<TypeNameRule>>
      CuDNNTypeNamesMap;

}; // class MapNamesDNN

} // namespace dpct
} // namespace clang

#endif //! DPCT_RULESDNN_MAPNAMES_DNN_H