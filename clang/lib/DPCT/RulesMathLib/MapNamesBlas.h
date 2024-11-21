//===--------------- MapNamesBlas.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_RULESMATHLIB_MAPNAMES_BLAS_H
#define DPCT_RULESMATHLIB_MAPNAMES_BLAS_H
#include "CommandOption/ValidateArguments.h"
#include "UserDefinedRules/UserDefinedRules.h"
#include "Utility.h"
#include <map>
#include <set>

namespace clang {
namespace dpct {

class MapNamesBlas {
  using MapTy = std::map<std::string, std::string>;

public:
  static void setExplicitNamespaceMap(
      const std::set<ExplicitNamespace> &ExplicitNamespaces);

  struct BLASFuncComplexReplInfo {
    std::vector<int> BufferIndexInfo;
    std::vector<int> PointerIndexInfo;
    std::vector<std::string> BufferTypeInfo;
    std::vector<std::string> PointerTypeInfo;
    std::vector<int> OperationIndexInfo;
    int FillModeIndexInfo;
    int SideModeIndexInfo;
    int DiagTypeIndexInfo;
    std::string ReplName;
  };

  struct BLASGemmExTypeInfo {
    std::string OriginScalarType;
    std::string ScalarType;
    std::string OriginABType;
    std::string ABType;
    std::string OriginCType;
    std::string CType;
  };

  static MapTy BLASEnumsMap;
  static MapTy SPBLASEnumsMap;

  static const std::map<std::string, MapNamesBlas::BLASFuncComplexReplInfo>
      LegacyBLASFuncReplInfoMap;
  static std::map<std::string, MapNamesBlas::BLASGemmExTypeInfo>
      BLASTGemmExTypeInfoMap;
  // This map is only used for non-usm.
  static const std::map<std::string, std::map<int, std::string>>
      MaySyncBLASFuncWithMultiArgs;
  static MapTy BLASAPIWithRewriter;
}; // class MapNamesBlas

} // namespace dpct
} // namespace clang

#endif //! DPCT_RULESMATHLIB_MAPNAMES_BLAS_H