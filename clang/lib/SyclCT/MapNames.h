//===--- MapNames.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_MAPNAMES_H
#define SYCLCT_MAPNAMES_H

#include "Utility.h"
#include <map>
#include <set>

const std::string StringLiteralUnsupported{"UNSUPPORTED"};

class MapNames {
public:
  struct BLASFuncReplInfo {
    std::vector<int> BufferIndexInfo;
    std::vector<int> PointerIndexInfo;
    std::vector<std::string> BufferTypeInfo;
    std::string ReplName;
  };

  struct BLASFuncComplexReplInfo {
    std::vector<int> BufferIndexInfo;
    std::vector<std::string> BufferTypeInfo;
    std::vector<int> PointerIndexInfo;
    std::vector<std::string> PointerTypeInfo;
    std::string ReplName;
  };

  struct ThrustFuncReplInfo {
    std::string ReplName;
    std::string ExtraParam;
  };

  struct LegacyBLASFuncReplInfo {
    std::vector<int> BufferIndexInfo;
    std::vector<int> PointerIndexInfo;
    std::vector<std::string> BufferTypeInfo;
    std::vector<int> OperationIndexInfo;
    int FillModeIndexInfo;
    int SideModeIndexInfo;
    int DiagTypeIndexInfo;
    std::string ReplName;
  };

  using MapTy = std::map<std::string, std::string>;
  using SetTy = std::set<std::string>;
  using ThrustMapTy = std::map<std::string, ThrustFuncReplInfo>;

  static const MapTy TypeNamesMap;
  static const MapTy Dim3MemberNamesMap;
  static const MapTy MacrosMap;
  static const MapTy BLASEnumsMap;
  static const std::map<std::string, MapNames::BLASFuncReplInfo>
      BLASFuncReplInfoMap;
  static const std::map<std::string, MapNames::BLASFuncComplexReplInfo>
      BLASFuncComplexReplInfoMap;
  static const SetTy ThrustFileExcludeSet;
  static const ThrustMapTy ThrustFuncNamesMap;

  static const std::map<std::string, MapNames::LegacyBLASFuncReplInfo>
      LegacyBLASFuncReplInfoMap;

  inline static const std::string &findReplacedName(const MapTy &Map,
                                                    const std::string &Name) {
    static const std::string EmptyString;

    auto Itr = Map.find(Name);
    if (Itr == Map.end())
      return EmptyString;
    return Itr->second;
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

class TranslationStatistics {
private:
  static std::map<std::string /*API Name*/, bool /*Is Migrated*/>
      TranslationTable;

public:
  static bool IsTranslated(const std::string &APIName);
  static std::vector<std::string> GetAllAPINames(void);
};

#endif
