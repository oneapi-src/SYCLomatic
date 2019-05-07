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

class MapNames {
public:
  using MapTy = std::map<std::string, std::string>;
  static const MapTy TypeNamesMap;
  static const MapTy Dim3MemberNamesMap;
  static const MapTy MacrosMap;
  static const MapTy CublasFunctionNamesMap;

  inline static const std::string &findReplacedName(const MapTy &Map,
                                                    const std::string &Name) {
    static std::string NullString;
    auto Itr = Map.find(Name);
    if (Itr == Map.end())
      return NullString;
    return Itr->second;
  }
  static bool replaceName(const MapTy &Map, std::string &Name) {
    auto &Result = findReplacedName(Map, Name);
    if (Result.empty())
      return false;
    Name = Result;
    return true;
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
