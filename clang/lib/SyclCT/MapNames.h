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

#include <map>

class MapNames {
public:
  static const std::map<std::string, std::string> TypeNamesMap;
  static const std::map<std::string, std::string> Dim3MemberNamesMap;
  static const std::map<std::string, std::string> Dim3MemberPointerNamesMap;
};

#endif
