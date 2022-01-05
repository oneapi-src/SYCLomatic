//===--- os_specific.h-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#pragma once
#include "GaNamespace.h"
#include <vector>
#include <string>
GAHELPER_NS_BEGIN

std::string getOsName();


//adds discovered proxy (different detection methods can deliver more than one proxy) to result vector
void collectProxyInfo(const char* for_url, std::vector<std::string>* result);

GAHELPER_NS_END
