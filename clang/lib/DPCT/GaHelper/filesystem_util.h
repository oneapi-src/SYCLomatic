//===--- filesystem_util.h-------------------------*- C++ -*---===//
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
#include <string>
#include "GaNamespace.h"

#ifdef _WIN32
typedef std::wstring ustring;
#define _U(x) L##x
#include <Windows.h>
#else
typedef std::string ustring;
#define _U(x) x
#endif

GAHELPER_NS_BEGIN

ustring dirname(const ustring& path);
bool createDirectories(const ustring& path);

GAHELPER_NS_END
