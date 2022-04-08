//===--- C2S.h -------------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#include <string>

extern bool HasSDKIncludeOption;
extern std::string RealSDKIncludePath;
extern bool HasSDKPathOption;
extern std::string RealSDKPath;
extern int SDKVersionMajor;
extern int SDKVersionMinor;
int run(int argc, const char **argv);
