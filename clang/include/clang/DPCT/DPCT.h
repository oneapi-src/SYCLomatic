//===--------------- DPCT.h -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <string>
#include <vector>

extern bool HasSDKIncludeOption;
extern std::string RealSDKIncludePath;
extern std::vector<std::string> ExtraIncPaths;
extern bool HasSDKPathOption;
extern std::string RealSDKPath;
extern int SDKVersionMajor;
extern int SDKVersionMinor;
int run(int argc, const char **argv);
