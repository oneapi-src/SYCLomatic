//===--- ValidateArguments.h ---------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef DPCT_VALIDATE_ARGUMENTS_H
#define DPCT_VALIDATE_ARGUMENTS_H

#include <string>
#include <vector>

enum ReportFormatEnum { notsetformat, csv, formatted };
enum ReportTypeEnum { notsettype, apis, stats, all, diags };

bool makeCanonicalOrSetDefaults(std::string &InRoot, std::string &OutRoot,
                                const std::vector<std::string> SourceFiles);

/// Make sure files passed to Intel(R) DPC++ Compatibility Tool are under the
/// input root directory and have an extension.
bool validatePaths(const std::string &InRoot,
                   const std::vector<std::string> &SourceFiles);
bool checkReportArgs(ReportTypeEnum &RType, ReportFormatEnum &RFormat,
        std::string &RFile, bool& ROnly, bool &GenReport, std::string &DVerbose);

/// Retrun value:
///  0: Path is valid
///  1: Path is empty, option SDK include path is not used
/// -1: Path is invaild
int checkSDKPathOrIncludePath(const std::string &Path, std::string &RealPath);
#endif // DPCT_VALIDATE_ARGUMENTS_H
