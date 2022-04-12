//===--- Error.h --------------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef C2S_ERROR_H
#define C2S_ERROR_H

#include <string>

/// ProcessStatus defines various statuses of c2s workflow
enum ProcessStatus {
  MigrationSucceeded = 0,
  MigrationNoCodeChangeHappen = 1,
  MigrationSkipped = 2,
  MigrationError = -1,
  MigrationSaveOutFail = -2, /*eg. have no write permission*/
  MigrationErrorRunFromSDKFolder = -3,
  MigrationErrorInRootContainCTTool = -4,
  MigrationErrorInvalidCudaIncludePath = -5,
  MigrationErrorInvalidInRootOrOutRoot = -6,
  MigrationErrorInvalidInRootPath = -7,
  MigrationErrorInvalidFilePath = -8,
  MigrationErrorInvalidReportArgs = -9,
  VcxprojPaserFileNotExist = -11,
  VcxprojPaserCreateCompilationDBFail = -12, /*eg. hav no write permission*/
  MigrationErrorInvalidInstallPath = -13,
  MigrationErrorPathTooLong = -14,
  MigrationErrorInvalidWarningID = -15,
  MigrationOptionParsingError = -16,
  MigrationErrorFileParseError = -17,
  MigrationErrorShowHelp = -18,
  MigrationErrorCannotFindDatabase = -19,
  MigrationErrorCannotParseDatabase = -20,
  MigrationErrorNoExplicitInRoot = -21,
  MigrationSKIPForMissingCompileCommand = -22,
  MigrationErrorSpecialCharacter = -23,
  MigrationErrorPrefixTooLong = -25,
  MigrationErrorNoFileTypeAvail = -27,
  MigrationErrorInRootContainSDKFolder = -28,
  MigrationErrorCannotAccessDirInDatabase = -29,
  MigrationErrorInconsistentFileInDatabase = -30,
  MigrationErrorCudaVersionUnsupported = -31,
  MigrationErrorSupportedCudaVersionNotAvailable = -32,
  MigrationErrorInvalidExplicitNamespace = -33,
  MigrationErrorCustomHelperFileNameContainInvalidChar = -34,
  MigrationErrorCustomHelperFileNameTooLong = -35,
  MigrationErrorCustomHelperFileNamePathTooLong = -36,
  MigrationErrorDifferentOptSet = -37,
  MigrationErrorInvalidRuleFilePath = -38,
  MigrationErrorCannotParseRuleFile = -39,
};

namespace clang {
namespace c2s {

void ShowStatus(int Status, std::string Message = "");
std::string getLoadYamlFailWarning(std::string YamlPath);
std::string getCheckVersionFailWarning();
} // namespace c2s
} // namespace clang

#endif // C2S_ERROR_H