//===--------------- Error.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_ERROR_H
#define DPCT_ERROR_H

#include <string>

/// ProcessStatus defines various statuses of dpct workflow
enum ProcessStatus {
  MigrationSucceeded = 0,
  MigrationNoCodeChangeHappen = 1,
  MigrationSkipped = 2,
  MigrationError = -1,
  MigrationSaveOutFail = -2, /*eg. have no write permission*/
  MigrationErrorRunFromSDKFolder = -3,
  MigrationErrorInputDirContainCTTool = -4,
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
  MigrationErrorInputDirContainSDKFolder = -28,
  MigrationErrorCannotAccessDirInDatabase = -29,
  MigrationErrorInconsistentFileInDatabase = -30,
  MigrationErrorCudaVersionUnsupported = -31,
  MigrationErrorDetectedCudaVersionUnsupported = -32,
  MigrationErrorInvalidExplicitNamespace = -33,
  MigrationErrorDifferentOptSet = -37,
  MigrationErrorInvalidRuleFilePath = -38,
  MigrationErrorCannotParseRuleFile = -39,
  MigrationErrorInvalidAnalysisScope = -40,
  MigrationErrorInvalidChangeFilenameExtension = -41,
  MigrationErrorConflictOptions = -42,
  MigrationErrorNoAPIMapping = -43,
  MigrationErrorAPIMappingWrongCUDAHeader = -44,
  MigrationErrorAPIMappingNoCUDAHeader = -45,
  MigrationErrorCannotDetectCudaPath = -46,
};

namespace clang {
namespace dpct {

void ShowStatus(int Status, std::string Message = "");
std::string getLoadYamlFailWarning(std::string YamlPath);
std::string getCheckVersionFailWarning();
} // namespace dpct
} // namespace clang

#endif // DPCT_ERROR_H