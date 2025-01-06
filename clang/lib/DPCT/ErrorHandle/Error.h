//===--------------- Error.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_ERROR_H
#define DPCT_ERROR_H

#include "clang/Tooling/Tooling.h"
#include <string>

/// ProcessStatus defines various statuses of dpct workflow
enum ProcessStatus {
  MigrationSucceeded = 0,
  MigrationNoCodeChangeHappen = 1,
  MigrationSkipped = 2,
  CallIndependentToolSucceeded = 3,
  MigrationBuildScriptCompleted = 4,
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
  VcxprojPaserCreateCompilationDBFail = -12, /*eg. have no write permission*/
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
  CallIndependentToolError = -47,
  MigrationErrorBuildScriptPathInvalid = -48,
  MigrateBuildScriptOnlyNotSpecifed = -49,
  MigrateBuildScriptIncorrectUse = -50,
  MigrationErrorNoExplicitInRootAndBuildScript = -51,
  MigrationErrorCannotWrite = -52,
  MigratePythonBuildScriptOnlyNotSpecifed = -53,
  MigratePythonBuildScriptSpecifiedButPythonRuleFileNotSpecified = -54,
};

namespace clang {
namespace dpct {

void ShowStatus(int Status, std::string Message = "");
std::string getLoadYamlFailWarning(const clang::tooling::UnifiedPath& YamlPath);
std::string getCheckVersionFailWarning();
std::string getBuildScriptNotSpecifiedWarning();
std::string getPythonRuleFileNotProvidedWarning();

extern bool IsUsingDefaultOutRoot;
void removeDefaultOutRootFolder(const clang::tooling::UnifiedPath &DefaultOutRoot);
void dpctExit(int ExitCode, bool NeedCleanUp = true);


} // namespace dpct
} // namespace clang

#endif // DPCT_ERROR_H