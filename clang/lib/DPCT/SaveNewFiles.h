//===--- SaveNewFiles.h --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef DPCT_SAVE_NEW_FILES_H
#define DPCT_SAVE_NEW_FILES_H

#include "ValidateArguments.h"
#include "llvm/Support/Error.h"
#include <map>

#define DiagRef \
"See Diagnostics Reference to resolve warnings and complete the migration:\n"\
"https://software.intel.com/content/www/us/en/develop/documentation/"\
"intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference.html"

namespace llvm {
class StringRef;
}

namespace clang {
namespace tooling {
class RefactoringTool;
}
} // namespace clang

/// ProcessStatus defines various statuses of dpct workflow
enum ProcessStatus {
  MigrationSucceeded = 0,
  MigrationNoCodeChangeHappen = 1,
  MigrationSkipped = 2,
  MigrationSuccessExpParingOrRuntimeErr=3,
  MigrationError = -1,
  MigrationSaveOutFail = -2, /*eg. have no write permission*/
  MigrationErrorRunFromSDKFolder = -3,
  MigrationErrorInRootContainCTTool = -4,
  MigrationErrorInvalidSDKPath = -5,
  MigrationErrorInvalidInRootOrOutRoot = -6,
  MigrationErrorInvalidInRootPath = -7,
  MigrationErrorInvalidFilePath = -8,
  MigrationErrorInvalidReportArgs = -9,
  MigrationErrorNotSupportFileType = -10,
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
  MigrationErrorNameTooLong = -24,
  MigrationErrorPrefixTooLong = -25,
  MigrationErrorFormatFail = -26,
  MigrationErrorNoFileTypeAvail = -27,
};

/// Apply all generated replacements, and immediately save the results to
/// files in output directory.
///
/// \returns 0 upon success. Non-zero upon failure.
/// Prerequisite: InRoot and OutRoot are both absolute paths
int saveNewFiles(clang::tooling::RefactoringTool &Tool, llvm::StringRef InRoot,
                 llvm::StringRef OutRoot);

// std::string:  source file name including path.
// bool: false: the source file has no replacement.
//       true:  the source file has replacement.
extern std::map<std::string, bool> IncludeFileMap;
#endif // DPCT_SAVE_NEW_FILES_H
