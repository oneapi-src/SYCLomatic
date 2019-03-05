//===--- SaveNewFiles.h --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef SYCLCT_SAVE_NEW_FILES_H
#define SYCLCT_SAVE_NEW_FILES_H

#include <map>

namespace llvm {
class StringRef;
}

namespace clang {
namespace tooling {
class RefactoringTool;
}
} // namespace clang

enum ProcessStatus {
  MigrationSucceeded = 0,
  MigrationNoCodeChangeHappen = 1,
  MigrationSkipped = 2,
  MigrationError = -1,
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
#endif // SYCLCT_SAVE_NEW_FILES_H
