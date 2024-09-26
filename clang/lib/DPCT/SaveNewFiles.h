//===--------------- SaveNewFiles.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_SAVE_NEW_FILES_H
#define DPCT_SAVE_NEW_FILES_H

#include "clang/Tooling/Refactoring.h"

#include <map>

using ReplTy = std::map<std::string, clang::tooling::Replacements>;

#define DiagRef                                                                \
  "See Diagnostics Reference to resolve warnings and complete the "            \
  "migration:\n"                                                               \
  "https://oneapi-src.github.io/SYCLomatic/dev_guide/reference/"               \
  "diagnostics-reference.html\n"

namespace llvm {
class StringRef;
} // namespace llvm

/// Apply all generated replacements, and immediately save the results to
/// files in output directory.
///
/// \returns 0 upon success. Non-zero upon failure.
/// Prerequisite: InRoot and OutRoot are both absolute paths
int saveNewFiles(clang::tooling::RefactoringTool &Tool,
                 clang::tooling::UnifiedPath InRoot,
                 clang::tooling::UnifiedPath OutRoot,
                 clang::tooling::UnifiedPath CUDAMigratedOutRoot,
                 ReplTy &ReplCUDA, ReplTy &ReplSYCL);

void loadYAMLIntoFileInfo(clang::tooling::UnifiedPath Path);

// clang::tooling::UnifiedPath:  source file name including path.
// bool: false: the source file has no replacement.
//       true:  the source file has replacement.
extern std::map<clang::tooling::UnifiedPath, bool> IncludeFileMap;

// This function is registered by SetFileProcessHandle() called by runDPCT() in
// DPCT.cpp, and called in Tooling.cpp::DoFileProcessHandle(). It traverses all
// the files in directory \pInRoot, collecting *.cu files not
// processed by the the first loop of calling processFiles() in
// Tooling.cpp::ClangTool::run()) into \pFilesNotProcessed, and copies the rest
// files to the output directory.
void processAllFiles(llvm::StringRef InRoot, llvm::StringRef OutRoot,
                     std::vector<std::string> &FilesNotProcessed);

/// Replace file path specified by \pInRoot with \pOutRoot in \pFilePath.
///
/// \returns true if file path is rewritten, false otherwise.
bool rewriteCanonicalDir(clang::tooling::UnifiedPath &FilePath,
                const clang::tooling::UnifiedPath &InRoot,
                const clang::tooling::UnifiedPath &OutRoot);
bool rewriteAbsoluteDir(clang::tooling::UnifiedPath &FilePath,
                const clang::tooling::UnifiedPath &InRoot,
                const clang::tooling::UnifiedPath &OutRoot);
// Replace file name \p FileName with new migrated name. For c source files, the
// file extension needs to wait until all replacements are generated to
// get the correct result.
void rewriteFileName(clang::tooling::UnifiedPath &FileName);
// Replace file name \p FileName with new migrated name.
// This overloaded function is added because in some cases, the \p FileName is
// relative path, and absolute path \p FullPathName is needed to determine
// whether the file is in database. For c source files, the file
// extension needs to wait until all replacements are generated to get the
// correct result.
void rewriteFileName(clang::tooling::UnifiedPath &FileName,
                     const clang::tooling::UnifiedPath &FullPathName);

// Replace file name \p FileName with new migrated name.
void rewriteFileName(std::string &FileName, const std::string &FullPathName);

// A mapping from output file path to it's corresponding input file.
extern std::map<std::string, std::string> OutFilePath2InFilePath;
#endif // DPCT_SAVE_NEW_FILES_H
