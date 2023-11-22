//===--------------- SaveNewFiles.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_SAVE_NEW_FILES_H
#define DPCT_SAVE_NEW_FILES_H

#include "ValidateArguments.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"
#include <map>

#define DiagRef                                                                \
  "See Diagnostics Reference to resolve warnings and complete the "            \
  "migration:\n"                                                               \
  "https://www.intel.com/content/www/us/en/docs/dpcpp-compatibility-tool/"     \
  "developer-guide-reference/current/diagnostics-reference.html\n"

namespace llvm {
class StringRef;
}

namespace clang {
namespace tooling {
class RefactoringTool;
}
} // namespace clang

/// Apply all generated replacements, and immediately save the results to
/// files in output directory.
///
/// \returns 0 upon success. Non-zero upon failure.
/// Prerequisite: InRoot and OutRoot are both absolute paths
int saveNewFiles(clang::tooling::RefactoringTool &Tool, llvm::StringRef InRoot,
                 llvm::StringRef OutRoot);

void loadYAMLIntoFileInfo(std::string Path);

// std::string:  source file name including path.
// bool: false: the source file has no replacement.
//       true:  the source file has replacement.
extern std::map<std::string, bool> IncludeFileMap;

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
bool rewriteDir(llvm::SmallString<512> &FilePath, const llvm::StringRef InRoot,
                const llvm::StringRef OutRoot);

// Replace file name \p FileName with new migrated name.
void rewriteFileName(llvm::SmallString<512> &FileName);
// Replace file name \p FileName with new migrated name.
// This overloaded function is added because in some cases, the \p FileName is
// relative path, and absolute path \p FullPathName is needed to determine
// whether the file is in database.
void rewriteFileName(llvm::SmallString<512> &FileName,
                     llvm::StringRef FullPathName);

// apply patter rewrite rules to migrate cmake script file
void applyPatternRewriterToCmakeScriptFile(const std::string &InputString,
                                           llvm::raw_os_ostream &Stream);
#endif // DPCT_SAVE_NEW_FILES_H
