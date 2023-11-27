//===--------------- MigrateCmakeScript.cpp--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "MigrateCmakeScript.h"
#include "SaveNewFiles.h"
#include "Statics.h"
#include "Utility.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>

using namespace clang::dpct;
using namespace llvm::cl;

namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

static std::string readFile(const clang::tooling::UnifiedPath &Name) {
  std::ifstream Stream(Name.getCanonicalPath().str(),
                       std::ios::in | std::ios::binary);
  std::string Contents((std::istreambuf_iterator<char>(Stream)),
                       (std::istreambuf_iterator<char>()));
  return Contents;
}

clang::tooling::UnifiedPath
getCmakeBuildPathFromInRoot(const clang::tooling::UnifiedPath &InRoot,
                            const clang::tooling::UnifiedPath &OutRoot) {
  std::error_code EC;

  clang::tooling::UnifiedPath CmakeBuildDirectory;
  for (fs::recursive_directory_iterator Iter(InRoot.getCanonicalPath(), EC),
       End;
       Iter != End; Iter.increment(EC)) {
    if ((bool)EC) {
      std::string ErrMsg =
          "[ERROR] Access : " + std::string(InRoot.getCanonicalPath()) +
          " fail: " + EC.message() + "\n";
      PrintMsg(ErrMsg);
    }

    clang::tooling::UnifiedPath FilePath(Iter->path());

    // Skip output directory if it is in the in-root directory.
    if (isChildOrSamePath(OutRoot, FilePath))
      continue;

    bool IsHidden = false;
    for (path::const_iterator PI = path::begin(FilePath.getCanonicalPath()),
                              PE = path::end(FilePath.getCanonicalPath());
         PI != PE; ++PI) {
      StringRef Comp = *PI;
      if (Comp.startswith(".")) {
        IsHidden = true;
        break;
      }
    }
    // Skip hidden folder or file whose name begins with ".".
    if (IsHidden) {
      continue;
    }

    if (Iter->type() == fs::file_type::directory_file) {
      const clang::tooling::UnifiedPath Path = Iter->path();
      if (fs::exists(appendPath(Path.getCanonicalPath().str(), "CMakeFiles")) &&
          fs::exists(
              appendPath(Path.getCanonicalPath().str(), "CMakeCache.txt"))) {
        CmakeBuildDirectory = Path;
        break;
      }
    }
  }
  return CmakeBuildDirectory;
}

void collectCmakeScripts(
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot,
    std::vector<clang::tooling::UnifiedPath> &CmakeScriptFiles) {
  std::error_code EC;

  clang::tooling::UnifiedPath CmakeBuildDirectory =
      getCmakeBuildPathFromInRoot(InRoot, OutRoot);
  for (fs::recursive_directory_iterator Iter(InRoot.getCanonicalPath(), EC),
       End;
       Iter != End; Iter.increment(EC)) {
    if ((bool)EC) {
      std::string ErrMsg =
          "[ERROR] Access : " + std::string(InRoot.getCanonicalPath()) +
          " fail: " + EC.message() + "\n";
      PrintMsg(ErrMsg);
    }

    clang::tooling::UnifiedPath FilePath(Iter->path());

    // Skip output directory if it is in the in-root directory.
    if (isChildOrSamePath(OutRoot, FilePath))
      continue;

    // Skip cmake build directory if it is in the in-root directory.
    if (!CmakeBuildDirectory.getPath().empty() &&
        isChildOrSamePath(CmakeBuildDirectory, FilePath))
      continue;

    bool IsHidden = false;
    for (path::const_iterator PI = path::begin(FilePath.getCanonicalPath()),
                              PE = path::end(FilePath.getCanonicalPath());
         PI != PE; ++PI) {
      StringRef Comp = *PI;
      if (Comp.startswith(".")) {
        IsHidden = true;
        break;
      }
    }
    // Skip hidden folder or file whose name begins with ".".
    if (IsHidden) {
      continue;
    }

    if (Iter->type() == fs::file_type::regular_file) {
      llvm::StringRef Name =
          llvm::sys::path::filename(FilePath.getCanonicalPath());
      if (Name == "CMakeLists.txt" || Name.ends_with(".cmake")) {
        CmakeScriptFiles.push_back(FilePath);
      }
    }
  }
}

bool migrateCmakeScriptFile(const clang::tooling::UnifiedPath &InRoot,
                            const clang::tooling::UnifiedPath &OutRoot,
                            const clang::tooling::UnifiedPath &InFileName) {
  clang::tooling::UnifiedPath OutFileName(InFileName);
  if (!rewriteDir(OutFileName, InRoot, OutRoot)) {
    return false;
  }
  auto Parent = path::parent_path(OutFileName.getCanonicalPath());
  std::error_code EC;
  EC = fs::create_directories(Parent);
  if ((bool)EC) {
    std::string ErrMsg = "[ERROR] Create Directory : " + Parent.str() +
                         " fail: " + EC.message() + "\n";
    PrintMsg(ErrMsg);
  }
  std::ofstream Out(OutFileName.getCanonicalPath().str(), std::ios::binary);
  if (Out.fail()) {
    std::string ErrMsg =
        "[ERROR] Create file : " + OutFileName.getCanonicalPath().str() +
        " failure!\n";
    PrintMsg(ErrMsg);
  }

  llvm::raw_os_ostream Stream(Out);
  applyPatternRewriterToCmakeScriptFile(readFile(InFileName), Stream);

  Stream.flush();
  Out.close();
  return true;
}

bool cmakeScriptFileSpecified(const std::vector<std::string> &SourceFiles) {
  bool IsCmakeScript = false;
  for (const auto &FilePath : SourceFiles) {
    if (!llvm::sys::path::has_extension(FilePath) ||
        llvm::sys::path::filename(FilePath).ends_with(".cmake") ||
        llvm::sys::path::filename(FilePath).ends_with(".txt"))
      IsCmakeScript = true;
    break;
  }
  return IsCmakeScript;
}

void migrateCmakeScriptOnly(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot) {
  auto CmakeScriptLists = OptParser->getSourcePathList();
  if (!CmakeScriptLists.empty()) {
    for (auto FilePath : CmakeScriptLists) {
      if (fs::is_directory(FilePath)) {
        std::vector<clang::tooling::UnifiedPath> CmakeScriptFiles;
        collectCmakeScripts(FilePath, OutRoot, CmakeScriptFiles);
        for (const auto &ScriptFile : CmakeScriptFiles) {
          if (!migrateCmakeScriptFile(InRoot, OutRoot, ScriptFile))
            continue;
        }
      } else {
        if (!migrateCmakeScriptFile(InRoot, OutRoot, FilePath))
          continue;
      }
    }
  } else {
    std::vector<clang::tooling::UnifiedPath> CmakeScriptFiles;
    collectCmakeScripts(InRoot, OutRoot, CmakeScriptFiles);
    for (const auto &ScriptFile : CmakeScriptFiles) {
      if (!migrateCmakeScriptFile(InRoot, OutRoot, ScriptFile))
        continue;
    }
  }
}
