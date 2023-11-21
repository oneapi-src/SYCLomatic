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

static std::string readFile(const std::string &Name) {
  std::ifstream Stream(Name, std::ios::in | std::ios::binary);
  std::string Contents((std::istreambuf_iterator<char>(Stream)),
                       (std::istreambuf_iterator<char>()));
  return Contents;
}

std::string getCmakeBuildPathFromInRoot(StringRef InRoot, StringRef OutRoot) {
  std::error_code EC;

  std::string CmakeBuildDirectory;
  for (fs::recursive_directory_iterator Iter(Twine(InRoot), EC), End;
       Iter != End; Iter.increment(EC)) {
    if ((bool)EC) {
      std::string ErrMsg = "[ERROR] Access : " + std::string(InRoot.str()) +
                           " fail: " + EC.message() + "\n";
      PrintMsg(ErrMsg);
    }

    auto FilePath = Iter->path();

    // Skip output directory if it is in the in-root directory.
    if (isChildOrSamePath(OutRoot.str(), FilePath))
      continue;

    bool IsHidden = false;
    for (path::const_iterator PI = path::begin(FilePath),
                              PE = path::end(FilePath);
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
      const auto Path = Iter->path();
      SmallString<512> OutDirectory = llvm::StringRef(Path);
      if (fs::exists(OutDirectory + "/CMakeFiles") &&
          fs::exists(OutDirectory + "/CMakeCache.txt")) {
        CmakeBuildDirectory = OutDirectory.str().str();
        break;
      }
    }
  }
  return CmakeBuildDirectory;
}

void collectCmakeScripts(StringRef InRoot, StringRef OutRoot,
                         std::vector<std::string> &CmakeScriptFiles) {
  std::error_code EC;

  std::string CmakeBuildDirectory =
      getCmakeBuildPathFromInRoot(InRoot, OutRoot);
  for (fs::recursive_directory_iterator Iter(Twine(InRoot), EC), End;
       Iter != End; Iter.increment(EC)) {
    if ((bool)EC) {
      std::string ErrMsg = "[ERROR] Access : " + std::string(InRoot.str()) +
                           " fail: " + EC.message() + "\n";
      PrintMsg(ErrMsg);
    }

    auto FilePath = Iter->path();

    // Skip output directory if it is in the in-root directory.
    if (isChildOrSamePath(OutRoot.str(), FilePath))
      continue;

    // Skip cmake build directory if it is in the in-root directory.
    if (!CmakeBuildDirectory.empty() &&
        isChildOrSamePath(CmakeBuildDirectory, FilePath))
      continue;

    bool IsHidden = false;
    for (path::const_iterator PI = path::begin(FilePath),
                              PE = path::end(FilePath);
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
      SmallString<512> OutputFile = llvm::StringRef(FilePath);

      llvm::StringRef Name = llvm::sys::path::filename(FilePath);
      if (Name == "CMakeLists.txt" || Name.ends_with(".cmake")) {
        CmakeScriptFiles.push_back(OutputFile.str().str());
      }
    }
  }
}

bool migrateCmakeScriptFile(StringRef InRoot, StringRef OutRoot,
                            std::string InFileName) {
  makeCanonical(InFileName);
  SmallString<512> OutFileName = llvm::StringRef(InFileName);
  if (!rewriteDir(OutFileName, InRoot, OutRoot)) {
    return false;
  }
  auto Parent = path::parent_path(OutFileName);
  std::error_code EC;
  EC = fs::create_directories(Parent);
  if ((bool)EC) {
    std::string ErrMsg =
        "[ERROR] Create Directory : " + std::string(Parent.str()) +
        " fail: " + EC.message() + "\n";
    PrintMsg(ErrMsg);
  }
  std::ofstream Out(OutFileName.c_str(), std::ios::binary);
  if (Out.fail()) {
    std::string ErrMsg =
        "[ERROR] Create file : " + std::string(OutFileName.c_str()) +
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
    StringRef InRoot, StringRef OutRoot) {

  auto CmakeScriptLists = OptParser->getSourcePathList();
  if (!CmakeScriptLists.empty()) {
    for (auto FilePath : CmakeScriptLists) {
      if (fs::is_directory(FilePath)) {
        std::vector<std::string> CmakeScriptFiles;
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
    std::vector<std::string> CmakeScriptFiles;
    collectCmakeScripts(InRoot, OutRoot, CmakeScriptFiles);
    for (const auto &ScriptFile : CmakeScriptFiles) {
      if (!migrateCmakeScriptFile(InRoot, OutRoot, ScriptFile))
        continue;
    }
  }
}
