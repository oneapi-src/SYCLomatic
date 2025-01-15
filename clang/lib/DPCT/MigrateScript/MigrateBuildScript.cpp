//===----------------------- MigrateBuildScript.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "MigrateScript/MigrateBuildScript.h"
#include "UserDefinedRules/PatternRewriter.h"
#include "Utility.h"

using namespace clang::dpct;

namespace fs = llvm::sys::fs;

namespace clang {
namespace dpct {
static std::vector<clang::tooling::UnifiedPath /*file path*/>
    CMakeBuildScriptFilesSet, PythonBuildScriptFilesSet;

bool cmakeScriptNotFound() { return CMakeBuildScriptFilesSet.empty(); }
bool pythonBuildScriptNotFound() { return PythonBuildScriptFilesSet.empty(); }

std::string readFile(const clang::tooling::UnifiedPath &Name) {
  std::ifstream Stream(Name.getCanonicalPath().str(),
                       std::ios::in | std::ios::binary);
  std::string Contents((std::istreambuf_iterator<char>(Stream)),
                       (std::istreambuf_iterator<char>()));
  return Contents;
}

std::vector<std::string> split(const std::string &Input,
                               const std::string &Delimiter) {
  std::vector<std::string> Vec;
  if (!Input.empty()) {

    size_t Index = 0;
    size_t Pos = Input.find(Delimiter, Index);
    while (Index < Input.size() && Pos != std::string::npos) {
      Vec.push_back(Input.substr(Index, Pos - Index));

      Index = Pos + Delimiter.size();
      Pos = Input.find(Delimiter, Index);
    }
    // Append the remaining part
    Vec.push_back(Input.substr(Index));
  }
  return Vec;
}

void storeBufferToFile(
    std::map<clang::tooling::UnifiedPath, std::string>
        &BuildScriptFileBufferMap,
    std::map<clang::tooling::UnifiedPath, bool> &ScriptFileCRLFMap) {
  for (auto &Entry : BuildScriptFileBufferMap) {
    auto &FileName = Entry.first;
    auto &Buffer = Entry.second;

    dpct::RawFDOStream Stream(FileName.getCanonicalPath().str());
    // Restore original endline format
    auto IsCRLF = ScriptFileCRLFMap[FileName];
    if (IsCRLF) {
      std::stringstream ResultStream;
      std::vector<std::string> SplitedStr = split(Buffer, '\n');
      for (auto &SS : SplitedStr) {
        ResultStream << SS << "\r\n";
      }
      Stream << llvm::StringRef(ResultStream.str().c_str());
    } else {
      Stream << llvm::StringRef(Buffer.c_str());
    }
    Stream.flush();
  }
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
      if (Comp.starts_with(".")) {
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
        CmakeBuildDirectory = std::move(Path);
        break;
      }
    }
  }
  return CmakeBuildDirectory;
}

void collectBuildScripts(const clang::tooling::UnifiedPath &InRoot,
                         const clang::tooling::UnifiedPath &OutRoot) {
  std::error_code EC;

  clang::tooling::UnifiedPath CmakeBuildDirectory;
  if (DpctGlobalInfo::migrateCMakeScripts()) {
    CmakeBuildDirectory = getCmakeBuildPathFromInRoot(InRoot, OutRoot);
  }
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

    if (DpctGlobalInfo::migrateCMakeScripts()) {
      // Skip cmake build directory if it is in the in-root directory.
      if (!CmakeBuildDirectory.getPath().empty() &&
          isChildOrSamePath(CmakeBuildDirectory, FilePath))
        continue;
    }

    bool IsHidden = false;
    for (path::const_iterator PI = path::begin(FilePath.getCanonicalPath()),
                              PE = path::end(FilePath.getCanonicalPath());
         PI != PE; ++PI) {
      StringRef Comp = *PI;
      if (Comp.starts_with(".")) {
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
      if (DpctGlobalInfo::migrateCMakeScripts()) {
#ifdef _WIN32
        if (Name.lower() == "cmakelists.txt" ||
            llvm::StringRef(Name.lower()).ends_with(".cmake")) {
#else
        if (Name == "CMakeLists.txt" || Name.ends_with(".cmake")) {
#endif
          CMakeBuildScriptFilesSet.push_back(FilePath.getCanonicalPath().str());
        }
      }
      if (DpctGlobalInfo::migratePythonScripts()) {
        if (Name.ends_with(".py")) {
          PythonBuildScriptFilesSet.push_back(
              FilePath.getCanonicalPath().str());
        }
      }
    }
  }
}

bool loadBufferFromScriptFile(const clang::tooling::UnifiedPath InRoot,
                              const clang::tooling::UnifiedPath OutRoot,
                              clang::tooling::UnifiedPath InFileName,
                              std::map<clang::tooling::UnifiedPath, std::string>
                                  &BuildScriptFileBufferMap) {
  clang::tooling::UnifiedPath OutFileName(InFileName);
  if (!rewriteCanonicalDir(OutFileName, InRoot, OutRoot)) {
    return false;
  }
  createDirectories(path::parent_path(OutFileName.getCanonicalPath()));
  BuildScriptFileBufferMap[OutFileName] = readFile(InFileName);
  return true;
}

bool buildScriptFileSpecified(const std::vector<std::string> &SourceFiles) {
  bool IsBuildScript = false;
  for (const auto &FilePath : SourceFiles) {
    if (!llvm::sys::path::has_extension(FilePath) ||
        llvm::sys::path::filename(FilePath).ends_with(".cmake") ||
        llvm::sys::path::filename(FilePath).ends_with(".txt") ||
        llvm::sys::path::filename(FilePath).ends_with(".py")) {
      IsBuildScript = true;
      break;
    }
  }
  return IsBuildScript;
}

void collectBuildScriptsSpecified(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot) {
  auto BuildScriptLists = OptParser->getSourcePathList();
  if (!BuildScriptLists.empty()) {
    for (auto &FilePath : BuildScriptLists) {
      if (fs::is_directory(FilePath)) {
        collectBuildScripts(FilePath, OutRoot);
      } else {
        if (DpctGlobalInfo::migrateCMakeScripts()) {
          if (llvm::sys::path::filename(FilePath).ends_with(".cmake") ||
              llvm::sys::path::filename(FilePath).ends_with(".txt")) {
            CMakeBuildScriptFilesSet.push_back(FilePath);
          }
        }
        if (DpctGlobalInfo::migratePythonScripts()) {
          if (llvm::sys::path::filename(FilePath).ends_with(".py")) {
            PythonBuildScriptFilesSet.push_back(FilePath);
          }
        }
      }
    }
  } else {
    collectBuildScripts(InRoot, OutRoot);
  }
}

void loadBufferFromFile(const clang::tooling::UnifiedPath &InRoot,
                        const clang::tooling::UnifiedPath &OutRoot,
                        std::map<clang::tooling::UnifiedPath, std::string>
                            &BuildScriptFileBufferMap,
                        bool isCMake) {
  std::vector<clang::tooling::UnifiedPath> &BuildScriptFilesSet =
      isCMake ? CMakeBuildScriptFilesSet : PythonBuildScriptFilesSet;

  for (const auto &ScriptFile : BuildScriptFilesSet) {
    if (!loadBufferFromScriptFile(InRoot, OutRoot, ScriptFile,
                                  BuildScriptFileBufferMap))
      continue;
  }
}

void unifyInputFileFormat(
    std::map<clang::tooling::UnifiedPath, std::string>
        &BuildScriptFileBufferMap,
    std::map<clang::tooling::UnifiedPath, bool> &ScriptFileCRLFMap) {
  for (auto &Entry : BuildScriptFileBufferMap) {
    auto &Buffer = Entry.second;
    const std::string FileName = Entry.first.getPath().str();

    // Convert input file to be LF
    bool IsCRLF = fixLineEndings(Buffer, Buffer);
    ScriptFileCRLFMap[Entry.first] = IsCRLF;
  }
}

} // namespace dpct
} // namespace clang
