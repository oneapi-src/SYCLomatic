//===-------------------- MigratePythonSetupScript.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "MigratePythonSetupScript.h"
#include "Diagnostics.h"
#include "Error.h"
#include "PatternRewriter.h"
#include "SaveNewFiles.h"
#include "Statics.h"
#include "Utility.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>

using namespace clang::dpct;
using namespace llvm::cl;

namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

static std::vector<clang::tooling::UnifiedPath /*file path*/>
    PythonSetupScriptFilesSet;
static std::map<clang::tooling::UnifiedPath /*file path*/,
                std::string /*content*/>
    PythonSetupScriptFileBufferMap;
static std::map<clang::tooling::UnifiedPath /*file name*/, bool /*is crlf*/>
    ScriptFileCRLFMap;

static std::map<std::string /*Python setup syntax*/,
                MetaRuleObject::PatternRewriter /*Python setup migraiton rule*/>
    PythonSetupBuildInRules;

static std::map<std::string /*file path*/,
                std::vector<std::string> /*warning msg*/>
    FileWarningsMap;

void PythonSetupSyntaxProcessed(std::string &Input);

static std::string readFile(const clang::tooling::UnifiedPath &Name) {
  std::ifstream Stream(Name.getCanonicalPath().str(),
                       std::ios::in | std::ios::binary);
  std::string Contents((std::istreambuf_iterator<char>(Stream)),
                       (std::istreambuf_iterator<char>()));
  return Contents;
}

void collectPythonSetupScripts(const clang::tooling::UnifiedPath &InRoot,
                         const clang::tooling::UnifiedPath &OutRoot) {
  std::error_code EC;

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

    if (Iter->type() == fs::file_type::regular_file) {
      llvm::StringRef Name =
          llvm::sys::path::filename(FilePath.getCanonicalPath());
      if (Name.ends_with(".py")) {
        PythonSetupScriptFilesSet.push_back(FilePath.getCanonicalPath().str());
      }
    }
  }
}

void collectPythonSetupScriptsSpecified(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot) {
  auto PythonSetupScriptLists = OptParser->getSourcePathList();
  
  if (!PythonSetupScriptLists.empty()) {
    for (auto &FilePath : PythonSetupScriptLists) {
      if (fs::is_directory(FilePath)) {
        collectPythonSetupScripts(FilePath, OutRoot);
      } else {
        if (llvm::sys::path::filename(FilePath).ends_with(".py")) {
          PythonSetupScriptFilesSet.push_back(FilePath);
        }
      }
    }
  } else {
    collectPythonSetupScripts(InRoot, OutRoot);
  }
}

static size_t skipWhiteSpaces(const std::string Input, size_t Index) {
  size_t Size = Input.size();
  for (; Index < Size && isWhitespace(Input[Index]); Index++) {
  }
  return Index;
}

static std::vector<std::string> split(const std::string &Input,
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

static void unifyInputFileFormat() {
  for (auto &Entry : PythonSetupScriptFileBufferMap) {
    auto &Buffer = Entry.second;
    const std::string FileName = Entry.first.getPath().str();

    // Convert input file to be LF
    bool IsCRLF = fixLineEndings(Buffer, Buffer);
    ScriptFileCRLFMap[Entry.first] = IsCRLF;
  }
}

static void
applyPythonSetupMigrationRules(const clang::tooling::UnifiedPath InRoot,
                         const clang::tooling::UnifiedPath OutRoot) {

  setFileTypeProcessed(SourceFileType::SFT_PySetupScript);

  for (auto &Entry : PythonSetupScriptFileBufferMap) {
    llvm::outs() << "Processing: " + Entry.first.getPath() + "\n";

    auto &Buffer = Entry.second;
    clang::tooling::UnifiedPath FileName = Entry.first.getPath();
    auto Iter = FileWarningsMap.find(FileName.getPath().str());
    if (Iter != FileWarningsMap.end()) {
      std::vector WarningsVec = Iter->second;
      for (auto &Warning : WarningsVec) {
        llvm::outs() << Warning;
      }
    }

    // Apply user define migration rules
    for (const auto &PythonSetupSyntaxEntry : PythonSetupBuildInRules) {
      const auto &PR = PythonSetupSyntaxEntry.second;
      if (PR.In.empty() && PR.Out.empty()) {
        // ...
      } else {
        Buffer = applyPatternRewriter(PR, Buffer);
      }
    }
  }
}

bool loadBufferFromPythonSetupScriptFile(const clang::tooling::UnifiedPath InRoot,
                              const clang::tooling::UnifiedPath OutRoot,
                              clang::tooling::UnifiedPath InFileName) {
  clang::tooling::UnifiedPath OutFileName(InFileName);
  if (!rewriteDir(OutFileName, InRoot, OutRoot)) {
    return false;
  }
  createDirectories(path::parent_path(OutFileName.getCanonicalPath()));
  PythonSetupScriptFileBufferMap[OutFileName] = readFile(InFileName);
  return true;
}

static void loadBufferFromFile(const clang::tooling::UnifiedPath &InRoot,
                               const clang::tooling::UnifiedPath &OutRoot) {
  for (const auto &ScriptFile : PythonSetupScriptFilesSet) {
    if (!loadBufferFromPythonSetupScriptFile(InRoot, OutRoot, ScriptFile))
      continue;
  }
}

bool pythonSetupScriptFileSpecified(const std::vector<std::string> &SourceFiles) {
  bool IsPythonSetupScript = false;
  for (const auto &FilePath : SourceFiles) {
    if (!llvm::sys::path::has_extension(FilePath) ||
        llvm::sys::path::filename(FilePath).ends_with(".py")) {
      IsPythonSetupScript = true;
      break;
    }
  }
  return IsPythonSetupScript;
}

bool pythonSetupScriptNotFound() { return PythonSetupScriptFilesSet.empty(); }

static void storeBufferToFile() {
  for (auto &Entry : PythonSetupScriptFileBufferMap) {
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

void doPythonSetupScriptMigration(const clang::tooling::UnifiedPath &InRoot,
                            const clang::tooling::UnifiedPath &OutRoot) {
  loadBufferFromFile(InRoot, OutRoot);
  unifyInputFileFormat();
  applyPythonSetupMigrationRules(InRoot, OutRoot);
  storeBufferToFile();
}

void registerPythonSetupMigrationRule(MetaRuleObject &R) {
  auto PR =
      MetaRuleObject::PatternRewriter(R.In, R.Out, R.Subrules, R.MatchMode,
                                      R.RuleId, R.Priority, "",
                                      R.PySetupSyntax);

  auto Iter = PythonSetupBuildInRules.find(PR.PySetupSyntax);
  if (Iter != PythonSetupBuildInRules.end()) {
    if (PR.Priority == RulePriority::Takeover &&
        Iter->second.Priority > PR.Priority) {
      PythonSetupBuildInRules[PR.PySetupSyntax] = PR;
    } else {
      llvm::outs() << "[Warnning]: Two migration rules (Rule:" << R.RuleId
                   << ", Rule:" << Iter->second.RuleId
                   << ") are duplicated, the migrtion rule (Rule:" << R.RuleId
                   << ") is ignored.\n";
    }
  } else {
    PythonSetupBuildInRules[PR.PySetupSyntax] = PR;
  }
}
