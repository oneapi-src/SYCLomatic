//===-------------------- MigratePythonBuildScript.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "MigratePythonBuildScript.h"
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
    PythonBuildScriptFilesSet;
static std::map<clang::tooling::UnifiedPath /*file path*/,
                std::string /*content*/>
    PythonBuildScriptFileBufferMap;
static std::map<clang::tooling::UnifiedPath /*file name*/, bool /*is crlf*/>
    ScriptFileCRLFMap;

static std::map<std::string /*Python syntax*/,
                MetaRuleObject::PatternRewriter /*Python migration rule*/>
    PythonBuildInRules;

static std::map<std::string /*file path*/,
                std::vector<std::string> /*warning msg*/>
    FileWarningsMap;

void collectPythonBuildScripts(const clang::tooling::UnifiedPath &InRoot,
                               const clang::tooling::UnifiedPath &OutRoot) {
  collectBuildScripts(InRoot, OutRoot, PythonBuildScriptFilesSet,
                      BuildScriptKind::BS_Python);
}

void collectPythonBuildScriptsSpecified(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot) {
  collectBuildScriptsSpecified(OptParser, InRoot, OutRoot,
                               PythonBuildScriptFilesSet,
                               BuildScriptKind::BS_Python);
}

void addPythonWarningMsg(const std::string &WarningMsg,
                         const std::string FileName) {
  FileWarningsMap[FileName].push_back(WarningMsg);
}

static void
applyPythonMigrationRules(const clang::tooling::UnifiedPath InRoot,
                          const clang::tooling::UnifiedPath OutRoot) {

  setFileTypeProcessed(SourceFileType::SFT_PySetupScript);

  for (auto &Entry : PythonBuildScriptFileBufferMap) {
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
    for (const auto &PythonSyntaxEntry : PythonBuildInRules) {
      const auto &PR = PythonSyntaxEntry.second;
      if (!PR.In.empty() || !PR.Out.empty()) {
        Buffer = applyPatternRewriter(PR, Buffer);
      }
    }
  }
}

bool pythonBuildScriptNotFound() { return PythonBuildScriptFilesSet.empty(); }

void doPythonBuildScriptMigration(const clang::tooling::UnifiedPath &InRoot,
                                  const clang::tooling::UnifiedPath &OutRoot) {
  loadBufferFromFile(InRoot, OutRoot, PythonBuildScriptFilesSet,
                     PythonBuildScriptFileBufferMap);
  unifyInputFileFormat(PythonBuildScriptFileBufferMap, ScriptFileCRLFMap);
  applyPythonMigrationRules(InRoot, OutRoot);
  storeBufferToFile(PythonBuildScriptFileBufferMap, ScriptFileCRLFMap);
}

void registerPythonMigrationRule(MetaRuleObject &R) {
  auto PR = MetaRuleObject::PatternRewriter(R.In, R.Out, R.Subrules,
                                            R.MatchMode, R.Warning, R.RuleId,
                                            R.BuildScriptSyntax, R.Priority);

  auto Iter = PythonBuildInRules.find(PR.BuildScriptSyntax);
  if (Iter != PythonBuildInRules.end()) {
    if (PR.Priority == RulePriority::Takeover &&
        Iter->second.Priority > PR.Priority) {
      PythonBuildInRules[PR.BuildScriptSyntax] = PR;
    } else {
      llvm::outs() << "[Warnning]: Two migration rules (Rule:" << R.RuleId
                   << ", Rule:" << Iter->second.RuleId
                   << ") are duplicated, the migrtion rule (Rule:" << R.RuleId
                   << ") is ignored.\n";
    }
  } else {
    PythonBuildInRules[PR.BuildScriptSyntax] = PR;
  }
}
