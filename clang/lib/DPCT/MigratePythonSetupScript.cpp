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

void collectPythonSetupScripts(const clang::tooling::UnifiedPath &InRoot,
                               const clang::tooling::UnifiedPath &OutRoot) {
  collectBuildScripts(InRoot, OutRoot, PythonSetupScriptFilesSet,
                      BuildScriptKind::BS_PySetup);
}

void collectPythonSetupScriptsSpecified(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot) {
  collectBuildScriptsSpecified(OptParser, InRoot, OutRoot,
                               PythonSetupScriptFilesSet,
                               BuildScriptKind::BS_PySetup);
}

void addPythonSetupWarningMsg(const std::string &WarningMsg,
                              const std::string FileName) {
  FileWarningsMap[FileName].push_back(WarningMsg);
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
      if (!PR.In.empty() || !PR.Out.empty()) {
        Buffer = applyPatternRewriter(PR, Buffer);
      }
    }
  }
}

bool pythonSetupScriptNotFound() { return PythonSetupScriptFilesSet.empty(); }

void doPythonSetupScriptMigration(const clang::tooling::UnifiedPath &InRoot,
                                  const clang::tooling::UnifiedPath &OutRoot) {
  loadBufferFromFile(InRoot, OutRoot, PythonSetupScriptFilesSet,
                     PythonSetupScriptFileBufferMap);
  unifyInputFileFormat(PythonSetupScriptFileBufferMap, ScriptFileCRLFMap);
  applyPythonSetupMigrationRules(InRoot, OutRoot);
  storeBufferToFile(PythonSetupScriptFileBufferMap, ScriptFileCRLFMap);
}

void registerPythonSetupMigrationRule(MetaRuleObject &R) {
  auto PR = MetaRuleObject::PatternRewriter(R.In, R.Out, R.Subrules,
                                            R.MatchMode, R.Warning, R.RuleId,
                                            R.BuildScriptSyntax, R.Priority);

  auto Iter = PythonSetupBuildInRules.find(PR.BuildScriptSyntax);
  if (Iter != PythonSetupBuildInRules.end()) {
    if (PR.Priority == RulePriority::Takeover &&
        Iter->second.Priority > PR.Priority) {
      PythonSetupBuildInRules[PR.BuildScriptSyntax] = PR;
    } else {
      llvm::outs() << "[Warnning]: Two migration rules (Rule:" << R.RuleId
                   << ", Rule:" << Iter->second.RuleId
                   << ") are duplicated, the migrtion rule (Rule:" << R.RuleId
                   << ") is ignored.\n";
    }
  } else {
    PythonSetupBuildInRules[PR.BuildScriptSyntax] = PR;
  }
}
