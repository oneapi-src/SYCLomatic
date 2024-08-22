//===--------------- MigrateCmakeScript.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_MIGRATE_CMAKE_SCRIPT_H
#define DPCT_MIGRATE_CMAKE_SCRIPT_H
#include "MigrateBuildScript.h"
#include "Rules.h"

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Core/UnifiedPath.h"

#include <map>

clang::tooling::UnifiedPath
getCmakeBuildPathFromInRoot(const clang::tooling::UnifiedPath &InRoot,
                            const clang::tooling::UnifiedPath &OutRoot);
void collectCmakeScripts(const clang::tooling::UnifiedPath &InRoot,
                         const clang::tooling::UnifiedPath &OutRoot);
void collectCmakeScriptsSpecified(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot);

void doCmakeScriptMigration(const clang::tooling::UnifiedPath &InRoot,
                            const clang::tooling::UnifiedPath &OutRoot);
bool cmakeScriptFileSpecified(const std::vector<std::string> &SourceFiles);

void registerCmakeMigrationRule(MetaRuleObject &R);
bool cmakeScriptNotFound();
void addWarningMsg(const std::string &WarningMsg, const std::string FileName);
#endif
