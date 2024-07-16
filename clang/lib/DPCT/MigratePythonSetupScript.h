//===--------------- MigrateCmakeScript.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_MIGRATE_PYTHON_SETUP_SCRIPT_H
#define DPCT_MIGRATE_PYTHON_SETUP_SCRIPT_H
#include "Rules.h"

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Core/UnifiedPath.h"

#include <map>

void collectPythonSetupScripts(const clang::tooling::UnifiedPath &InRoot,
                               const clang::tooling::UnifiedPath &OutRoot);
void collectPythonSetupScriptsSpecified(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot);

void doPythonSetupScriptMigration(const clang::tooling::UnifiedPath &InRoot,
                                  const clang::tooling::UnifiedPath &OutRoot);
bool pythonSetupScriptFileSpecified(
    const std::vector<std::string> &SourceFiles);

void registerPythonSetupMigrationRule(MetaRuleObject &R);
bool pythonSetupScriptNotFound();
#endif
