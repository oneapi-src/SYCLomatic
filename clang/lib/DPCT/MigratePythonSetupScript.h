//===--------------- MigrateCmakeScript.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_MIGRATE_PYTHON_SETUP_SCRIPT_H
#define DPCT_MIGRATE_PYTHON_SETUP_SCRIPT_H
#include "MigrateBuildScript.h"
#include "Rules.h"

void collectPythonSetupScripts(const clang::tooling::UnifiedPath &InRoot,
                               const clang::tooling::UnifiedPath &OutRoot);
void collectPythonSetupScriptsSpecified(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot);
void doPythonSetupScriptMigration(const clang::tooling::UnifiedPath &InRoot,
                                  const clang::tooling::UnifiedPath &OutRoot);
void registerPythonSetupMigrationRule(MetaRuleObject &R);
bool pythonSetupScriptNotFound();
void addPythonSetupWarningMsg(const std::string &WarningMsg,
                              const std::string FileName);
#endif
