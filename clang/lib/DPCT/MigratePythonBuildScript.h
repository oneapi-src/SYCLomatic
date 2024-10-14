//===--------------- MigrateCmakeScript.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_MIGRATE_PYTHON_BUILD_SCRIPT_H
#define DPCT_MIGRATE_PYTHON_BUILD_SCRIPT_H
#include "MigrateBuildScript.h"
#include "Rules.h"

void collectPythonBuildScripts(const clang::tooling::UnifiedPath &InRoot,
                               const clang::tooling::UnifiedPath &OutRoot);
void collectPythonBuildScriptsSpecified(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot);
void doPythonBuildScriptMigration(const clang::tooling::UnifiedPath &InRoot,
                                  const clang::tooling::UnifiedPath &OutRoot);
void registerPythonMigrationRule(MetaRuleObject &R);
bool pythonBuildScriptNotFound();
void addPythonWarningMsg(const std::string &WarningMsg,
                              const std::string FileName);
#endif
