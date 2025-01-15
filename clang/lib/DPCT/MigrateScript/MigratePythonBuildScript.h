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
#include "UserDefinedRules/UserDefinedRules.h"

namespace clang {
namespace dpct {
void doPythonBuildScriptMigration(const clang::tooling::UnifiedPath &InRoot,
                                  const clang::tooling::UnifiedPath &OutRoot);
void registerPythonMigrationRule(MetaRuleObject &R);
bool pythonMigrationRulesRegistered();
void addPythonWarningMsg(const std::string &WarningMsg,
                         const std::string FileName);
} // namespace dpct
} // namespace clang
#endif //DPCT_MIGRATE_PYTHON_BUILD_SCRIPT_H
