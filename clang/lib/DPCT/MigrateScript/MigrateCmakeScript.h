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
#include "UserDefinedRules/UserDefinedRules.h"

namespace clang {
namespace dpct {
void doCmakeScriptMigration(const clang::tooling::UnifiedPath &InRoot,
                            const clang::tooling::UnifiedPath &OutRoot);
void registerCmakeMigrationRule(MetaRuleObject &R);
void addCmakeWarningMsg(const std::string &WarningMsg,
                        const std::string FileName);

} // namespace dpct
} // namespace clang
#endif //DPCT_MIGRATE_CMAKE_SCRIPT_H
