//===--------------- MigrateCmakeScript.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_MIGRATE_CMAKE_SCRIPT_H
#define DPCT_MIGRATE_CMAKE_SCRIPT_H

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Core/UnifiedPath.h"

#include <map>

clang::tooling::UnifiedPath
getCmakeBuildPathFromInRoot(const clang::tooling::UnifiedPath &InRoot,
                            const clang::tooling::UnifiedPath &OutRoot);
void collectCmakeScripts(
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot,
    std::vector<clang::tooling::UnifiedPath> &CmakeScriptFiles);
bool migrateCmakeScriptFile(const clang::tooling::UnifiedPath &InRoot,
                            const clang::tooling::UnifiedPath &OutRoot,
                            const clang::tooling::UnifiedPath &InFileName);
bool cmakeScriptFileSpecified(const std::vector<std::string> &SourceFiles);

void migrateCmakeScriptOnly(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot);

void parseVariable(const std::string &Input,
                   std::map<std::string, std::string> &VariablesMap);

void cmakeSyntaxProcessed(
    std::string &Input, const std::map<std::string, std::string> &VariablesMap);

std::string convertCmakeCommandsToLower(const std::string &InputString);
#endif
