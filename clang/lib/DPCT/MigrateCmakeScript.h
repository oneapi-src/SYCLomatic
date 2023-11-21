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
#include "clang/Tooling/Core/DpctPath.h"

clang::tooling::DpctPath
getCmakeBuildPathFromInRoot(const clang::tooling::DpctPath &InRoot,
                            const clang::tooling::DpctPath &OutRoot);
void collectCmakeScripts(
    const clang::tooling::DpctPath &InRoot,
    const clang::tooling::DpctPath &OutRoot,
    std::vector<clang::tooling::DpctPath> &CmakeScriptFiles);
bool migrateCmakeScriptFile(const clang::tooling::DpctPath &InRoot,
                            const clang::tooling::DpctPath &OutRoot,
                            const clang::tooling::DpctPath &InFileName);
bool cmakeScriptFileSpecified(const std::vector<std::string> &SourceFiles);

void migrateCmakeScriptOnly(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::DpctPath &InRoot,
    const clang::tooling::DpctPath &OutRoot);
#endif