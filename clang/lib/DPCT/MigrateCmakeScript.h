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
#include "llvm/ADT/StringRef.h"


std::string getCmakeBuildPathFromInRoot(llvm::StringRef InRoot,
                                        llvm::StringRef OutRoot);
void collectCmakeScripts(llvm::StringRef InRoot, llvm::StringRef OutRoot,
                         std::vector<std::string> &CmakeScriptFiles);
bool migrateCmakeScriptFile(llvm::StringRef InRoot, llvm::StringRef OutRoot,
                            std::string InFileName);
bool cmakeScriptFileSpecified(const std::vector<std::string> &SourceFiles);

void migrateCmakeScriptOnly(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    llvm::StringRef InRoot, llvm::StringRef OutRoot);
#endif