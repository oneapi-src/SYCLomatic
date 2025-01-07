//===--------------- MigrateBuildScript.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_MIGRATE_BUILD_SCRIPT_H
#define DPCT_MIGRATE_BUILD_SCRIPT_H

#include "Diagnostics/Diagnostics.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Core/UnifiedPath.h"

#include <map>
namespace clang {
namespace dpct {

bool cmakeScriptNotFound();
bool pythonBuildScriptNotFound();
std::string readFile(const clang::tooling::UnifiedPath &Name);
std::vector<std::string> split(const std::string &Input,
                               const std::string &Delimiter);
void storeBufferToFile(
    std::map<clang::tooling::UnifiedPath, std::string>
        &BuildScriptFileBufferMap,
    std::map<clang::tooling::UnifiedPath, bool> &ScriptFileCRLFMap);
void collectBuildScripts(const clang::tooling::UnifiedPath &InRoot,
                         const clang::tooling::UnifiedPath &OutRoot);
bool loadBufferFromScriptFile(const clang::tooling::UnifiedPath InRoot,
                              const clang::tooling::UnifiedPath OutRoot,
                              clang::tooling::UnifiedPath InFileName,
                              std::map<clang::tooling::UnifiedPath, std::string>
                                  &BuildScriptFileBufferMap);
bool buildScriptFileSpecified(const std::vector<std::string> &SourceFiles);
void collectBuildScriptsSpecified(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    const clang::tooling::UnifiedPath &InRoot,
    const clang::tooling::UnifiedPath &OutRoot);
void loadBufferFromFile(const clang::tooling::UnifiedPath &InRoot,
                        const clang::tooling::UnifiedPath &OutRoot,
                        std::map<clang::tooling::UnifiedPath, std::string>
                            &BuildScriptFileBufferMap,
                        bool isCMake = true);
void unifyInputFileFormat(
    std::map<clang::tooling::UnifiedPath, std::string>
        &BuildScriptFileBufferMap,
    std::map<clang::tooling::UnifiedPath, bool> &ScriptFileCRLFMap);


} // namespace dpct
} // namespace clang

#endif //! DPCT_MIGRATE_BUILD_SCRIPT_H
