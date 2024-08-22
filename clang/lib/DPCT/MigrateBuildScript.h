//===--------------- MigrateBuildScript.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_MIGRATE_BUILD_SCRIPT_H
#define DPCT_MIGRATE_BUILD_SCRIPT_H

#include "clang/Tooling/Core/UnifiedPath.h"

#include <map>

std::string readFile(const clang::tooling::UnifiedPath &Name);

std::vector<std::string> split(const std::string &Input,
                                      const std::string &Delimiter);

void storeBufferToFile(std::map<clang::tooling::UnifiedPath, std::string>
                                  BuildScriptFileBufferMap,
                       std::map<clang::tooling::UnifiedPath, bool>
                                  ScriptFileCRLFMap);
#endif
