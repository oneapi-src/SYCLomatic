//===--------------- GenMakefile.h ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_GEN_MAKEFILE_H
#define DPCT_GEN_MAKEFILE_H

#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"

#include <map>
#include <string>

namespace llvm {
class StringRef;
}

namespace clang {
namespace tooling {
class RefactoringTool;
}
} // namespace clang

namespace clang {
namespace dpct {
/// Generates makefile for migrated file(s) in -out-root directory.
/// The name of generated makefile is specified by \p BuildScriptName
void genBuildScript(clang::tooling::RefactoringTool &Tool,
                    clang::tooling::UnifiedPath& InRoot, clang::tooling::UnifiedPath& OutRoot,
                    const std::string &BuildScriptName);

extern std::map<clang::tooling::UnifiedPath /*target*/,
                std::vector<clang::tooling::CompilationInfo>>
    CompileCmdsPerTarget;

extern std::vector<
    std::pair<clang::tooling::UnifiedPath /*target*/,
              std::vector<std::string> /*orginal compile command*/>>
    CompileTargetsMap;
} // namespace dpct
} // namespace clang
#endif // DPCT_GEN_MAKEFILE_H
