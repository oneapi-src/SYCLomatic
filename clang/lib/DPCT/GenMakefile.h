//===--- GenMakefile.h ---------------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef DPCT_GEN_MAKEFILE_H
#define DPCT_GEN_MAKEFILE_H

#include "clang/Tooling/Core/Replacement.h"

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

/// Generates makefile for migrated file(s) in -out-root directory.
/// The name of generated makefile is specified by \p BuildScriptName
void genBuildScript(clang::tooling::RefactoringTool &Tool,
                    llvm::StringRef InRoot, llvm::StringRef OutRoot,
                    const std::string &BuildScriptName);

extern std::map<std::string /*target*/,
                std::vector<clang::tooling::CompilationInfo>>
    CompileCmdsPerTarget;

extern std::vector<
    std::pair<std::string /*target*/,
              std::vector<std::string> /*orginal compile command*/>>
    CompileTargetsMap;

#endif // DPCT_GEN_MAKEFILE_H
