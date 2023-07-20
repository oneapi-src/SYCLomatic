//===--------------- ExternalReplacement.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __EXTERNAL_REPLACEMENT_H__
#define __EXTERNAL_REPLACEMENT_H__

#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <vector>

namespace llvm {
class StringRef;
}

namespace clang {
namespace tooling {
class RefactoringTool;
class Replacements;
} // namespace tooling
} // namespace clang

int mergeExternalReps(std::string InRootSrcFilePath,
                      std::string OutRootSrcFilePath,
                      clang::tooling::Replacements &Replaces);
int loadFromYaml(llvm::StringRef Input,
                 clang::tooling::TranslationUnitReplacements &TU);
int save2Yaml(
    llvm::StringRef YamlFile, llvm::StringRef SrcFileName,
    const std::vector<clang::tooling::Replacement> &Replaces,
    const std::vector<std::pair<std::string, std::string>> &MainSrcFilesDigest,
    const std::map<std::string, std::vector<clang::tooling::CompilationInfo>>
        &CompileTargets);

void mergeAndUniqueReps(
    clang::tooling::Replacements &Replaces,
    const std::vector<clang::tooling::Replacement> &PreRepls);

#endif
