//===--------------- ExternalReplacement.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __EXTERNAL_REPLACEMENT_H__
#define __EXTERNAL_REPLACEMENT_H__

#include "IncMigration/ReMigration.h"

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

namespace clang {
namespace dpct {
int loadTUFromYaml(const clang::tooling::UnifiedPath &Input,
                   clang::tooling::TranslationUnitReplacements &TU);
void loadGDCFromYaml(const clang::tooling::UnifiedPath &Input,
                     clang::dpct::GitDiffChanges &GDC);
int save2Yaml(
    const std::vector<clang::tooling::Replacement> &Replaces,
    const std::map<clang::tooling::UnifiedPath,
                   std::vector<clang::tooling::CompilationInfo>>
        &CompileTargets);

void mergeAndUniqueReps(
    clang::tooling::Replacements &Replaces,
    const std::vector<clang::tooling::Replacement> &PreRepls);

bool tryLoadingUpstreamChangesAndUserChanges();
} // namespace dpct
} // namespace clang

#endif  // __EXTERNAL_REPLACEMENT_H__
