//===----------------------- ReMigration.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExternalReplacement.h"
#include "AnalysisInfo.h"

using namespace clang::dpct;

int tryLoadingUpstreamChangesAndUserChanges() {
  llvm::SmallString<128> UpstreamChangesFilePath(
      DpctGlobalInfo::getInRoot().getCanonicalPath());
  llvm::SmallString<128> UserChangesFilePath(
      DpctGlobalInfo::getInRoot().getCanonicalPath());
  llvm::sys::path::append(UpstreamChangesFilePath, "UpstreamChanges.yaml");
  llvm::sys::path::append(UserChangesFilePath, "UserChanges.yaml");

  clang::tooling::GitDiffChanges UpstreamChanges;
  clang::tooling::GitDiffChanges UserChanges;
  loadGDCFromYaml(UpstreamChangesFilePath, UpstreamChanges);
  loadGDCFromYaml(UserChangesFilePath, UserChanges);

  return 0;
}
