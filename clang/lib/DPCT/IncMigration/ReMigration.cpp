//===----------------------- ReMigration.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "ExternalReplacement.h"

namespace clang::dpct {

void tryLoadingUpstreamChangesAndUserChanges() {
  llvm::SmallString<128> UpstreamChangesFilePath(
      DpctGlobalInfo::getInRoot().getCanonicalPath());
  llvm::SmallString<128> UserChangesFilePath(
      DpctGlobalInfo::getInRoot().getCanonicalPath());
  llvm::sys::path::append(UpstreamChangesFilePath, "UpstreamChanges.yaml");
  llvm::sys::path::append(UserChangesFilePath, "UserChanges.yaml");

  if (llvm::sys::fs::exists(UpstreamChangesFilePath)) {
    loadGDCFromYaml(UpstreamChangesFilePath,
                    DpctGlobalInfo::getUpstreamChanges());
  }
  if (llvm::sys::fs::exists(UserChangesFilePath)) {
    loadGDCFromYaml(UserChangesFilePath, DpctGlobalInfo::getUserChanges());
  }
}
} // namespace clang::dpct
