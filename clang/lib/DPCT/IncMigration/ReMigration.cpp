//===----------------------- ReMigration.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReMigration.h"

#include "AnalysisInfo.h"
#include "ExternalReplacement.h"

namespace clang::dpct {
static GitDiffChanges UpstreamChanges;
static GitDiffChanges UserChanges;
AddFileHunk::AddFileHunk(std::string NewFilePath)
    : Hunk(AddFile),
      NewFilePath(tooling::UnifiedPath(NewFilePath).getCanonicalPath()) {}
DeleteFileHunk::DeleteFileHunk(std::string OldFilePath)
    : Hunk(DeleteFile),
      OldFilePath(tooling::UnifiedPath(OldFilePath).getCanonicalPath()) {}
GitDiffChanges &getUpstreamChanges() { return UpstreamChanges; }
GitDiffChanges &getUserChanges() { return UserChanges; }
static void dumpGitDiffChanges(const GitDiffChanges &GHC) {
  llvm::errs() << "GitDiffChanges:\n";
  llvm::errs() << "  ModifyFileHunks:\n";
  for (const auto &Hunk : GHC.ModifyFileHunks) {
    llvm::errs() << "    - FilePath:        " << Hunk.getFilePath() << "\n";
    llvm::errs() << "      Offset:          " << Hunk.getOffset() << "\n";
    llvm::errs() << "      Length:          " << Hunk.getLength() << "\n";
    llvm::errs() << "      ReplacementText: " << Hunk.getReplacementText()
                 << "\n";
  }
  llvm::errs() << "  AddFileHunks:\n";
  for (const auto &Hunk : GHC.AddFileHunks) {
    llvm::errs() << "    - NewFilePath: " << Hunk.getNewFilePath() << "\n";
  }
  llvm::errs() << "  DeleteFileHunks:\n";
  for (const auto &Hunk : GHC.DeleteFileHunks) {
    llvm::errs() << "    - OldFilePath: " << Hunk.getOldFilePath() << "\n";
  }
  llvm::errs() << "  MoveFileHunks:\n";
  for (const auto &Hunk : GHC.MoveFileHunks) {
    llvm::errs() << "    - FilePath:        " << Hunk.getFilePath() << "\n";
    llvm::errs() << "      Offset:          " << Hunk.getOffset() << "\n";
    llvm::errs() << "      Length:          " << Hunk.getLength() << "\n";
    llvm::errs() << "      ReplacementText: " << Hunk.getReplacementText()
                 << "\n";
    llvm::errs() << "      NewFilePath:     " << Hunk.getNewFilePath() << "\n";
  }
}

void tryLoadingUpstreamChangesAndUserChanges() {
  llvm::SmallString<128> UpstreamChangesFilePath(
      DpctGlobalInfo::getInRoot().getCanonicalPath());
  llvm::SmallString<128> UserChangesFilePath(
      DpctGlobalInfo::getInRoot().getCanonicalPath());
  llvm::sys::path::append(UpstreamChangesFilePath, "UpstreamChanges.yaml");
  llvm::sys::path::append(UserChangesFilePath, "UserChanges.yaml");

  if (llvm::sys::fs::exists(UpstreamChangesFilePath)) {
    ::loadGDCFromYaml(UpstreamChangesFilePath, getUpstreamChanges());
  }
  if (llvm::sys::fs::exists(UserChangesFilePath)) {
    ::loadGDCFromYaml(UserChangesFilePath, getUserChanges());
  }
}
} // namespace clang::dpct
