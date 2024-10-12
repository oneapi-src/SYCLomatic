//===--------------- ExternalReplacement.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//  This file target to process replacement based operation:
//   -Save replacement to external (disk file)
//   -Load replacement from external (disk file)
//   -Merge replacement in current migration with previous migration.

#include "AnalysisInfo.h"
#include "Utility.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "ExternalReplacement.h"
#include "IncrementalMigrationUtility.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_os_ostream.h"

#include <algorithm>
#include <cassert>
#include <fstream>

using namespace llvm;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;
using clang::tooling::Replacements;

int save2Yaml(
    clang::tooling::UnifiedPath& YamlFile, clang::tooling::UnifiedPath& SrcFileName,
    const std::vector<clang::tooling::Replacement> &Replaces,
    const std::vector<clang::tooling::MainSourceFileInfo> &MainSrcFilesDigest,
    const std::map<clang::tooling::UnifiedPath, std::vector<clang::tooling::CompilationInfo>>
        &CompileTargets) {
  std::string YamlContent;
  llvm::raw_string_ostream YamlContentStream(YamlContent);
  llvm::yaml::Output YAMLOut(YamlContentStream);

  // list all the replacement.
  clang::tooling::TranslationUnitReplacements TUR;
  TUR.MainSourceFile = SrcFileName.getCanonicalPath();
  TUR.Replacements.insert(TUR.Replacements.end(), Replaces.begin(),
                          Replaces.end());

  // std::transform(
  //     MainSrcFilesDigest.begin(), MainSrcFilesDigest.end(),
  //     std::back_insert_iterator<
  //         std::vector<clang::tooling::MainSourceFileInfo>(
  //         TUR.MainSourceFilesDigest),
  //     [](const clang::tooling::MainSourceFileInfo &P) {
  //       return clang::tooling::MainSourceFileInfo(P.MainSourceFile,
  //                                                  P.Digest, P.HasCUDASyntax);
  //     });

  //TUR.MainSourceFilesDigest = MainSrcFilesDigest;
  for (auto &Entry : MainSrcFilesDigest) {
    printf("1 %s\n", Entry.MainSourceFile.c_str());
    printf("2 %s\n", Entry.Digest.c_str());
    printf("3 %d\n", Entry.HasCUDASyntax);
    TUR.MainSourceFilesDigest.push_back(clang::tooling::MainSourceFileInfo(
        Entry.MainSourceFile, Entry.Digest, Entry.HasCUDASyntax));
  }

  for (const auto &Entry : CompileTargets) {
    TUR.CompileTargets[Entry.first.getCanonicalPath().str()] = Entry.second;
  }

  TUR.DpctVersion = clang::dpct::getDpctVersionStr();
  TUR.OptionMap = clang::dpct::DpctGlobalInfo::getCurrentOptMap();

  YAMLOut << TUR;
  YamlContentStream.flush();
  clang::dpct::writeDataToFile(YamlFile.getCanonicalPath().str(), YamlContent);
  return 0;
}

int loadFromYaml(const clang::tooling::UnifiedPath& Input,
                 clang::tooling::TranslationUnitReplacements &TU) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(Input.getCanonicalPath());
  if (!Buffer) {
    llvm::errs() << "error: failed to read " << Input.getCanonicalPath() << ": "
                 << Buffer.getError().message() << "\n";
    return -1;
  }

  llvm::yaml::Input YAMLIn(Buffer.get()->getBuffer());
  YAMLIn >> TU;

  bool IsSrcFileChanged = false;
  for (const auto &digest: TU.MainSourceFilesDigest) {
    auto Hash = llvm::sys::fs::md5_contents(digest.MainSourceFile);
    if (Hash && Hash->digest().c_str() != digest.Digest) {
      llvm::errs() << "Warning: The file '" << digest.MainSourceFile
                   << "' has been changed during incremental migration.\n";
      IsSrcFileChanged = true;
    }
  }

  if (IsSrcFileChanged || YAMLIn.error()) {
    // File doesn't appear to be a header change description. Ignore it.
    TU = clang::tooling::TranslationUnitReplacements();
    return -1;
  }

  return 0;
}

void mergeAndUniqueReps(
    Replacements &Replaces,
    const std::vector<clang::tooling::Replacement> &PreRepls) {

  bool DupFlag = false;
  for (const auto &OldR : PreRepls) {
    DupFlag = false;

    for (const auto &CurrR : Replaces) {

      if (CurrR.getFilePath() != OldR.getFilePath()) {
        llvm::errs() << "Ignore " << OldR.getFilePath().str()
                     << " for differnt path!\n";
        return;
      }
      if ((CurrR.getFilePath() == OldR.getFilePath()) &&
          (CurrR.getOffset() == OldR.getOffset()) &&
          (CurrR.getLength() == OldR.getLength()) &&
          (CurrR.getReplacementText() == OldR.getReplacementText())) {
        DupFlag = true;
        break;
      }
    }
    if (DupFlag == false) {
      if (auto Err = Replaces.add(OldR)) {
        llvm::dbgs() << "Adding replacement when merging previous "
                        "replacement: Error occured!\n"
                     << Err << "\n";
      }
    }
  }
}

int mergeExternalReps(clang::tooling::UnifiedPath InRootSrcFilePath,
                      clang::tooling::UnifiedPath OutRootSrcFilePath, Replacements &Replaces) {
  clang::tooling::UnifiedPath YamlFile = OutRootSrcFilePath.getCanonicalPath() + ".yaml";

  auto PreTU = clang::dpct::DpctGlobalInfo::getInstance()
                   .getReplInfoFromYAMLSavedInFileInfo(InRootSrcFilePath);

  if (PreTU) {
    llvm::errs() << YamlFile << " exist, try to merge it.\n";

    mergeAndUniqueReps(Replaces, (*PreTU).Replacements);
  }

  llvm::errs() << "Saved new version of " << YamlFile << " file\n";

  std::vector<clang::tooling::Replacement> Repls(Replaces.begin(),
                                                 Replaces.end());

  auto Hash = llvm::sys::fs::md5_contents(InRootSrcFilePath.getCanonicalPath());

  bool HasCUDASyntax = false;
  if (auto FileInfo = dpct::DpctGlobalInfo::getInstance().findFile(InRootSrcFilePath.getCanonicalPath())) {
    if (FileInfo->hasCUDASyntax()) {
      HasCUDASyntax = true;
    }
  }
  clang::tooling::MainSourceFileInfo FileDigest(InRootSrcFilePath.getPath().str(),
                                                    Hash->digest().c_str(), HasCUDASyntax);

  printf("###mergeExternalReps######## InRootSrcFilePath.getPath().str(): %s\n", FileDigest.MainSourceFile.c_str());
  printf("###mergeExternalReps######## InRootSrcFilePath.getPath().str(): %s\n", FileDigest.Digest.c_str());
  printf("###mergeExternalReps######## InRootSrcFilePath.getPath().str(): %d\n", FileDigest.HasCUDASyntax);
  std::map<clang::tooling::UnifiedPath, std::vector<clang::tooling::CompilationInfo>>
      CompileTargets;
  save2Yaml(YamlFile, OutRootSrcFilePath, Repls,
            {FileDigest}, CompileTargets);
  return 0;
}
