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
    StringRef YamlFile, StringRef SrcFileNameRef,
    const std::vector<clang::tooling::Replacement> &Replaces,
    const std::vector<std::pair<std::string, std::string>> &MainSrcFilesDigest,
    const std::map<std::string, std::vector<clang::tooling::CompilationInfo>>
        &CompileTargets) {
  SmallString<512> SrcFileName;
  llvm::sys::fs::real_path(SrcFileNameRef, SrcFileName, true);
#if defined(_WIN32)
  SrcFileName = SrcFileName.str().lower();
#endif
  std::string YamlContent;
  llvm::raw_string_ostream YamlContentStream(YamlContent);
  llvm::yaml::Output YAMLOut(YamlContentStream);

  // list all the replacement.
  clang::tooling::TranslationUnitReplacements TUR;
  TUR.MainSourceFile = SrcFileName.str();
  TUR.Replacements.insert(TUR.Replacements.end(), Replaces.begin(),
                          Replaces.end());

  TUR.MainSourceFilesDigest.insert(TUR.MainSourceFilesDigest.end(),
                                   MainSrcFilesDigest.begin(),
                                   MainSrcFilesDigest.end());

  for (const auto &Entry : CompileTargets) {
    TUR.CompileTargets[Entry.first] = Entry.second;
  }

  TUR.DpctVersion = clang::dpct::getDpctVersionStr();
  TUR.OptionMap = clang::dpct::DpctGlobalInfo::getCurrentOptMap();

  // For really hidden options, do not add it in yaml file if it is not
  // specified.
  if (TUR.OptionMap[clang::dpct::OPTION_NoUseGenericSpace].Value == "false") {
    TUR.OptionMap.erase(clang::dpct::OPTION_NoUseGenericSpace);
  }

  YAMLOut << TUR;
  YamlContentStream.flush();
  // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
  // on windows.
  std::ofstream File(YamlFile.str(), std::ios::binary);
  llvm::raw_os_ostream Stream(File);
  Stream << YamlContent;
  return 0;
}

int loadFromYaml(StringRef Input,
                 clang::tooling::TranslationUnitReplacements &TU) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(Input);
  if (!Buffer) {
    llvm::errs() << "error: failed to read " << Input << ": "
                 << Buffer.getError().message() << "\n";
    return -1;
  }

  llvm::yaml::Input YAMLIn(Buffer.get()->getBuffer());
  YAMLIn >> TU;

  bool IsSrcFileChanged = false;
  for (const auto &digest: TU.MainSourceFilesDigest) {
    auto Hash = llvm::sys::fs::md5_contents(digest.first);
    if (Hash && Hash->digest().c_str() != digest.second) {
      llvm::errs() << "Warning: The file '" << digest.first
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

int mergeExternalReps(std::string InRootSrcFilePath,
                      std::string OutRootSrcFilePath, Replacements &Replaces) {
  std::string YamlFile = OutRootSrcFilePath + ".yaml";

  auto PreTU = clang::dpct::DpctGlobalInfo::getInstance()
                   .getReplInfoFromYAMLSavedInFileInfo(InRootSrcFilePath);

  if (PreTU) {
    llvm::errs() << YamlFile << " exist, try to merge it.\n";

    mergeAndUniqueReps(Replaces, (*PreTU).Replacements);
  }

  llvm::errs() << "Saved new version of " << YamlFile << " file\n";

  std::vector<clang::tooling::Replacement> Repls(Replaces.begin(),
                                                 Replaces.end());

  auto Hash = llvm::sys::fs::md5_contents(InRootSrcFilePath);
  std::pair<std::string, std::string> FileDigest = {InRootSrcFilePath,
                                                    Hash->digest().c_str()};
  std::map<std::string, std::vector<clang::tooling::CompilationInfo>>
      CompileTargets;
  save2Yaml(std::move(YamlFile), std::move(OutRootSrcFilePath), Repls,
            {FileDigest}, CompileTargets);
  return 0;
}
