//===--- ExternalReplacement.cpp ------------------------*- C++-*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
//  This file target to process replacement based operation:
//   -save replacement to external(disk file)
//   -load replacement from external(disk file)
//   -merage replacement in current migration with previous migration.

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
    StringRef YamlFile, StringRef SrcFileName,
    const std::vector<clang::tooling::Replacement> &Replaces,
    const std::vector<std::pair<std::string, std::string>> &MainSrcFilesDigest,
    const std::map<std::string, std::vector<clang::tooling::CompilationInfo>>
        &CompileTargets) {
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

  clang::c2s::updateTUR(TUR);
  TUR.C2SVersion = clang::c2s::getC2SVersionStr();
  TUR.MainHelperFileName =
      clang::c2s::C2SGlobalInfo::getCustomHelperFileName();
  TUR.OptionMap = clang::c2s::C2SGlobalInfo::getCurrentOptMap();

  // For really hidden options, do not add it in yaml file if it is not
  // specified.
  if (TUR.OptionMap[clang::c2s::OPTION_NoUseGenericSpace].Value == "false") {
    TUR.OptionMap.erase(clang::c2s::OPTION_NoUseGenericSpace);
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
                 clang::tooling::TranslationUnitReplacements &TU,
                 bool OverwriteHelperFilesInfo) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(Input);
  if (!Buffer) {
    llvm::errs() << "error: failed to read " << Input << ": "
                 << Buffer.getError().message() << "\n";
    return -1;
  }

  llvm::yaml::Input YAMLIn(Buffer.get()->getBuffer());
  YAMLIn >> TU;

  if (YAMLIn.error()) {
    // File doesn't appear to be a header change description. Ignore it.
    TU = clang::tooling::TranslationUnitReplacements();
    return -1;
  }

  if (OverwriteHelperFilesInfo) {
    clang::c2s::updateHelperNameContentMap(TU);
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

  auto PreTU = clang::c2s::C2SGlobalInfo::getInstance()
                   .getReplInfoFromYAMLSavedInFileInfo(InRootSrcFilePath);

  if (PreTU) {
    llvm::errs() << YamlFile << " exist, try to merge it.\n";

    mergeAndUniqueReps(Replaces, (*PreTU).Replacements);
  }

  llvm::errs() << "Saved new version of " << YamlFile << " file\n";

  std::vector<clang::tooling::Replacement> Repls(Replaces.begin(),
                                                 Replaces.end());

  std::vector<std::pair<std::string, std::string>> MainSrcFilesDigest;
  std::map<std::string, std::vector<clang::tooling::CompilationInfo>>
      CompileTargets;
  save2Yaml(std::move(YamlFile), std::move(OutRootSrcFilePath), Repls,
            MainSrcFilesDigest, CompileTargets);
  return 0;
}
