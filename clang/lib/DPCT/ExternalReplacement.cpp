//===--- ExternalReplacement.cpp ------------------------*- C++-*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
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

#include "Utility.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "ExternalReplacement.h"
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

int save2Yaml(StringRef YamlFile, StringRef SrcFileName,
              const Replacements &Replaces) {
  std::string YamlContent;
  llvm::raw_string_ostream YamlContentStream(YamlContent);
  llvm::yaml::Output YAMLOut(YamlContentStream);

  // list all the replacement.
  clang::tooling::TranslationUnitReplacements TUR;
  TUR.MainSourceFile = SrcFileName;
  TUR.Replacements.insert(TUR.Replacements.end(), Replaces.begin(),
                          Replaces.end());
  YAMLOut << TUR;
  YamlContentStream.flush();
  // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
  // on windows.
  std::ofstream File(YamlFile, std::ios::binary);
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
  if (YAMLIn.error()) {
    // File doesn't appear to be a header change description. Ignore it.
    return -1;
  }
  return 0;
}

void mergeAndUniqueReps(Replacements &Replaces,
                        clang::tooling::TranslationUnitReplacements &PreTU) {

  bool DupFlag = false;
  for (const auto &OldR : PreTU.Replacements) {
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

int mergeExternalReps(std::string SrcFileName, Replacements &Replaces) {
  std::string YamlFile = SrcFileName + ".yaml";
  clang::tooling::TranslationUnitReplacements PreTU;
  int ret = 0;
  if (fs::exists(YamlFile)) {
    llvm::errs() << YamlFile << " exist, try to merge it.\n";
    ret = loadFromYaml(std::move(YamlFile), PreTU);
    if (ret == 0) {
      mergeAndUniqueReps(Replaces, PreTU);
    }
  }
  llvm::errs() << "Save out new version " << YamlFile << " file\n";
  save2Yaml(std::move(YamlFile), std::move(SrcFileName), Replaces);
  return ret;
}
