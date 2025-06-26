//===--------------- RecommendLibraries.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RecommendLibraries.h"
#include "FileGenerator/GenFiles.h"
#include "Statics.h"
#include <string>
#include <unordered_map>

namespace clang {
namespace dpct {
struct RecommendLib {
  std::string Feature;
  std::string SupportedVersion;
  std::string ReplacementText;
  LibType CompType;
  std::string Description;

  RecommendLib() = default;
  RecommendLib(std::unordered_map<LibType, RecommendLib> &Table,
               std::string Feature, std::string SupportedVersion, LibType CT,
               std::string ReplacementText, std::string Description)
      : Feature(Feature), SupportedVersion(SupportedVersion),
        ReplacementText(ReplacementText), CompType(CT),
        Description(Description) {
    Table[CT] = *this;
  }
};
static std::unordered_map<clang::dpct::LibType, clang::dpct::RecommendLib>
    RecommendLibs;
std::vector<clang::dpct::RecommendLib> RecommendLibList;

#define RECOMMENDLIBRARY(NAME, Feature, VERSION, COMPTYPE, REPLACEMENT, MSG)   \
  RecommendLib DepRecommend_##NAME(RecommendLibs, Feature, VERSION, COMPTYPE,  \
                                   REPLACEMENT, MSG);
#include "RecommendLibrariesVersion.inc"

void CollectDepLib(ReplTy &Repls) {
  for (auto Entry : RecommendLibs) {
    auto &Lib = Entry.second;
    [&]() {
      for (auto Repl : Repls) {
        for (auto Item : Repl.second) {
          if (Item.getReplacementText().str().find(Lib.ReplacementText) !=
              std::string::npos) {
            RecommendLibList.push_back(Lib);
            return;
          }
        }
      }
    }();
  }
}

std::string LibTypeToString(LibType version) {
  switch (version) {
  case LibType::DPCPP:
    return "oneAPI DPC++ compiler Open Source Version";
  case LibType::oneDPL:
    return "oneAPI DPC++ Library Open Source Version";
  case LibType::oneMath:
    return "oneAPI Math Library Open Source Version";
  case LibType::oneCCL:
    return "oneAPI Collective Communications Library Open Source Version";
  case LibType::oneDNNL:
    return "oneAPI Deep Neural Network Library Open Source Version";
  case LibType::ISHMEM:
    return "oneAPI SHMEM Library Open Source Version";
  default:
    return "Unknown Component Type  ";
  }
}

void PrintRecommendLibs(llvm::raw_ostream &OStream) {
  if (RecommendLibList.empty())
    return;
  if (DpctGlobalInfo::isAnalysisModeEnabled())
    OStream << llvm::raw_ostream::Colors::BLUE;

  OStream << "Recommend Library Dependencies of SYCL Project:\n";

  if (DpctGlobalInfo::isAnalysisModeEnabled())
    OStream << llvm::raw_ostream::Colors::RESET;
  for (auto &Status : RecommendLibList) {
    OStream << "  - The " + Status.Feature + " is supported in " +
                   LibTypeToString(Status.CompType) + " and after " +
                   Status.SupportedVersion + ". " + Status.Description + "\n";
  }
}


} // namespace dpct
} // namespace clang