//===----------------------- MigrateBuildScript.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "MigrateBuildScript.h"
#include "Utility.h"

using namespace clang::dpct;

std::string readFile(const clang::tooling::UnifiedPath &Name) {
  std::ifstream Stream(Name.getCanonicalPath().str(),
                       std::ios::in | std::ios::binary);
  std::string Contents((std::istreambuf_iterator<char>(Stream)),
                       (std::istreambuf_iterator<char>()));
  return Contents;
}

std::vector<std::string> split(const std::string &Input,
                               const std::string &Delimiter) {
  std::vector<std::string> Vec;
  if (!Input.empty()) {

    size_t Index = 0;
    size_t Pos = Input.find(Delimiter, Index);
    while (Index < Input.size() && Pos != std::string::npos) {
      Vec.push_back(Input.substr(Index, Pos - Index));

      Index = Pos + Delimiter.size();
      Pos = Input.find(Delimiter, Index);
    }
    // Append the remaining part
    Vec.push_back(Input.substr(Index));
  }
  return Vec;
}

void storeBufferToFile(std::map<clang::tooling::UnifiedPath, std::string>
                                  BuildScriptFileBufferMap,
                       std::map<clang::tooling::UnifiedPath, bool>
                                  ScriptFileCRLFMap) {
  for (auto &Entry : BuildScriptFileBufferMap) {
    auto &FileName = Entry.first;
    auto &Buffer = Entry.second;

    dpct::RawFDOStream Stream(FileName.getCanonicalPath().str());
    // Restore original endline format
    auto IsCRLF = ScriptFileCRLFMap[FileName];
    if (IsCRLF) {
      std::stringstream ResultStream;
      std::vector<std::string> SplitedStr = split(Buffer, '\n');
      for (auto &SS : SplitedStr) {
        ResultStream << SS << "\r\n";
      }
      Stream << llvm::StringRef(ResultStream.str().c_str());
    } else {
      Stream << llvm::StringRef(Buffer.c_str());
    }
    Stream.flush();
  }
}
