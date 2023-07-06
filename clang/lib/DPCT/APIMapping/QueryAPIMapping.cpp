//===--------------- QueryAPIMapping.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QueryAPIMapping.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_map>

namespace clang {
namespace dpct {

std::unordered_map<std::string, size_t> APIMapping::EntryMap;
std::vector<llvm::StringRef> APIMapping::EntryArray;

void APIMapping::registerEntry(std::string Name, llvm::StringRef SourceCode) {
  std::replace(Name.begin(), Name.end(), '$', ':');
  // Try to fuzz the original API name:
  // 1. Remove 0/1/2/... leader '_'.
  // 2. For each name got by step 1, put 4 kind of fuzzed name into the map
  // keys:
  //   (1) original name
  //   (2) first char upper case name
  //   (3) all char upper case name
  //   (4) all char lower case name
  for (int i = Name.find_first_not_of("_"); i >= 0; --i) {
    EntryMap[Name] = EntryArray.size();
    auto TempName = Name;
    TempName[i] = std::toupper(TempName[i]);
    EntryMap[TempName] = EntryArray.size();
    std::transform(TempName.begin(), TempName.end(), TempName.begin(),
                   ::toupper);
    EntryMap[TempName] = EntryArray.size();
    std::transform(TempName.begin(), TempName.end(), TempName.begin(),
                   ::tolower);
    EntryMap[TempName] = EntryArray.size();
    Name.erase(0, 1);
  }
  EntryArray.emplace_back(SourceCode);
}

void APIMapping::initEntryMap(){
#include "APIMappingRegister.def"
}

llvm::StringRef APIMapping::getAPISourceCode(std::string Key) {
  Key.erase(0, Key.find_first_not_of(" "));
  Key.erase(Key.find_last_not_of(" ") + 1);
  auto Iter = EntryMap.find(Key);
  if (Iter == EntryMap.end()) {
    std::transform(Key.begin(), Key.end(), Key.begin(), ::tolower);
    Iter = EntryMap.find(Key);
  }
  if (Iter == EntryMap.end())
    return llvm::StringRef{};
  return EntryArray[Iter->second];
}

} // namespace dpct
} // namespace clang
