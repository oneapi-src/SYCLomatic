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
std::unordered_map<std::string, size_t> APIMapping::EntryMapBuffer;
std::vector<llvm::StringRef> APIMapping::EntryArray;
std::set<std::string> APIMapping::EntrySet;
bool APIMapping::PrintAll;

void APIMapping::registerEntry(std::string Name, llvm::StringRef Str,
                               std::unordered_map<std::string, size_t> &Map) {
  std::replace(Name.begin(), Name.end(), '$', ':');
  if (getPrintAll()) {
    EntrySet.insert(Name);
    return;
  }
  const auto TargetIndex = EntryArray.size();
  Map[Name] = TargetIndex; // Set the entry whether it exist or not.
  // Try to fuzz the original API name (only when the entry not exist):
  // 1. Remove partial or all leading '_'.
  // 2. For each name got by step 1, put 4 kind of fuzzed name into the map
  // keys:
  //   (1) original name
  //   (2) remove or add Suffix "_v2"
  //   (3) first char upper case name
  //   (4) all char upper case name
  //   (5) all char lower case name
  for (int i = Name.find_first_not_of("_"); i >= 0; --i) {
    auto TempName = Name;
    std::string Suffix = "_v2";
    if (TempName.size() > Suffix.length() &&
        TempName.substr(TempName.size() - Suffix.length()) == Suffix) {
      Map.try_emplace(TempName.substr(0, TempName.size() - 3), TargetIndex);
    } else {
      Map.try_emplace(TempName + Suffix, TargetIndex);
    }
    TempName[i] = std::toupper(TempName[i]);
    Map.try_emplace(TempName, TargetIndex);
    std::transform(TempName.begin(), TempName.end(), TempName.begin(),
                   ::toupper);
    Map.try_emplace(TempName, TargetIndex);
    std::transform(TempName.begin(), TempName.end(), TempName.begin(),
                   ::tolower);
    Map.try_emplace(TempName, TargetIndex);
    Name.erase(0, 1);
    Map.try_emplace(Name, TargetIndex);
  }
  EntryArray.emplace_back(Str);
}

void APIMapping::initEntryMap() {
#define REGIST(API_NAME, FILE_STR)                                             \
  registerEntry(API_NAME, FILE_STR, EntryMapBuffer);
#include "APIMappingRegisterBuffer.def"
#undef REGIST
#define REGIST(API_NAME, FILE_STR) registerEntry(API_NAME, FILE_STR, EntryMap);
#include "APIMappingRegister.def"
#undef REGIST
}

llvm::StringRef
APIMapping::getAPIStr(std::string &Key,
                      const std::unordered_map<std::string, size_t> &Map) {
  Key.erase(0, Key.find_first_not_of(" "));
  Key.erase(Key.find_last_not_of(" ") + 1);
  auto Iter = Map.find(Key);
  if (Iter == Map.end()) {
    std::transform(Key.begin(), Key.end(), Key.begin(), ::tolower);
    Iter = Map.find(Key);
  }
  if (Iter == Map.end())
    return llvm::StringRef{};
  return EntryArray[Iter->second];
}

llvm::StringRef APIMapping::getAPIStr(std::string Key) {
  auto Str = getAPIStr(Key, EntryMapBuffer);
  if (Str.empty()) {
    Str = getAPIStr(Key, EntryMap);
  }
  return Str;
}

void APIMapping::printAll() {
  for (const auto &Name : EntrySet)
    llvm::outs() << Name << "\n";
}

} // namespace dpct
} // namespace clang
