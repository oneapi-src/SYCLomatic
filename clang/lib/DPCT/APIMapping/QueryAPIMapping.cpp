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
std::set<std::string> APIMapping::EntrySet;
bool APIMapping::PrintAll;

void APIMapping::registerEntry(std::string Name, llvm::StringRef SourceCode) {
  std::replace(Name.begin(), Name.end(), '$', ':');
  if (getPrintAll()) {
    EntrySet.insert(Name);
    return;
  }
  const auto TargetIndex = EntryArray.size();
  EntryMap[Name] = TargetIndex; // Set the entry whether it exist or not.
  // Try to fuzz the original API name (only when the entry not exist):
  // 1. Change "Name" to lower case. (Querying will change "Key" to lower too)
  // 2. Remove partial or all suffix '_'.
  std::transform(Name.begin(), Name.end(), Name.begin(), ::tolower);
  while (Name.back() == '_') {
    Name.erase(Name.end() - 1);
    EntryMap.try_emplace(Name, TargetIndex);
  }
  const auto EmplaceWithAndWithoutSuffix =
      [TargetIndex](const std::string &Name, llvm::StringRef Suffix) {
        EntryMap.try_emplace(Name, TargetIndex);
        if (Name.size() > Suffix.size() &&
            Name.substr(Name.size() - Suffix.size()) == Suffix) {
          EntryMap.try_emplace(Name.substr(0, Name.size() - Suffix.size()),
                               TargetIndex);
        } else {
          EntryMap.try_emplace(Name + Suffix.str(), TargetIndex);
        }
      };
  // 3. Remove partial or all leading '_'.
  // 4. For each name got by step 1, put 2 kind of fuzzed name into the map
  // keys:
  //   (1) original name
  //   (2) remove or add Suffix "_v2"
  EmplaceWithAndWithoutSuffix(Name, "_v2");
  while (Name.front() == '_') {
    Name.erase(0, 1);
    EmplaceWithAndWithoutSuffix(Name, "_v2");
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
    if (Key.find('<') != std::string::npos ||
        Key.find('>') != std::string::npos) {
      Key = "kernel";
    }
    std::transform(Key.begin(), Key.end(), Key.begin(), ::tolower);
    Iter = EntryMap.find(Key);
  }
  if (Iter == EntryMap.end())
    return llvm::StringRef{};
  return EntryArray[Iter->second];
}

void APIMapping::printAll() {
  for (const auto &Name : EntrySet)
    llvm::outs() << Name << "\n";
}

} // namespace dpct
} // namespace clang
