//===--------------- QueryAPIMapping.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QueryAPIMapping.h"

namespace clang {
namespace dpct {

std::unordered_map<std::string, std::string> APIMapping::EntryMap;

void APIMapping::registerEntry(const std::string &Name,
                               const std::string &SourceCode) {
  EntryMap[Name] = SourceCode;
}

void APIMapping::initEntryMap() {
#include "APIMappingRegister.def"
}

const std::string &APIMapping::getAPISourceCode(const std::string &Key) {
  auto Iter = EntryMap.find(Key);
  if (Iter == EntryMap.end()) {
    static const std::string Empty{""};
    return Empty;
  }
  return Iter->second;
}

} // namespace dpct
} // namespace clang
