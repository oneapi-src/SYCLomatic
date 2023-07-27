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

llvm::DenseMap<llvm::StringRef, llvm::StringRef> APIMapping::EntryMap;

void APIMapping::registerEntry(const llvm::StringRef Name,
                               const llvm::StringRef SourceCode) {
  EntryMap[Name] = SourceCode;
}

void APIMapping::initEntryMap() {
#include "APIMappingRegister.def"
}

const llvm::StringRef APIMapping::getAPISourceCode(const llvm::StringRef Key) {
  auto Iter = EntryMap.find(Key);
  if (Iter == EntryMap.end()) {
    static const std::string Empty{""};
    return Empty;
  }
  return Iter->second;
}

} // namespace dpct
} // namespace clang
