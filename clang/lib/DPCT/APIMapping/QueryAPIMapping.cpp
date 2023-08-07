//===--------------- QueryAPIMapping.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QueryAPIMapping.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace dpct {

llvm::DenseMap<llvm::StringRef, llvm::StringRef> APIMapping::EntryMap;

void APIMapping::registerEntry(llvm::StringRef Name,
                               llvm::StringRef SourceCode) {
  EntryMap[Name] = SourceCode;
}

void APIMapping::initEntryMap(){
#include "APIMappingRegister.def"
}

llvm::StringRef APIMapping::getAPISourceCode(llvm::StringRef Key) {
  auto Iter = EntryMap.find(Key);
  if (Iter == EntryMap.end())
    return llvm::StringRef{};
  return Iter->second;
}

} // namespace dpct
} // namespace clang
