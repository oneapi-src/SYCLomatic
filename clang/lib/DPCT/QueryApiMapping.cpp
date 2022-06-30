//===--------------- QueryApiMapping.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QueryApiMapping.h"

namespace clang {
namespace dpct {

std::unordered_map<std::string, std::shared_ptr<ApiMappingEntry>>
    ApiMappingEntry::EntryMap;

void ApiMappingEntry::registerEntry(std::shared_ptr<ApiMappingEntry> Entry) {
  EntryMap[Entry->Source] = Entry;
  EntryMap[Entry->Dest] = Entry;
}

void ApiMappingEntry::initEntryMap() {
#define REGISTER_ENTRY(SOURCE, DEST, DESC)                                       \
  registerEntry(std::make_shared<ApiMappingEntry>(SOURCE, DEST, DESC));

#include "ApiMapping/ApiMapping.inc"

#undef REGISTER_ENTRY
}

} // namespace dpct
} // namespace clang