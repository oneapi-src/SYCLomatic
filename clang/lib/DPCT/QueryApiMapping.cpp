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
  // TODO: Now the SYCL API and cannot be used to query.
}

void ApiMappingEntry::initEntryMap() {
#define REGISTER_ENTRY(SOURCE, DESC)                                           \
  registerEntry(std::make_shared<ApiMappingEntry>(SOURCE, "", DESC));

#undef REGISTER_ENTRY
}

} // namespace dpct
} // namespace clang