//===--------------- QueryApiMapping.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPC_QUERY_API_MAPPING_H
#define DPC_QUERY_API_MAPPING_H

#include <memory>
#include <string>
#include <unordered_map>

namespace clang {
namespace dpct {

class ApiMappingEntry {
  std::string Source;
  std::string Dest;
  std::string Description;

  static std::unordered_map<std::string, std::shared_ptr<ApiMappingEntry>>
      EntryMap;

  static void registerEntry(std::shared_ptr<ApiMappingEntry> Entry);

public:
  ApiMappingEntry(std::string Src, std::string Destination, std::string Desc)
      : Source(std::move(Src)), Dest(std::move(Destination)),
        Description(std::move(Desc)) {}

  static void initEntryMap();

  template <class StreamTy>
  static void printMappingDesc(StreamTy &Stream, const std::string &Key) {
    auto Iter = EntryMap.find(Key);
    if (Iter == EntryMap.end()) {
      static const std::string NotFoundPrefix = "Mapping for ";
      static const std::string NotFoundSuffix = " is not available";
      Stream << NotFoundPrefix << Key << NotFoundSuffix << "\n";
      return;
    }
    Stream << Iter->second->Description;
  }

};

} // namespace dpct
} // namespace clang

#endif // DPC_QUERY_API_MAPPING_H