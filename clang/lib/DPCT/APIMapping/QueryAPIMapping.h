//===--------------- QueryAPIMapping.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_QUERY_API_MAPPING_H
#define DPCT_QUERY_API_MAPPING_H

#include <string>
#include <unordered_map>

namespace clang {
namespace dpct {

class APIMappingEntrys {
  std::string Name;
  std::string Description;

  static std::unordered_map<std::string, std::string> EntryMap;

  static void registerEntry(const std::string &Name,
                            const std::string &Description);

public:
  static void initEntryMap();

  template <class StreamTy>
  static void printMappingDesc(StreamTy &Stream, const std::string &Key) {
    auto Iter = EntryMap.find(Key);
    if (Iter == EntryMap.end() || Iter->second == "NA")
      Stream << "The API Mapping is not available\n";
    else {
      Stream << "CUDA API: " << Key << "\n";
      Stream << "Is migrated to: " << Iter->second << "\n";
    }
  }
};

} // namespace dpct
} // namespace clang

#endif // DPCT_QUERY_API_MAPPING_H
