//===--------------- QueryAPIMapping.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_QUERY_API_MAPPING_H
#define DPCT_QUERY_API_MAPPING_H

#include "llvm/ADT/StringRef.h"

#include <set>
#include <unordered_map>
#include <vector>

namespace clang {
namespace dpct {

class APIMapping {
  static std::unordered_map<std::string, size_t> EntryMap;
  static std::unordered_map<std::string, size_t> EntryMapBuffer;
  static std::vector<llvm::StringRef> EntryArray;
  static std::set<std::string> EntrySet;
  static bool PrintAll;

  static void registerEntry(std::string Name, llvm::StringRef Description,
                            std::unordered_map<std::string, size_t> &Map);
  static llvm::StringRef
  getAPIStr(std::string &Key,
            const std::unordered_map<std::string, size_t> &Map);

public:
  static void initEntryMap();

  static llvm::StringRef getAPISourceCode(std::string Key);
  static llvm::StringRef getAPIMapping(std::string Key);

  inline static void setPrintAll(bool Flag) { PrintAll = Flag; }
  inline static bool getPrintAll() { return PrintAll; }
  static void printAll();
};

} // namespace dpct
} // namespace clang

#endif // DPCT_QUERY_API_MAPPING_H
