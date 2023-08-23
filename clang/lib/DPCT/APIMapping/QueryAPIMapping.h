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

#include <unordered_map>
#include <vector>

namespace clang {
namespace dpct {

class APIMapping {
  static std::unordered_map<std::string, size_t> EntryMap;
  static std::vector<llvm::StringRef> EntryArray;

  static void registerEntry(std::string Name, llvm::StringRef Description);

public:
  static void initEntryMap();

  static llvm::StringRef getAPISourceCode(std::string Key);
};

} // namespace dpct
} // namespace clang

#endif // DPCT_QUERY_API_MAPPING_H
