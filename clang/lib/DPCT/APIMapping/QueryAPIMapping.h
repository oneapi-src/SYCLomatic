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

class APIMapping {
  static std::unordered_map<std::string, std::string> EntryMap;

  static void registerEntry(const std::string &Name,
                            const std::string &Description);

public:
  static void initEntryMap();

  static const std::string &getAPISourceCode(const std::string &Key);
};

} // namespace dpct
} // namespace clang

#endif // DPCT_QUERY_API_MAPPING_H
