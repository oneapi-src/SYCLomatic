//===--------------- QueryAPIMapping.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_QUERY_API_MAPPING_H
#define DPCT_QUERY_API_MAPPING_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <unordered_map>

namespace clang {
namespace dpct {

class APIMapping {
  static llvm::DenseMap<llvm::StringRef, llvm::StringRef> EntryMap;

  static void registerEntry(llvm::StringRef Name, llvm::StringRef Description);

public:
  static void initEntryMap();

  static llvm::StringRef getAPISourceCode(llvm::StringRef Key);
};

} // namespace dpct
} // namespace clang

#endif // DPCT_QUERY_API_MAPPING_H
