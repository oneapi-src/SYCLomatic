//===--------------- AutoComplete.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_AUTOCOMPLETE_H
#define DPCT_AUTOCOMPLETE_H

#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <set>

namespace clang {
namespace dpct {
class AutoCompletePrinter {
  std::set<std::string> Suggestions;
  std::set<std::string> OptionSet;
  std::map<std::string, std::set<std::string>> OptionEnumsMap;
  std::string Input;
  void printAndExit();
  void process();
  std::string::size_type LastSharpPos;
  void addSuggestions(const std::set<std::string>& Set,
                      const llvm::StringRef Prefix,
                      const llvm::StringRef ResultPrefix) {
    for (const auto &Item : Set) {
      llvm::StringRef ItemRef(Item);
      if (ItemRef.starts_with(Prefix)) {
        Suggestions.insert(ResultPrefix.str() + Item);
      }
    }
  }
  void addSuggestions(const std::set<std::string>& Set,
                      bool MustBeginWithDashDash) {
    for (const auto &Opt : OptionSet) {
      if (Opt.size() == 1 && !MustBeginWithDashDash) {
        Suggestions.insert("-" + Opt);
      } else {
        Suggestions.insert("--" + Opt);
      }
    }
  }
public:
  AutoCompletePrinter() {}
  void operator=(std::string RawInput);
};
}
}

#endif
