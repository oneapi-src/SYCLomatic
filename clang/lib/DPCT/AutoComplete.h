//===--------------- AutoComplete.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_AUTOCOMPLETE_H
#define DPCT_AUTOCOMPLETE_H

#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <set>

namespace clang {
namespace dpct {
class AutoCompletePrinter {
  std::set<std::string> Suggestions;
  std::vector<std::string> OptionList;
  std::map<std::string, std::set<std::string>> OptionEnumsMap;
  std::string Input;
  void printAndExit();
  void process();
  std::string::size_type LastSharpPos;
public:
  AutoCompletePrinter() {}
  void operator=(std::string RawInput);
};
}
}

#endif
