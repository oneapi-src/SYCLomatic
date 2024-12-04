//===--------------- AutoComplete.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Linux/AutoComplete.h"
#include "ErrorHandle/Error.h"
#include "Utility.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace clang {
namespace dpct {

static std::map<std::string, std::set<std::string>> DPCTOptionInfoMap = {
    // To avoid make llvm library depends on this file, hard code 2 options
    // here.
    {"help",
     {"basic", "advanced", "code-gen", "report-gen", "build-script",
      "query-api", "warnings", "help-info", "intercept-build", "examples"}},
    {"version", {}},

#define DPCT_OPTION_ENUM_VALUE(NAME, ...) NAME
#define DPCT_OPTION_VALUES(...)                                                \
  { __VA_ARGS__ }
#define DPCT_OPTION(TEMPLATE, TYPE, NAME, OPTION_CLASS, OPTION_ACTIONS,        \
                    OPTION_NAME, ...)                                          \
  {OPTION_NAME, {}},
#define DPCT_ENUM_OPTION(TEMPLATE, TYPE, NAME, OPTION_CLASS, OPTION_ACTIONS,   \
                         OPTION_NAME, OPTION_VALUES, ...)                      \
  {OPTION_NAME, OPTION_VALUES},
#define DPCT_SOURCEPATH_OPTION(...)
#define DPCT_HIDDEN_OPTION(...)
#define DPCT_ALIASE(ALIASE_FOR, OPTION_NAME, ...) {OPTION_NAME, {}},

#include "clang/DPCT/DPCTOptions.inc"
};

void AutoCompletePrinter::process() {
  llvm::StringRef InputRef(Input);
  LastSharpPos = InputRef.find_last_of('#');
  if (LastSharpPos == llvm::StringRef::npos ||
      LastSharpPos == (InputRef.size() - 1)) {
    // No char after last '#', it means user press tab after a space,
    // return empty line to show files/folders
    printAndExit();
  }

  bool SuggestEnumValue =
      (InputRef.find("=", LastSharpPos) != llvm::StringRef::npos) ||
      (LastSharpPos >= 1 && InputRef[LastSharpPos - 1] == '=');
  if (!SuggestEnumValue) {
    // suggest option names
    llvm::StringRef OptionPrefixRef = InputRef.substr(LastSharpPos + 1);

    // Case0: "" : return empty.
    // Case1: "-" : suggest all options. All candidates have "--" prefix except "-p".
    // Case2: "--" : suggest all options except -p. All candidates have "--" prefix.
    // Case3: "-abc" : suggest options have abc prefix. All candidates have "-" prefix.
    // Case4: "--abc" : suggest options have abc prefix. All candidates have "--" prefix.

    if (OptionPrefixRef.size() == 0) {
      // Case0
      printAndExit();
    } else if (OptionPrefixRef.size() == 1 && OptionPrefixRef[0] == '-') {
      // Case1
      addSuggestions(OptionSet, false);
      printAndExit();
    } else if (OptionPrefixRef.size() == 2 && OptionPrefixRef[0] == '-' &&
        OptionPrefixRef[1] == '-') {
      // Case2
      addSuggestions(OptionSet, true);
      printAndExit();
    } else if (OptionPrefixRef.size() >= 2 && OptionPrefixRef[0] == '-' &&
        OptionPrefixRef[1] != '-') {
      // Case3
      addSuggestions(OptionSet, OptionPrefixRef.substr(1), "-");
      printAndExit();
    } else if (OptionPrefixRef.size() >= 3 && OptionPrefixRef[0] == '-' &&
        OptionPrefixRef[1] == '-') {
      // Case4
      addSuggestions(OptionSet, OptionPrefixRef.substr(2), "--");
      printAndExit();
    }
    printAndExit();
  }

  // suggest enum values
  // E.g., "--foobar=abc,def,g"
  // CurOptionNameRef is "foobar="
  // EnumPrefixRef is "g"
  // ResultPrefixRef is "abc,def,"
  llvm::StringRef CurOptionNameRef;
  llvm::StringRef EnumPrefixRef;
  llvm::StringRef ResultPrefixRef;

  auto EqualPos = InputRef.find("=", LastSharpPos);
  if (EqualPos != llvm::StringRef::npos) {
    // InputRef: #--foo=
    auto OptionNameStartPos = InputRef.find_first_not_of('-', LastSharpPos + 1);
    CurOptionNameRef = InputRef.substr(OptionNameStartPos,
                                       InputRef.size() - OptionNameStartPos);
    addSuggestions(OptionEnumsMap[CurOptionNameRef.str()], "", "");
    printAndExit();
  }
  // InputRef: #--foo=#...
  auto SharpPosBeforeLastSharp =
      InputRef.substr(0, LastSharpPos).find_last_of('#');
  auto OptionNameStartPos =
      InputRef.find_first_not_of('-', SharpPosBeforeLastSharp + 1);
  CurOptionNameRef =
      InputRef.substr(OptionNameStartPos, LastSharpPos - OptionNameStartPos);
  auto StrAfterLastSharp = InputRef.substr(LastSharpPos + 1);
  auto LastCommaPosAfterLastSharp = StrAfterLastSharp.find_last_of(',');
  if (LastCommaPosAfterLastSharp != llvm::StringRef::npos) {
    // InputRef: #--foo=#abc,def,g
    EnumPrefixRef = StrAfterLastSharp.substr(LastCommaPosAfterLastSharp + 1);
    ResultPrefixRef =
        StrAfterLastSharp.substr(0, LastCommaPosAfterLastSharp + 1);
    addSuggestions(OptionEnumsMap[CurOptionNameRef.str()], EnumPrefixRef, ResultPrefixRef);
    printAndExit();
  } else {
    // InputRef: #--foo=#ab
    EnumPrefixRef = StrAfterLastSharp;
    addSuggestions(OptionEnumsMap[CurOptionNameRef.str()], EnumPrefixRef, "");
    printAndExit();
  }
}

void AutoCompletePrinter::operator=(std::string RawInput) {
  for (const auto &I : DPCTOptionInfoMap) {
    if (I.second.empty()) {
      OptionSet.insert(I.first);
    } else {
      const std::string Key = I.first + "=";
      OptionSet.insert(Key);
      for (const auto &Enum : I.second) {
        OptionEnumsMap[Key].insert(Enum);
      }
    }
  }
  Input = "#" + RawInput;
  process();
}

void AutoCompletePrinter::printAndExit() {
  if (Suggestions.empty()) {
    llvm::outs() << "\n";
    dpctExit(MigrationSucceeded);
  }

  for (const auto &Opt : Suggestions) {
    llvm::outs() << Opt << "\n";
  }
  dpctExit(MigrationSucceeded);
}

} // namespace dpct
}
