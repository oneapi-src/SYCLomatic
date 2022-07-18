//===--------------- AutoComplete.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AutoComplete.h"
#include "Error.h"
#include "Utility.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>

namespace clang {
namespace dpct {

static std::map<std::string, std::set<std::string>> DPCTOptionInfoMap = {
#define DPCT_OPTIONS_IN_LLVM_SUPPORT
#define DPCT_OPTIONS_IN_CLANG_TOOLING
#define DPCT_OPTIONS_IN_CLANG_DPCT

#define DPCT_OPT_TYPE(...) 0
#define DPCT_OPT_ENUM(NAME, ...)  NAME
#define DPCT_OPTION_VALUES(...)   {__VA_ARGS__}
#define DPCT_NON_ENUM_OPTION(OPT_TYPE, OPT_VAR, OPTION_NAME, ...)                 \
{OPTION_NAME, {}},
#define DPCT_ENUM_OPTION(OPT_TYPE, OPT_VAR, OPTION_NAME, OPTION_VALUES, ...)      \
{OPTION_NAME, OPTION_VALUES},

#include "llvm/DPCT/DPCTOptions.inc"

#undef DPCT_ENUM_OPTION
#undef DPCT_NON_ENUM_OPTION
#undef DPCT_OPT_ENUM
#undef DPCT_OPT_TYPE

#undef DPCT_OPTIONS_IN_CLANG_DPCT
#undef DPCT_OPTIONS_IN_CLANG_TOOLING
#undef DPCT_OPTIONS_IN_LLVM_SUPPORT
};

void AutoCompletePrinter::process() {
  LastSharpPos = Input.find_last_of('#');
  if (LastSharpPos == (Input.size() - 1)) {
    // No char after last '#', it means user press tab after a space,
    // return empty line to show files/folders
    printAndExit();
  }

  bool SuggestEnumValue = (Input.find("=", LastSharpPos) != std::string::npos) || (LastSharpPos >= 1 && Input[LastSharpPos - 1] == '=');
  if (!SuggestEnumValue) {
    // suggest option names
    std::string OptionPrefix = Input.substr(LastSharpPos + 1);
    for (auto Opt : OptionList) {
      llvm::StringRef OptRef(Opt);
      if (OptRef.startswith(OptionPrefix)) {
        Suggestions.insert(Opt);
      }
    }
    printAndExit();
  }

  // suggest enum values
  // E.g., "--foobar=abc,def,g"
  // CurOptionName is "--foobar="
  // EnumPrefix is "g"
  // ResultPrefix is "abc,def,"
  std::string CurOptionName;
  std::string EnumPrefix;
  std::string ResultPrefix;

  auto EqualPos = Input.find("=", LastSharpPos);
  if (EqualPos != std::string::npos) {
    // Input: #--foo=
    CurOptionName = Input.substr(LastSharpPos + 1);
    EnumPrefix = "";
    for (auto Enum : OptionEnumsMap[CurOptionName]) {
      llvm::StringRef EnumRef(Enum);
      if (EnumRef.startswith(EnumPrefix)) {
        Suggestions.insert(Enum);
      }
    }
    printAndExit();
  }
  // Input: #--foo=#...
  auto SharpPosBeforeLastSharp = Input.substr(0, LastSharpPos).find_last_of('#');
  std::string CurOpt = Input.substr(SharpPosBeforeLastSharp + 1, LastSharpPos - SharpPosBeforeLastSharp - 1);
  auto StrAfterLastSharp = Input.substr(LastSharpPos + 1);
  auto LastCommaPosAfterLastSharp = StrAfterLastSharp.find_last_of(',');
  if (LastCommaPosAfterLastSharp != std::string::npos) {
    // Input: #--foo=#abc,def,g
    CurOptionName = CurOpt;
    EnumPrefix = StrAfterLastSharp.substr(LastCommaPosAfterLastSharp + 1);
    ResultPrefix = StrAfterLastSharp.substr(0, LastCommaPosAfterLastSharp + 1);
    for (auto Enum : OptionEnumsMap[CurOptionName]) {
      llvm::StringRef EnumRef(Enum);
      if (EnumRef.startswith(EnumPrefix)) {
        Suggestions.insert(ResultPrefix + Enum);
      }
    }
    printAndExit();
  } else {
    // Input: #--foo=#ab
    CurOptionName = CurOpt;
    EnumPrefix = StrAfterLastSharp;
    for (auto Enum : OptionEnumsMap[CurOptionName]) {
      llvm::StringRef EnumRef(Enum);
      if (EnumRef.startswith(EnumPrefix)) {
        Suggestions.insert(Enum);
      }
    }
    printAndExit();
  }
}

void AutoCompletePrinter::operator=(std::string RawInput) {
  for (const auto& I : DPCTOptionInfoMap) {
    std::string OptStr;
    if (I.first.size() == 1) {
      OptStr = "-" + I.first;
    } else {
      OptStr = "--" + I.first;
    }

    if (I.second.empty()) {
      OptionList.push_back(OptStr);
    } else {
      for (const auto& Enum : I.second) {
        OptionEnumsMap[OptStr + "="].insert(Enum);
        OptionList.push_back(OptStr + "=");
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

  for (auto Opt : Suggestions) {
    llvm::outs() << Opt << "\n";
  }
  dpctExit(MigrationSucceeded);
}

}
}
