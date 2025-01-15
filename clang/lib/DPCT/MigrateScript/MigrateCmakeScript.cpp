//===--------------- MigrateCmakeScript.cpp--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "MigrateCmakeScript.h"
#include "Diagnostics/Diagnostics.h"
#include "ErrorHandle/Error.h"
#include "FileGenerator/GenFiles.h"
#include "MigrationReport/Statics.h"
#include "UserDefinedRules/PatternRewriter.h"
#include "Utility.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>

using namespace clang::dpct;
using namespace llvm::cl;

namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

namespace clang {
namespace dpct {
std::map<std::string /*CMake command*/,
         std::tuple<bool /*ProcessedOrNot*/, bool /*CUDASpecificOrNot*/>>
    cmake_commands{
#define ENTRY_TYPE(TYPENAME, VALUE1, COMMENT, VALUE2)                          \
  {#TYPENAME, {VALUE1, VALUE2}},
#include "CMakeCommands.inc"
#undef ENTRY_TYPE
    };

static std::map<clang::tooling::UnifiedPath /*file path*/,
                std::string /*content*/>
    CmakeScriptFileBufferMap;
static std::map<clang::tooling::UnifiedPath /*file name*/, bool /*is crlf*/>
    ScriptFileCRLFMap;
static std::map<std::string /*variable name*/, std::string /*value*/>
    CmakeVarMap;

static std::map<std::string /*cmake syntax*/,
                MetaRuleObject::PatternRewriter /*cmake migration rule*/>
    CmakeBuildInRules;

static std::map<std::string /*file path*/,
                std::vector<std::string> /*warning msg*/>
    FileWarningsMap;

void cmakeSyntaxProcessed(std::string &Input);

static size_t skipWhiteSpaces(const std::string Input, size_t Index) {
  size_t Size = Input.size();
  for (; Index < Size && isWhitespace(Input[Index]); Index++) {
  }
  return Index;
}

static size_t gotoEndOfCmakeWord(const std::string Input, size_t Index,
                                 char Delim) {
  size_t Size = Input.size();
  for (; Index < Size && !isWhitespace(Input[Index]) && Input[Index] != Delim;
       Index++) {
  }
  return Index;
}

static size_t gotoEndOfCmakeCommand(const std::string Input, size_t Index) {
  size_t Size = Input.size();
  for (; Index < Size && !isWhitespace(Input[Index]) && Input[Index] != '(';
       Index++) {
  }
  if (Index < Size && Input[Index] == '(') {
    return Index;
  }
  return std::string::npos;
}

static size_t gotoEndOfCmakeCommandStmt(const std::string Input, size_t Index) {
  size_t Size = Input.size();
  for (; Index < Size && Input[Index] != ')'; Index++) {
  }
  return Index;
}

std::string getVarName(const std::string &Variable) {
  std::string Value;
  if (Variable[0] == '$' && Variable[1] == '{') {
    auto Name = Variable.substr(2, Variable.size() - 3);

    auto Iter = CmakeVarMap.find(Name);
    if (Iter != CmakeVarMap.end()) {
      Value = Iter->second;
    }
  } else {
    Value = Variable;
  }

  return Value;
}

static void parseVariable(const std::string &Input) {

  const int Size = Input.size();
  int Index = 0;
  while (Index < Size) {
    // Skip comments
    if (Input[Index] == '#') {
      for (; Index < Size && Input[Index] != '\n'; Index++) {
      }
    }
    int Begin, End;
    // Go the begin of cmake command
    Index = skipWhiteSpaces(Input, Index);
    Begin = Index;

    // Go the end of cmake command
    Index = gotoEndOfCmakeWord(Input, Begin + 1, '(');
    End = Index;

    // Skip possible space
    Index = skipWhiteSpaces(Input, Index);

    if (Index < Size && Input[Index] == '(') {
      std::string Command = Input.substr(Begin, End - Begin);
      std::transform(Command.begin(), Command.end(), Command.begin(),
                     [](unsigned char c) { return std::tolower(c); });

      if (Command == "set") {
        Index++; // Skip '('
        std::string VarName;

        // Get the begin of first argument of set
        Index = skipWhiteSpaces(Input, Index);
        Begin = Index;

        // Get the end of the first argument of set
        Index = gotoEndOfCmakeWord(Input, Index, ')');
        End = Index;

        // Get the name of the first argument
        VarName = Input.substr(Begin, End - Begin);

        // Get the begin of the second argument
        for (Index = End + 1; Index < Size && isWhitespace(Input[Index]);
             Index++) {
        }
        Begin = Index;

        // Get the end of the second argument
        Index = gotoEndOfCmakeWord(Input, Begin + 1, ')');
        End = Index;

        // Only check the set commands which has two arguments.
        // And skip cmake reserves identifiers that begins with "CMAKE_" and
        // "_CMAKE_".
        if (Input[Index] == ')' &&
            !llvm::StringRef(VarName).starts_with("CMAKE_") &&
            !llvm::StringRef(VarName).starts_with("_CMAKE_ ")) {
          // Get the name of the second argument
          CmakeVarMap[VarName] = Input.substr(Begin, End - Begin);
        }
      }

      // Go the ')' of cmake command
      Index = gotoEndOfCmakeCommandStmt(Input, Index);
    }

    Index++;
  }
}

static std::string processArgOfCmakeVersionRequired(
    const std::string &Arg,
    const std::map<std::string, std::string> &CmakeVarMap) {
  const std::string CmakeMinVerion =
      "3.24"; // The minimal version used in SYCL compiler
  size_t Pos = Arg.find("...");
  std::string ReplArg;
  if (Pos != std::string::npos) {
    const auto StrArray = split(Arg, "...");

    std::string MinVer = getVarName(StrArray[0]);
    std::string MaxVer = getVarName(StrArray[1]);

    if (std::atof(MinVer.c_str()) >= std::atof(CmakeMinVerion.c_str())) {
      ReplArg = MinVer + "..." + MaxVer;
    } else if (std::atof(MinVer.c_str()) < std::atof(CmakeMinVerion.c_str()) &&
               std::atof(MaxVer.c_str()) > std::atof(CmakeMinVerion.c_str())) {
      ReplArg = CmakeMinVerion + "..." + MaxVer;
    } else {
      ReplArg = std::move(CmakeMinVerion);
    }

  } else {
    std::string Ver = getVarName(Arg);
    if (std::atof(Ver.c_str()) < std::atof(CmakeMinVerion.c_str())) {
      ReplArg = std::move(CmakeMinVerion);
    } else {
      ReplArg = std::move(Ver);
    }
  }
  return ReplArg;
}

static bool skipCmakeComments(const std::string &Input, size_t &Index) {
  const size_t Size = Input.size();
  bool CommentFound = false;
  if (Input[Index] == '#') {
    CommentFound = true;
    for (; Index < Size && Input[Index] != '\n'; Index++) {
    }
    Index++;
  }
  return CommentFound;
}

void processCmakeMinimumRequired(std::string &Input, size_t &Size,
                                 size_t &Index) {
  std::string VarName;
  std::string Value;
  size_t Begin, End;

  // Get the begin of first argument of cmake_minimum_required
  Index = skipWhiteSpaces(Input, Index);
  Begin = Index;

  // Get the end of the first argument of cmake_minimum_required
  Index = gotoEndOfCmakeWord(Input, Index, ')');
  End = Index;

  // Get the name of the first argument
  VarName = Input.substr(Begin, End - Begin);

  // Get the begin of the second argument
  Index = skipWhiteSpaces(Input, End + 1);
  Begin = Index;

  // Get the end of the second argument
  Index = gotoEndOfCmakeWord(Input, Begin + 1, ')');
  End = Index;

  // Get the name of the second argument
  Value = Input.substr(Begin, End - Begin);

  std::string ReplStr = processArgOfCmakeVersionRequired(Value, CmakeVarMap);

  Input.replace(Begin, End - Begin, ReplStr);
  Size = Input.size();            // Update string size
  Index = Begin + ReplStr.size(); // Update index
}

void processExecuteProcess(std::string &Input, size_t &Size, size_t &Index) {
  std::string VarName;
  std::string Value;
  size_t Begin, End;

  // Get the begin of first argument
  Index = skipWhiteSpaces(Input, Index);
  Begin = Index;

  Index = gotoEndOfCmakeCommandStmt(Input, Index);
  End = Index;

  // Get the value of the second argument
  Value = Input.substr(Begin, End - Begin);

  size_t Pos = Value.find("-Xcompiler");
  if (Pos != std::string::npos) {
    size_t NextPos = Pos + strlen("-Xcompiler");
    NextPos = skipWhiteSpaces(Value, NextPos);

    // clang-format off
    // To check if the value of opition "-Xcompiler" is a string literal, if it
    // is a string literal, just remove '"' for outmost '"' and '\\' for inner string, like:
    // -Xcompiler "-dumpfullversion" -> -Xcompiler -dumpfullversion
    // -Xcompiler "\"-dumpfullversion \"" -> -Xcompiler -dumpfullversion
    // clang-format on
    if (Value[NextPos] == '"') {
      Value[NextPos] = ' ';
      size_t Size = Value.size();
      size_t Idx = NextPos;
      for (; Idx < Size; Idx++) {
        if (Value[Idx] == '\\') {
          Value[Idx] = ' ';
        } else if (Value[Idx] == '"') {
          Value[Idx] = ' ';
        }
      }
    }
  }

  Input.replace(Begin, End - Begin, Value);
  Size = Input.size();          // Update string size
  Index = Begin + Value.size(); // Update index
}

void processCudaCompile(std::string &Input, size_t &Size, size_t &Index) {
  size_t Begin, End;

  // Get the beginning of the first argument
  Index = skipWhiteSpaces(Input, Index);
  Begin = Index;

  // Get the end of the Cmake Stmt
  Index = gotoEndOfCmakeCommandStmt(Input, Index);
  End = Index;

  // Extract the content of the command
  std::string Content = Input.substr(Begin, End - Begin);

  // Locate "OPTIONS" in the command
  size_t OptionsPos = Content.find("OPTIONS");
  if (OptionsPos != std::string::npos) {
    // Locate Options Begin and End
    size_t Pos = Content.find("OPTIONS", OptionsPos) + strlen("OPTIONS");
    size_t OptionsBegin = skipWhiteSpaces(Content, Pos);
    size_t OptionsEnd = gotoEndOfCmakeCommandStmt(Content, Pos);

    // Split by Whitespaces and Filter
    std::string Options =
        Content.substr(OptionsBegin, OptionsEnd - OptionsBegin);
    std::istringstream Stream(Options);
    std::string Token;
    std::vector<std::string> FilteredOptions;

    // Find the Options starting with "-D"
    while (Stream >> Token) {
      if (Token.rfind("-D", 0) == 0) { // Keep only options that start with "-D"
        FilteredOptions.push_back(Token);
      }
    }

    // Create a new  replacement string with the filtered options
    std::string FilteredOptionsString;
    if (!FilteredOptions.empty()) {
      FilteredOptionsString =
          "OPTIONS " + std::accumulate(std::next(FilteredOptions.begin()),
                                       FilteredOptions.end(),
                                       FilteredOptions[0],
                                       [](std::string a, std::string b) {
                                         return a + " " + b;
                                       });
    } else {
      // If there are no options, remove the OPTIONS keyword
      OptionsPos--;
    }

    Content.replace(OptionsPos, OptionsEnd - OptionsPos, FilteredOptionsString);
    Input.replace(Begin, End - Begin, Content);
    Size = Input.size();            // Update the size of the input
    Index = Begin + Content.size(); // Update the index
  }
}

// Implicit migration rule is used when the migration logic is difficult to be
// described with yaml based rule syntax. Currently only migration of
// cmake_minimum_required() is implemented by implicit migration rule.
void applyImplicitMigrationRule(std::string &Input,
                                const std::string &BuildScriptSyntax,
                                void (*Func)(std::string &, size_t &,
                                             size_t &)) {

  size_t Size = Input.size();
  size_t Index = 0;
  while (Index < Size) {
    Index = skipWhiteSpaces(Input, Index);
    // Skip comments
    if (skipCmakeComments(Input, Index)) {
      continue;
    }

    size_t Begin, End;
    // Go the begin of cmake command
    Index = skipWhiteSpaces(Input, Index);
    Begin = Index;

    // Go the end of cmake command
    Index = gotoEndOfCmakeWord(Input, Begin + 1, '(');
    End = Index;

    // Skip possible space
    Index = skipWhiteSpaces(Input, Index);

    if (Index < Size && Input[Index] == '(') {
      std::string Command = Input.substr(Begin, End - Begin);

      // Process implict cmake syntax
      if (Command == BuildScriptSyntax) {
        (*Func)(Input, Size, Index);
      }

      // Go the ')' of cmake command
      Index = gotoEndOfCmakeCommandStmt(Input, Index);
    }

    Index++;
  }
}

static std::string convertCmakeCommandsToLower(const std::string &InputString,
                                               const std::string FileName) {
  std::stringstream OutputStream;

  const auto Lines = split(InputString, '\n');

  std::vector<std::string> Output;
  unsigned int Count = 1;
  for (auto Line : Lines) {

    size_t Size = Line.size();
    size_t Index = 0;

    // Go the begin of cmake command
    Index = skipWhiteSpaces(Line, Index);
    int Begin = Index;

    // Go the end of cmake command
    Index = gotoEndOfCmakeCommand(Line, Begin + 1);
    int End = Index;
    if (Index < Size && (Line[Index] == '(' || isWhitespace(Line[Index]))) {
      std::string Str = Line.substr(Begin, End - Begin);
      std::transform(Str.begin(), Str.end(), Str.begin(),
                     [](unsigned char Char) { return std::tolower(Char); });
      auto Iter = cmake_commands.find(Str);
      if (Iter != cmake_commands.end()) {
        for (int Idx = Begin; Idx < End; Idx++) {
          Line[Idx] = Str[Idx - Begin];
        }

        if (!std::get<0>(Iter->second) && !std::get<1>(Iter->second)) {
          std::string WarningMsg =
              FileName + ":" + std::to_string(Count) + ":warning:";
          WarningMsg += DiagnosticsUtils::getMsgText(
              CMakeScriptMigrationMsgs::CMAKE_CONFIG_FILE_WARNING, Str);
          WarningMsg += "\n";
          FileWarningsMap[FileName].push_back(std::move(WarningMsg));

          OutputStream
              << "# "
              << DiagnosticsUtils::getMsgText(
                     CMakeScriptMigrationMsgs::CMAKE_CONFIG_FILE_WARNING, Str)
              << "\n";
        }

        if (!std::get<0>(Iter->second) && std::get<1>(Iter->second)) {
          std::string WarningMsg =
              FileName + ":" + std::to_string(Count) + ":warning:";
          WarningMsg += DiagnosticsUtils::getMsgText(
              CMakeScriptMigrationMsgs::CMAKE_NOT_SUPPORT_WARNING, Str);
          WarningMsg += "\n";
          FileWarningsMap[FileName].push_back(std::move(WarningMsg));

          OutputStream
              << "# "
              << DiagnosticsUtils::getMsgText(
                     CMakeScriptMigrationMsgs::CMAKE_NOT_SUPPORT_WARNING, Str)
              << "\n";
        }
      }
    }

    OutputStream << Line << "\n";
    Count++;
  }

  return OutputStream.str();
}

void addCmakeWarningMsg(const std::string &WarningMsg,
                        const std::string FileName) {
  FileWarningsMap[FileName].push_back(WarningMsg);
}

static void doCmakeScriptAnalysis() {

  for (auto &Entry : CmakeScriptFileBufferMap) {
    auto &Buffer = Entry.second;

    // Collect varible name and its value
    parseVariable(Buffer);
  }
}

static void convertAllCmakeCommandsToLowerCase() {
  for (auto &Entry : CmakeScriptFileBufferMap) {
    auto &Buffer = Entry.second;

    // Convert cmake command to lower case in cmake script files
    Buffer = convertCmakeCommandsToLower(Buffer, Entry.first.getPath().str());
  }
}

static void
applyCmakeMigrationRules(const clang::tooling::UnifiedPath InRoot,
                         const clang::tooling::UnifiedPath OutRoot) {

  static const std::map<std::string,
                        void (*)(std::string &, size_t &, size_t &)>
      DispatchTable = {
          {"cmake_minimum_required", processCmakeMinimumRequired},
          {"execute_process", processExecuteProcess},
          {"cuda_compile", processCudaCompile},
          {"cuda_compile_fatbin", processCudaCompile},
      };

  setFileTypeProcessed(SourceFileType::SFT_CMakeScript);

  for (auto &Entry : CmakeScriptFileBufferMap) {
    llvm::outs() << "Processing: " + Entry.first.getPath() + "\n";

    auto &Buffer = Entry.second;
    clang::tooling::UnifiedPath FileName = Entry.first.getPath();

    // Apply user define migration rules
    for (const auto &CmakeSyntaxEntry : CmakeBuildInRules) {
      const auto &PR = CmakeSyntaxEntry.second;
      if (PR.In.empty() && PR.Out.empty()) {
        // Implicit migration rule is used when the migration logic is difficult
        // to be described with yaml based rule syntax. Currently only migration
        // of cmake_minimum_required() is implemented by implicit migration
        // rule.
        applyImplicitMigrationRule(Buffer, PR.BuildScriptSyntax,
                                   DispatchTable.at(PR.BuildScriptSyntax));

      } else {
        if (PR.RuleId == "rule_project") {
          auto NewPR = PR;
          SmallString<512> RelativePath(FileName.getCanonicalPath());
          llvm::sys::path::replace_path_prefix(RelativePath,
                                               OutRoot.getCanonicalPath(), ".");

#ifdef _WIN32
          std::vector<std::string> SplitedStr =
              split(RelativePath.c_str(), '\\');
#else
          std::vector<std::string> SplitedStr =
              split(RelativePath.c_str(), '/');
#endif
          std::string RelativePathPrefix = "";

          auto Size = SplitedStr.size();
          for (size_t Idx = 0; Size > 2 && Idx < Size - 2; Idx++) {
            RelativePathPrefix += "../";
          }
          RelativePathPrefix += "dpct.cmake";
          NewPR.Out += "include(" + RelativePathPrefix + ")\n";
          Buffer = applyPatternRewriter(
              NewPR, Buffer, Entry.first.getPath().str(), "", OutRoot);
        } else {
          Buffer = applyPatternRewriter(PR, Buffer, Entry.first.getPath().str(),
                                        "", OutRoot);
        }
      }
    }

    auto Iter = FileWarningsMap.find(FileName.getPath().str());
    if (Iter != FileWarningsMap.end()) {
      std::vector WarningsVec = Iter->second;
      for (auto &Warning : WarningsVec) {
        llvm::outs() << Warning;
      }
    }
  }
}

// cmake systaxes need to be processed by implicit migration rules, as they are
// difficult to be described with yaml based rule syntax.
static const std::vector<std::string> ImplicitMigrationRules = {
    "cmake_minimum_required", "execute_process", "cuda_compile",
    "cuda_compile_fatbin"};

static void reserveImplicitMigrationRules() {
  for (const auto &Rule : ImplicitMigrationRules) {
    MetaRuleObject::PatternRewriter PrePR;
    PrePR.BuildScriptSyntax = Rule;
    CmakeBuildInRules[PrePR.BuildScriptSyntax] = PrePR;
  }
}

void doCmakeScriptMigration(const clang::tooling::UnifiedPath &InRoot,
                            const clang::tooling::UnifiedPath &OutRoot) {
  loadBufferFromFile(InRoot, OutRoot, CmakeScriptFileBufferMap);
  unifyInputFileFormat(CmakeScriptFileBufferMap, ScriptFileCRLFMap);
  convertAllCmakeCommandsToLowerCase();
  reserveImplicitMigrationRules();
  doCmakeScriptAnalysis();
  applyCmakeMigrationRules(InRoot, OutRoot);
  storeBufferToFile(CmakeScriptFileBufferMap, ScriptFileCRLFMap);
}

void registerCmakeMigrationRule(MetaRuleObject &R) {
  auto PR = MetaRuleObject::PatternRewriter(R.In, R.Out, R.Subrules,
                                            R.MatchMode, R.Warning, R.RuleId,
                                            R.BuildScriptSyntax, R.Priority);
  auto Iter = CmakeBuildInRules.find(PR.BuildScriptSyntax);
  if (Iter != CmakeBuildInRules.end()) {
    if (PR.Priority == RulePriority::Takeover &&
        Iter->second.Priority > PR.Priority) {
      CmakeBuildInRules[PR.BuildScriptSyntax] = PR;
    } else {
      llvm::outs() << "[Warnning]: Two migration rules (Rule:" << R.RuleId
                   << ", Rule:" << Iter->second.RuleId
                   << ") are duplicated, the migration rule (Rule:" << R.RuleId
                   << ") is ignored.\n";
    }
  } else {
    CmakeBuildInRules[PR.BuildScriptSyntax] = PR;
  }
}

} // namespace dpct
} // namespace clang