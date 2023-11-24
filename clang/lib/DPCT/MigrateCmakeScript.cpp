//===--------------- MigrateCmakeScript.cpp--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "MigrateCmakeScript.h"
#include "SaveNewFiles.h"
#include "Statics.h"
#include "Utility.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>

using namespace clang::dpct;
using namespace llvm::cl;

namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

const std::unordered_map<std::string /*command*/, bool /*need lower*/>
    cmake_commands = {
        {"cmake_minimum_required", 1},
        {"cmake_parse_arguments", 1},
        {"cmake_path", 1},
        {"cmake_policy", 1},
        {"file", 1},
        {"find_file", 1},
        {"find_library", 1},
        {"find_package", 1},
        {"find_path", 1},
        {"find_program", 1},
        {"foreach", 1},
        {"function", 1},
        {"get_cmake_property", 1},
        {"get_directory_property", 1},
        {"get_filename_component", 1},
        {"get_property", 1},
        {"list", 1},
        {"macro", 1},
        {"mark_as_advanced", 1},
        {"message", 1},
        {"separate_arguments", 1},
        {"set", 1},
        {"set_directory_properties", 1},
        {"set_property", 1},
        {"string", 1},
        {"unset", 1},
        {"add_compile_definitions", 1},
        {"add_compile_options", 1},
        {"add_custom_command", 1},
        {"add_custom_target", 1},
        {"add_definitions", 1},
        {"add_dependencies", 1},
        {"add_executable", 1},
        {"add_library", 1},
        {"add_link_options", 1},
        {"add_subdirectory", 1},
        {"add_test", 1},
        {"build_command", 1},
        {"define_property", 1},
        {"include_directories", 1},
        {"install", 1},
        {"link_directories", 1},
        {"link_libraries", 1},
        {"project", 1},
        {"set_source_files_properties", 1},
        {"set_target_properties", 1},
        {"set_tests_properties", 1},
        {"source_group", 1},
        {"target_compile_definitions", 1},
        {"target_compile_features", 1},
        {"target_compile_options", 1},
        {"target_include_directories", 1},
        {"target_link_directories", 1},
        {"target_link_libraries", 1},
        {"target_link_options", 1},
        {"target_sources", 1},
        {"try_compile", 1},
        {"try_run", 1},
        {"build_name", 1},
        {"exec_program", 1},
        {"export_library_dependencies", 1},
        {"make_directory", 1},
        {"remove", 1},
        {"subdir_depends", 1},
        {"subdirs", 1},
        {"use_mangled_mesa", 1},
        {"utility_source", 1},
        {"variable_requires", 1},
        {"write_file", 1},
        {"cuda_add_cufft_to_target", 1},
        {"cuda_add_cublas_to_target", 1},
        {"cuda_add_executable", 1},
        {"cuda_add_library", 1},
        {"cuda_build_clean_target", 1},
        {"cuda_compile", 1},
        {"cuda_compile_ptx", 1},
        {"cuda_compile_fatbin", 1},
        {"cuda_compile_cubin", 1},
        {"cuda_compute_separable_compilation_object_file_name", 1},
        {"cuda_include_directories", 1},
        {"cuda_link_separable_compilation_objects", 1},
        {"cuda_select_nvcc_arch_flags", 1},
        {"cuda_wrap_srcs", 1},

};

static std::string readFile(const std::string &Name) {
  std::ifstream Stream(Name, std::ios::in | std::ios::binary);
  std::string Contents((std::istreambuf_iterator<char>(Stream)),
                       (std::istreambuf_iterator<char>()));
  return Contents;
}

std::string getCmakeBuildPathFromInRoot(StringRef InRoot, StringRef OutRoot) {
  std::error_code EC;

  std::string CmakeBuildDirectory;
  for (fs::recursive_directory_iterator Iter(Twine(InRoot), EC), End;
       Iter != End; Iter.increment(EC)) {
    if ((bool)EC) {
      std::string ErrMsg = "[ERROR] Access : " + std::string(InRoot.str()) +
                           " fail: " + EC.message() + "\n";
      PrintMsg(ErrMsg);
    }

    auto FilePath = Iter->path();

    // Skip output directory if it is in the in-root directory.
    if (isChildOrSamePath(OutRoot.str(), FilePath))
      continue;

    bool IsHidden = false;
    for (path::const_iterator PI = path::begin(FilePath),
                              PE = path::end(FilePath);
         PI != PE; ++PI) {
      StringRef Comp = *PI;
      if (Comp.startswith(".")) {
        IsHidden = true;
        break;
      }
    }
    // Skip hidden folder or file whose name begins with ".".
    if (IsHidden) {
      continue;
    }

    if (Iter->type() == fs::file_type::directory_file) {
      const auto Path = Iter->path();
      SmallString<512> OutDirectory = llvm::StringRef(Path);
      if (fs::exists(OutDirectory + "/CMakeFiles") &&
          fs::exists(OutDirectory + "/CMakeCache.txt")) {
        CmakeBuildDirectory = OutDirectory.str().str();
        break;
      }
    }
  }
  return CmakeBuildDirectory;
}

void collectCmakeScripts(StringRef InRoot, StringRef OutRoot,
                         std::vector<std::string> &CmakeScriptFiles) {
  std::error_code EC;

  std::string CmakeBuildDirectory =
      getCmakeBuildPathFromInRoot(InRoot, OutRoot);
  for (fs::recursive_directory_iterator Iter(Twine(InRoot), EC), End;
       Iter != End; Iter.increment(EC)) {
    if ((bool)EC) {
      std::string ErrMsg = "[ERROR] Access : " + std::string(InRoot.str()) +
                           " fail: " + EC.message() + "\n";
      PrintMsg(ErrMsg);
    }

    auto FilePath = Iter->path();

    // Skip output directory if it is in the in-root directory.
    if (isChildOrSamePath(OutRoot.str(), FilePath))
      continue;

    // Skip cmake build directory if it is in the in-root directory.
    if (!CmakeBuildDirectory.empty() &&
        isChildOrSamePath(CmakeBuildDirectory, FilePath))
      continue;

    bool IsHidden = false;
    for (path::const_iterator PI = path::begin(FilePath),
                              PE = path::end(FilePath);
         PI != PE; ++PI) {
      StringRef Comp = *PI;
      if (Comp.startswith(".")) {
        IsHidden = true;
        break;
      }
    }
    // Skip hidden folder or file whose name begins with ".".
    if (IsHidden) {
      continue;
    }

    if (Iter->type() == fs::file_type::regular_file) {
      SmallString<512> OutputFile = llvm::StringRef(FilePath);

      llvm::StringRef Name = llvm::sys::path::filename(FilePath);
      if (Name == "CMakeLists.txt" || Name.ends_with(".cmake")) {
        CmakeScriptFiles.push_back(OutputFile.str().str());
      }
    }
  }
}

bool migrateCmakeScriptFile(StringRef InRoot, StringRef OutRoot,
                            std::string InFileName) {
  makeCanonical(InFileName);
  SmallString<512> OutFileName = llvm::StringRef(InFileName);
  if (!rewriteDir(OutFileName, InRoot, OutRoot)) {
    return false;
  }
  auto Parent = path::parent_path(OutFileName);
  std::error_code EC;
  EC = fs::create_directories(Parent);
  if ((bool)EC) {
    std::string ErrMsg =
        "[ERROR] Create Directory : " + std::string(Parent.str()) +
        " fail: " + EC.message() + "\n";
    PrintMsg(ErrMsg);
  }
  std::ofstream Out(OutFileName.c_str(), std::ios::binary);
  if (Out.fail()) {
    std::string ErrMsg =
        "[ERROR] Create file : " + std::string(OutFileName.c_str()) +
        " failure!\n";
    PrintMsg(ErrMsg);
  }

  llvm::raw_os_ostream Stream(Out);
  applyPatternRewriterToCmakeScriptFile(readFile(InFileName), Stream);

  Stream.flush();
  Out.close();
  return true;
}

bool cmakeScriptFileSpecified(const std::vector<std::string> &SourceFiles) {
  bool IsCmakeScript = false;
  for (const auto &FilePath : SourceFiles) {
    if (!llvm::sys::path::has_extension(FilePath) ||
        llvm::sys::path::filename(FilePath).ends_with(".cmake") ||
        llvm::sys::path::filename(FilePath).ends_with(".txt"))
      IsCmakeScript = true;
    break;
  }
  return IsCmakeScript;
}

void migrateCmakeScriptOnly(
    const llvm::Expected<clang::tooling::CommonOptionsParser> &OptParser,
    StringRef InRoot, StringRef OutRoot) {

  auto CmakeScriptLists = OptParser->getSourcePathList();
  if (!CmakeScriptLists.empty()) {
    for (auto FilePath : CmakeScriptLists) {
      if (fs::is_directory(FilePath)) {
        std::vector<std::string> CmakeScriptFiles;
        collectCmakeScripts(FilePath, OutRoot, CmakeScriptFiles);
        for (const auto &ScriptFile : CmakeScriptFiles) {
          if (!migrateCmakeScriptFile(InRoot, OutRoot, ScriptFile))
            continue;
        }
      } else {
        if (!migrateCmakeScriptFile(InRoot, OutRoot, FilePath))
          continue;
      }
    }
  } else {
    std::vector<std::string> CmakeScriptFiles;
    collectCmakeScripts(InRoot, OutRoot, CmakeScriptFiles);
    for (const auto &ScriptFile : CmakeScriptFiles) {
      if (!migrateCmakeScriptFile(InRoot, OutRoot, ScriptFile))
        continue;
    }
  }
}

static size_t skipWhileSpaces(const std::string Input, size_t Index) {
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

static size_t gotoEndOfCmakeCommandStmt(const std::string Input, size_t Index) {
  size_t Size = Input.size();
  for (; Index < Size && Input[Index] != ')'; Index++) {
  }
  return Index;
}

void parseVariable(const std::string &Input,
                   std::map<std::string, std::string> &VariablesMap) {

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
    Index = skipWhileSpaces(Input, Index);
    Begin = Index;

    // Go the end of cmake command
    Index = gotoEndOfCmakeWord(Input, Begin + 1, '(');
    End = Index;

    // Skip possible space
    Index = skipWhileSpaces(Input, Index);

    if (Index < Size && Input[Index] == '(') {
      std::string Command = Input.substr(Begin, End - Begin);
      std::transform(Command.begin(), Command.end(), Command.begin(),
                     [](unsigned char c) { return std::tolower(c); });

      if (Command == "set") {
        Index++; // Skip '('
        std::string VarName;
        std::string Value;

        // Get the begin of firt argument of set
        Index = skipWhileSpaces(Input, Index);
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

        // Get the name of the second argument
        Value = Input.substr(Begin, End - Begin);

        // Only check the set commands which has two arguments
        if (Input[Index] == ')' &&
            !llvm::StringRef(VarName).starts_with("CMAKE_") &&
            !llvm::StringRef(VarName).starts_with("_CMAKE_ ")) {
          VariablesMap[VarName] = Value;
        }
      }

      // Go the ')' of cmake command
      Index = gotoEndOfCmakeCommandStmt(Input, Index);
    }

    Index++;
  }
}

static std::vector<std::string> split(const std::string &Input,
                                      const std::string &Delimiter) {
  std::vector<std::string> Vec;
  if (!Input.empty()) {

    size_t Index = 0;
    size_t Pos = Input.find(Delimiter, Index);
    while (Index < Input.size() && Pos != std::string::npos) {
      Vec.push_back(Input.substr(Index, Pos - Index));

      Index = Pos + Delimiter.size();
      Pos = Input.find(Delimiter, Index);
    }
    // Append the remaining part
    Vec.push_back(Input.substr(Index));
  }
  return Vec;
}

std::string getVarName(const std::string &Variable,
                       const std::map<std::string, std::string> &VariablesMap) {
  std::string Value;
  if (Variable[0] == '$' && Variable[1] == '{') {
    auto Name = Variable.substr(2, Variable.size() - 3);

    auto Iter = VariablesMap.find(Name);
    if (VariablesMap.find(Name) != VariablesMap.end()) {
      Value = Iter->second;
    }
  } else {
    Value = Variable;
  }

  return Value;
}

static std::string processArgOfCmakeVersionRequired(
    const std::string &Arg,
    const std::map<std::string, std::string> &VariablesMap) {
  size_t Pos = Arg.find("...");
  std::string ReplArg;
  if (Pos != std::string::npos) {
    const auto StrArray = split(Arg, "...");

    std::string MinVer = getVarName(StrArray[0], VariablesMap);
    std::string MaxVer = getVarName(StrArray[1], VariablesMap);

    if (std::atof(MinVer.c_str()) >= 3.24) {
      ReplArg = MinVer + "..." + MaxVer;
    } else if (std::atof(MinVer.c_str()) < 3.24 &&
               std::atof(MaxVer.c_str()) > 3.24) {
      ReplArg = "3.24..." + MaxVer;
    } else {
      ReplArg = "3.24";
    }

  } else {
    std::string Ver = getVarName(Arg, VariablesMap);
    if (std::atof(Ver.c_str()) < 3.24) {
      ReplArg = "3.24";
    } else {
      ReplArg = Ver;
    }
  }
  return ReplArg;
}

int skipCmakeComments(const std::string &Input, int Index) {
  const int Size = Input.size();
  if (Input[Index] == '#') {
    for (; Index < Size && Input[Index] != '\n'; Index++) {
    }
  }
  return Index;
}

void cmakeSyntaxProcessed(
    std::string &Input,
    const std::map<std::string, std::string> &VariablesMap) {

  int Size = Input.size();
  int Index = 0;
  while (Index < Size) {

    // Skip comments
    skipCmakeComments(Input, Index);

    int Begin, End;
    // Go the begin of cmake command
    Index = skipWhileSpaces(Input, Index);
    Begin = Index;

    // Go the end of cmake command
    Index = gotoEndOfCmakeWord(Input, Begin + 1, '(');
    End = Index;

    // Skip possible space
    Index = skipWhileSpaces(Input, Index);

    if (Index < Size && Input[Index] == '(') {
      std::string Command = Input.substr(Begin, End - Begin);

      if (Command == "cmake_minimum_required") {

        std::string VarName;
        std::string Value;

        // Get the begin of firt argument of cmake_minimum_required
        Index = skipWhileSpaces(Input, Index);
        Begin = Index;

        // Get the end of the first argument of cmake_minimum_required
        Index = gotoEndOfCmakeWord(Input, Index, ')');
        End = Index;

        // Get the name of the first argument
        VarName = Input.substr(Begin, End - Begin);

        // Get the begin of the second argument
        Index = skipWhileSpaces(Input, End + 1);
        Begin = Index;

        // Get the end of the second argument
        Index = gotoEndOfCmakeWord(Input, Begin + 1, ')');
        End = Index;

        // Get the name of the second argument
        Value = Input.substr(Begin, End - Begin);

        std::string ReplStr =
            processArgOfCmakeVersionRequired(Value, VariablesMap);

        Input.replace(Begin, End - Begin, ReplStr);
        Size = Input.size();            // Update string size
        Index = Begin + ReplStr.size(); // update index
      }

      // Go the ')' of cmake command
      Index = gotoEndOfCmakeCommandStmt(Input, Index);
    }

    Index++;
  }
}

std::string convertCmakeCommandsToLower(const std::string &InputString) {
  std::stringstream OutputStream;

  const auto Lines = split(InputString, '\n');
  std::vector<std::string> Output;
  for (auto Line : Lines) {

    int Size = Line.size();
    int Index = 0;

    // Go the begin of cmake command
    Index = skipWhileSpaces(Line, Index);
    int Begin = Index;

    // Go the end of cmake command
    Index = gotoEndOfCmakeWord(Line, Begin + 1, '(');
    int End = Index;

    if (Index < Size && Line[Index] == '(') {
      std::string Str = Line.substr(Begin, End - Begin);
      std::transform(Str.begin(), Str.end(), Str.begin(),
                     [](unsigned char Char) { return std::tolower(Char); });
      if (cmake_commands.find(Str) != cmake_commands.end()) {
        for (int Idx = Begin; Idx < End; Idx++) {
          Line[Idx] = Str[Idx - Begin];
        }
      }
    }

    OutputStream << Line << "\n";
  }

  return OutputStream.str();
}
