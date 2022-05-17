//===--------------- GenMakefile.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "GenMakefile.h"
#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "SaveNewFiles.h"
#include "ValidateArguments.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "clang/Tooling/Refactoring.h"

#include "llvm/Support/raw_os_ostream.h"
#include <fstream>
#include <unordered_map>

using namespace clang::dpct;
using namespace llvm;

// Used to identify compilation commands without target
const std::string EmptyTarget = "NoLinker";

std::map<std::string /*migrated file name*/, clang::tooling::CompilationInfo>
    CompileCmdsMap;

std::map<std::string /*target*/, std::vector<clang::tooling::CompilationInfo>>
    CompileCmdsPerTarget;

std::vector<std::pair<std::string /*original file name or linker entry*/,
                      std::vector<std::string> /*original compile command*/>>
    CompileTargetsMap;

static void fillCompileCmds(
    std::map<std::string, std::vector<clang::tooling::CompilationInfo>>
        &CompileCmds,
    std::string MigratedFileName, std::string CompileOptions,
    std::string Compiler, std::string TargetName) {
  clang::tooling::CompilationInfo CmpInfo;
  CmpInfo.MigratedFileName = MigratedFileName;
  CmpInfo.CompileOptions = CompileOptions;
  CmpInfo.Compiler = Compiler;
  CompileCmds[TargetName].push_back(CmpInfo);
}

// To get customized basename from the file path.
// E.g: /path/to/foo.cc.cpp --> foo.cc
static std::string getCustomBaseName(const std::string &Path) {
  std::string Filename = llvm::sys::path::filename(Path).str();
  std::size_t Pos = Filename.find_last_of('.');
  if (Pos != std::string::npos) {
    std::string BaseName = Filename.substr(0, Pos);
    return BaseName;
  } else
    return Filename;
}

static void getCompileInfo(
    StringRef InRoot, StringRef OutRoot,
    std::map<std::string, std::vector<clang::tooling::CompilationInfo>>
        &CompileCmds) {

  std::unordered_map<std::string, bool> ObjsInLinkerCmd;
  std::unordered_map<std::string, std::unordered_map<std::string, bool>>
      ObjsInLinkerCmdPerTarget;

  std::map<std::string, clang::tooling::CompilationInfo> CmdsMap;

  for (const auto &Entry : CompileTargetsMap) {
    std::string FileName = Entry.first;
    if (llvm::StringRef(FileName).startswith("LinkerEntry")) {
      // Parse linker cmd to get target name and objfile names
      std::unordered_map<std::string, bool> ObjsInLinker;

      bool IsTargetName = false;
      std::string ObjName;
      std::string TargetName;
      for (const auto &Obj : Entry.second) {
        if (Obj == "-o") {
          IsTargetName = true;
        } else if (IsTargetName) {
          // Set the target name
          TargetName = Obj;
          IsTargetName = false;
        } else if (llvm::StringRef(Obj).endswith(".o")) {
          llvm::SmallString<512> FilePathAbs(Obj);
          llvm::sys::path::native(FilePathAbs);
          llvm::sys::fs::make_absolute(FilePathAbs);
          llvm::sys::path::remove_dots(FilePathAbs, true);
          ObjsInLinkerCmd[std::string(FilePathAbs.str())] = true;
          ObjsInLinker[std::string(FilePathAbs.str())] = true;
          ObjName = std::string(FilePathAbs.str());
        }
      }

      if (llvm::StringRef(TargetName).endswith(".o") &&
          llvm::StringRef(Entry.second[1]).endswith("nvcc")) {
        // Skip linker command like:
        // foo_generated_foo.cu.o ->foo_intermediate_link.o
        continue;
      }

      ObjsInLinkerCmdPerTarget[TargetName] = ObjsInLinker;
    } else {
      continue;
    }
  }

  std::unordered_map<std::string /*origname*/, std::string /*objname*/>
      Orig2ObjMap;

  for (const auto &Entry : CompileTargetsMap) {
    std::string FileName = Entry.first;

    if (llvm::StringRef(FileName).startswith("LinkerEntry")) {
      continue;
    }

    std::string NewOptions;
    bool IsObjName = false;
    bool IsObjSpecified = false;
    const std::string Directory = Entry.second[0];
    std::unordered_set<std::string> DuplicateDuplicateFilter;
    for (const auto &Option : Entry.second) {
      if (llvm::StringRef(Option).startswith("-I")) {
        // Parse include path specified by "-I"
        std::string IncPath = Option.substr(strlen("-I"));
        size_t Begin = IncPath.find_first_not_of(" ");
        IncPath = IncPath.substr(Begin);

        if (!llvm::sys::fs::exists(IncPath)) {
          // Skip including path that does not exist.
          continue;
        }

        SmallString<512> OutDirectory = llvm::StringRef(IncPath);
        llvm::sys::fs::make_absolute(OutDirectory);
        llvm::sys::path::remove_dots(OutDirectory, /*remove_dot_dot=*/true);
        makeCanonical(OutDirectory);

        if (!isChildPath(InRoot.str(), std::string(OutDirectory.c_str()),
                         false)) {
          // Skip include path that is not in inRoot directory
          continue;
        }

        rewriteDir(OutDirectory, InRoot, OutRoot);

        NewOptions += "-I";
        llvm::sys::path::replace_path_prefix(OutDirectory, OutRoot, ".");
        NewOptions += OutDirectory.c_str();
        NewOptions += " ";

      } else if (llvm::StringRef(Option).startswith("-D")) {
        // Parse macros defined.
        std::size_t Len = Option.length() - strlen("-D");
        std::size_t Pos = Option.find("=");
        if (Pos != std::string::npos) {
          Len = Pos - strlen("-D");
        }
        std::string MacroName = Option.substr(strlen("-D"), Len);
        auto Iter = MapNames::MacrosMap.find(MacroName);
        if (Iter != MapNames::MacrosMap.end())
          // Skip macros defined in helper function header files
          continue;
        else
          NewOptions += Option + " ";
      } else if (llvm::StringRef(Option).startswith("-std=")) {

        size_t Idx = 0;
        for (; Idx < Option.length(); Idx++) {
          if (std::isdigit(Option[Idx]))
            break;
        }
        auto Version = Option.substr(Idx, Option.length() - Idx);
        int Val = std::atoi(Version.c_str());
        // DPC++ support c++17 as default.
        if (llvm::StringRef(Entry.second[1]).endswith("nvcc") && Val <= 17)
          continue;

        // Skip duplicate options.
        if (DuplicateDuplicateFilter.find(Option) !=
            end(DuplicateDuplicateFilter))
          continue;
        DuplicateDuplicateFilter.insert(Option);

        NewOptions += Option + " ";
      } else if (Option == "-o") {
        IsObjName = true;
        IsObjSpecified = true;
      } else if (IsObjName) {
        llvm::SmallString<512> FilePathAbs(Option);
        llvm::sys::path::native(FilePathAbs);
        llvm::sys::fs::make_absolute(FilePathAbs);
        llvm::sys::path::remove_dots(FilePathAbs, true);
        Orig2ObjMap[FileName] = std::string(FilePathAbs.str());
        IsObjName = false;
      } else if (llvm::StringRef(Option).startswith("-O")) {
        // Keep optimization level same as original compile command.
        NewOptions += Option + " ";
      }
    }
    if (!IsObjSpecified) {
      // For the case that "-o" is not specified in the compile command, the
      // default obj file is generated in the directory where the compile
      // command runs.
      Orig2ObjMap[FileName] =
          Directory + "/" + getCustomBaseName(FileName) + ".o";
    }

    // if option "--use-custom-helper=<value>" is used to customize the helper
    // header files for migrated code, the path of the helper header files
    // should be included.
    if (llvm::sys::fs::exists(OutRoot.str() + "/include")) {
      NewOptions += " -I ./include ";
    }

    SmallString<512> OutDirectory = llvm::StringRef(FileName);
    makeCanonical(OutDirectory);
    rewriteDir(OutDirectory, InRoot, OutRoot);
    auto OrigFileName = OutDirectory;
    rewriteFileName(OutDirectory);

    if (llvm::sys::fs::exists(OutDirectory)) {
      llvm::sys::path::replace_path_prefix(OutDirectory, OutRoot, ".");
      clang::tooling::CompilationInfo CmpInfo;
      CmpInfo.MigratedFileName = OutDirectory.c_str();
      CmpInfo.CompileOptions = NewOptions;
      CmpInfo.Compiler = Entry.second[1];
      CmdsMap[Orig2ObjMap[FileName]] = CmpInfo;
    } else {
      llvm::sys::path::replace_path_prefix(OrigFileName, OutRoot, ".");
      clang::tooling::CompilationInfo CmpInfo;
      CmpInfo.MigratedFileName = OrigFileName.c_str();
      CmpInfo.CompileOptions = NewOptions;
      CmpInfo.Compiler = Entry.second[1];
      CmdsMap[Orig2ObjMap[FileName]] = CmpInfo;
    }
  }

  for (const auto &Entry : ObjsInLinkerCmdPerTarget) {
    for (const auto &Obj : Entry.second) {

      auto Iter = CmdsMap.find(Obj.first);
      if (Iter != CmdsMap.end()) {
        auto CmpInfo = Iter->second;
        fillCompileCmds(CompileCmds, CmpInfo.MigratedFileName,
                        CmpInfo.CompileOptions, CmpInfo.Compiler, Entry.first);
      }
    }
  }

  if (ObjsInLinkerCmdPerTarget.empty()) {
    for (const auto &Cmd : CmdsMap) {
      fillCompileCmds(CompileCmds, Cmd.second.MigratedFileName,
                      Cmd.second.CompileOptions, Cmd.second.Compiler,
                      EmptyTarget);
    }
  }
}

static void
genMakefile(clang::tooling::RefactoringTool &Tool, StringRef OutRoot,
            const std::string &BuildScriptName,
            std::map<std::string, std::vector<clang::tooling::CompilationInfo>>
                &CmdsPerTarget) {
  std::string Buf;
  llvm::raw_string_ostream OS(Buf);
  std::string TargetName;

  OS << "CC := dpcpp\n\n";
  OS << "LD := $(CC)\n\n";
  OS << buildString(
      "#", DiagnosticsUtils::getMsgText(MakefileMsgs::GEN_MAKEFILE_LIB), "\n");
  OS << "LIB := \n\n";

  OS << buildString("FLAGS := \n\n");

  std::map<std::string, std::string> ObjsPerTarget;

  int TargetIdx = 0;
  for (const auto &Entry : CmdsPerTarget) {
    TargetName = Entry.first;
    SmallString<512> TargetFilePath =
        llvm::StringRef(OutRoot.str() + "/" + TargetName);

    auto Parent = path::parent_path(TargetFilePath);
    if (!llvm::sys::fs::exists(Parent)) {

      std::error_code EC;
      EC = llvm::sys::fs::create_directories(Parent);
      if ((bool)EC) {
        std::string ErrMsg =
            "[ERROR] Create Directory : " + std::string(Parent.str()) +
            " fail: " + EC.message() + "\n";
        PrintMsg(ErrMsg);
      }
    }

    auto CmpInfos = Entry.second;
    int Count = 0;
    for (const auto &CmpInfo : CmpInfos) {
      const auto &MigratedName = CmpInfo.MigratedFileName;

      std::string SrcStrName = "TARGET_" + std::to_string(TargetIdx) + "_SRC_" +
                               std::to_string(Count);
      std::string ObjStrName = "TARGET_" + std::to_string(TargetIdx) + "_OBJ_" +
                               std::to_string(Count);
      std::string FlagStrName = "TARGET_" + std::to_string(TargetIdx) +
                                "_FLAG_" + std::to_string(Count);
      OS << buildString(SrcStrName, " = ", MigratedName, "\n");
      SmallString<512> FilePath = StringRef(MigratedName);
      path::replace_extension(FilePath, "o");
      OS << buildString(ObjStrName, " = ", FilePath, "\n");
      OS << buildString(FlagStrName, " = ", CmpInfo.CompileOptions,
                        "${FLAGS}\n\n");

      ObjsPerTarget[TargetName] +=
          buildString(" ${", buildString(ObjStrName), "}");
      Count++;
    }
    TargetIdx++;
  }

  std::string Target;
  std::string ObjStr;
  TargetIdx = 0;
  for (const auto &Entry : CmdsPerTarget) {
    if (TargetName == EmptyTarget)
      continue;

    OS << buildString("TARGET_", std::to_string(TargetIdx), " := ", Entry.first,
                      "\n");
    Target += " " + buildString("${TARGET_", std::to_string(TargetIdx), "}");
    TargetIdx++;
  }

  if (!Tool.isInputfileSpecified() && TargetName != EmptyTarget) {
    OS << buildString("\nTARGET := ", Target, "\n\n");
  }
  OS << buildString(".PHONY:all clean\n");

  TargetIdx = 0;
  std::string AllObjs;
  for (const auto &Entry : CmdsPerTarget) {
    OS << buildString("OBJS_", std::to_string(TargetIdx),
                      " := ", ObjsPerTarget[Entry.first], "\n");
    AllObjs += buildString(" ${OBJS_", std::to_string(TargetIdx), "}");
    TargetIdx++;
  }

  if (Tool.isInputfileSpecified() || TargetName == EmptyTarget)
    OS << buildString("OBJS := ", AllObjs, "\n\n");

  if (Tool.isInputfileSpecified() || TargetName == EmptyTarget) {
    // For the case that target name is not available or input file(s) only
    // specified in command line, only compile command(s) generated in Makefile.
    OS << buildString("all: $(OBJS)\n\n");
    TargetIdx = 0;
    for (const auto &Entry : CmdsPerTarget) {

      for (unsigned Idx = 0; Idx < Entry.second.size(); Idx++) {
        std::string SrcStrName = "TARGET_" + std::to_string(TargetIdx) +
                                 "_SRC_" + std::to_string(Idx);
        std::string ObjStrName = "TARGET_" + std::to_string(TargetIdx) +
                                 "_OBJ_" + std::to_string(Idx);
        std::string FlagStrName = "TARGET_" + std::to_string(TargetIdx) +
                                  "_FLAG_" + std::to_string(Idx);
        OS << buildString("$(", ObjStrName, "):$(", SrcStrName, ")\n");

        // Only apply dpcpp compiler to the files which are originally built
        // with nvcc.
        std::string Compiler =
            llvm::StringRef((Entry.second)[Idx].Compiler).endswith("nvcc")
                ? "$(CC)"
                : (Entry.second)[Idx].Compiler;

        OS << buildString("\t", Compiler, " -c ${", SrcStrName, "} -o ${",
                          ObjStrName, "} $(", FlagStrName, ")\n\n");
      }
      TargetIdx++;
    }

  } else {
    std::string MKLOption = DpctGlobalInfo::isMKLHeaderUsed() ? "-qmkl" : "";
    OS << buildString("all: $(TARGET)\n");
    TargetIdx = 0;
    for (const auto &Entry : CmdsPerTarget) {
      OS << buildString("$(TARGET_", std::to_string(TargetIdx), "): $(OBJS_",
                        std::to_string(TargetIdx), ")\n");
      OS << buildString("\t$(LD) -o $@ $^ $(LIB) ", MKLOption, "\n\n");

      for (unsigned Idx = 0; Idx < Entry.second.size(); Idx++) {
        std::string SrcStrName = "TARGET_" + std::to_string(TargetIdx) +
                                 "_SRC_" + std::to_string(Idx);
        std::string ObjStrName = "TARGET_" + std::to_string(TargetIdx) +
                                 "_OBJ_" + std::to_string(Idx);
        std::string FlagStrName = "TARGET_" + std::to_string(TargetIdx) +
                                  "_FLAG_" + std::to_string(Idx);
        OS << buildString("$(", ObjStrName, "):$(", SrcStrName, ")\n");

        std::string Compiler =
            llvm::StringRef((Entry.second)[Idx].Compiler).endswith("nvcc")
                ? "$(CC)"
                : (Entry.second)[Idx].Compiler;

        OS << buildString("\t", Compiler, " -c ${", SrcStrName, "} -o ${",
                          ObjStrName, "} $(", FlagStrName, ")\n\n");
      }
      TargetIdx++;
    }
  }

  OS << "clean:\n";
  if (!Tool.isInputfileSpecified() && TargetName != EmptyTarget)
    OS << buildString("\trm -f ", AllObjs, " $(TARGET)\n");
  else
    OS << buildString("\trm -f ", AllObjs, "\n");

  std::string FileOut = OutRoot.str() + "/" + BuildScriptName;
  std::ofstream File;
  File.open(FileOut, std::ios::binary);
  if (File) {
    File << OS.str();
    File.close();
  }
}

void genBuildScript(clang::tooling::RefactoringTool &Tool, StringRef InRoot,
                    StringRef OutRoot, const std::string &BuildScriptName) {

  std::map<std::string /*traget name*/,
           std::vector<clang::tooling::CompilationInfo>>
      NewCompileCmdsMap;

  getCompileInfo(InRoot, OutRoot, NewCompileCmdsMap);

  bool NeedMergetYaml = false;

  const std::string Target = EmptyTarget;

  // Increment compilation support only if input file(s) only specified in
  // command line
  auto Iter = NewCompileCmdsMap.find(Target);
  if (Iter != NewCompileCmdsMap.end()) {
    auto Iter = CompileCmdsPerTarget.find(Target);
    if (Iter != CompileCmdsPerTarget.end() || CompileCmdsPerTarget.empty()) {
      NeedMergetYaml = true;
    }
  }

  std::map<std::string, bool> DuplicateFilter;
  for (const auto &Entry : CompileCmdsPerTarget) {
    std::string FileName = Entry.first;
    for (const auto &Option : Entry.second) {
      std::string Key = Entry.first + Option.MigratedFileName +
                        Option.CompileOptions + Option.Compiler;
      DuplicateFilter[Key] = true;
    }
  }

  if (NeedMergetYaml) {
    for (const auto &Entry : NewCompileCmdsMap) {
      for (const auto &Option : Entry.second) {
        std::string Key = Entry.first + Option.MigratedFileName +
                          Option.CompileOptions + Option.Compiler;
        auto Iter = DuplicateFilter.find(Key);
        if (Iter == DuplicateFilter.end()) {
          CompileCmdsPerTarget[Entry.first].push_back(Option);
        }
      }
    }
  }

  if (!NeedMergetYaml)
    CompileCmdsPerTarget = NewCompileCmdsMap;

  genMakefile(Tool, OutRoot, BuildScriptName, CompileCmdsPerTarget);
}
