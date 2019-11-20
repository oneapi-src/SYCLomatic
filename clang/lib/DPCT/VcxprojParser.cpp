//===--- VcxprojParser.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//
#include "VcxprojParser.h"
#include "Debug.h"
#include "SaveNewFiles.h"
#include "Utility.h"

std::map<std::string, std::vector<std::string>> OptionsMap;
std::map<std::string, std::string> VariablesMap;

std::set<std::string> MacroDefinedSet;
std::set<std::string> DirIncludedSet;
std::map<std::string /*compiler*/, std::vector<std::string> /*source file*/>
    FilesSet;

const std::unordered_map<std::string /*option*/, std::string /*option*/>
    OptionsMapped = {
        {"--fatbin", "-Xcuda-fatbinary"},
        {"-fatbin", "-Xcuda-fatbinary"},
        {"-G", "--cuda-noopt-device-debug"},
        {"--device-debug", "--cuda-noopt-device-debug"},
        {"--machine", "-m"},
        {"-m", "-m"},
        {"--gpu-architecture", "--cuda-gpu-arch="},
        {"-arch", "--cuda-gpu-arch="},
        {"--disable-warnings", "--no-warnings"},
        {"-w", "--no-warnings"},
};

const std::unordered_map<std::string /*option*/, bool /*has value*/>
    OptionsIgnored = {
        {"-MD", 0},
        {"-MMD", 0},
        {"-MG", 0},
        {"-MP", 0},
        {"-MF", 1},
        {"--dependency-output", 1},
        {"-MT", 1},
        {"-MQ", 1},
        {"-static", 0},
        {"-shared", 0},
        {"-s", 0},
        {"-rdynamic", 0},
        {"-l", 1},
        {"-L", 1},
        {"-u", 1},
        {"-z", 1},
        {"-T", 1},
        {"-march", 1},
        {"--cuda", 0},
        {"-cuda", 0},
        {"--cubin", 0},
        {"-cubin", 0},
        {"--ptx", 0},
        {"-ptx", 0},
        {"--device-c", 0},
        {"-dc", 0},
        {"--device-w", 0},
        {"-dw", 0},
        {"--device-link", 0},
        {"-dlink", 0},
        {"--link", 0},
        {"-link", 0},
        {"--lib", 0},
        {"-lib", 0},
        {"--run", 0},
        {"-run", 0},
        {"--pre-include", 1},
        {"--library", 1},
        {"-l", 1},
        {"--library-path", 1},
        {"-L", 1},
        {"--output-directory", 1},
        {"-odir", 1},
        {"--compiler-bindir", 1},
        {"-ccbin", 1},
        {"-cudart", 1},
        {"--cudart", 1},
        {"--libdevice-directory", 1},
        {"-ldir", 1},
        {"--use-local-env", 0},
        {"--profile", 0},
        {"-pg", 0},
        {"--debug", 0},
        {"-g", 0},
        {"--generate-line-info", 0},
        {"-lineinfo", 0},
        {"--shared", 0},
        {"-shared", 0},
        {"--x", 1},
        {"-x", 1},
        {"--no-host-device-initializer-list", 0},
        {"-nohdinitlist", 0},
        {"--no-host-device-move-forward", 0},
        {"-nohdmoveforward", 0},
        {"--expt-relaxed-constexpr", 0},
        {"-expt-relaxed-constexpr", 0},
        {"--expt-extended-lambda", 0},
        {"-expt-extended-lambda", 0},
        {"-Xcompiler", 1},
        {"--compiler-options", 1},
        {"--compiler-options", 1},
        {"-Xcompiler", 1},
        {"--linker-options", 1},
        {"-Xlinker", 1},
        {"--archive-options", 1},
        {"-Xarchive", 1},
        {"--ptxas-options", 1},
        {"-Xptxas", 1},
        {"--nvlink-options", 1},
        {"-Xnvlink", 1},
        {"-noprof", 0},
        {"--dont-use-profile", 0},
        {"-dryrun", 0},
        {"--dryrun", 0},
        {"--verbose", 0},
        {"-v", 0},
        {"--keep", 0},
        {"-keep", 0},
        {"--keep-dir", 1},
        {"-keep-dir", 1},
        {"--save-temps", 0},
        {"-save-temps", 0},
        {"--clean-targets", 0},
        {"-clean", 0},
        {"--run-args", 1},
        {"-run-args", 1},
        {"--input-drive-prefix", 1},
        {"-idp", 1},
        {"--dependency-drive-prefix", 1},
        {"-ddp", 1},
        {"--drive-prefix", 1},
        {"-dp", 1},
        {"--dependency-target-name", 1},
        {"-MT", 1},
        {"--no-align-double", 0},
        {"--no-device-link", 0},
        {"-nodlink", 0},
        {"--gpu-code", 1},
        {"-code", 1},
        {"-gencode", 1},
        {"--generate-code", 1},
        {"--relocatable-device-code", 1},
        {"-rdc", 1},
        {"--entries", 1},
        {"-e", 1},
        {"--maxrregcount", 1},
        {"-maxrregcount", 1},
        {"--use_fast_math", 0},
        {"-use_fast_math", 0},
        {"--ftz", 1},
        {"-ftz", 1},
        {"--prec-div", 1},
        {"-prec-div", 1},
        {"--prec-sqrt", 1},
        {"-prec-sqrt", 1},
        {"--fmad", 1},
        {"-fmad", 1},
        {"--default-stream", 1},
        {"-default-stream", 1},
        {"--keep-device-functions", 0},
        {"-keep-device-functions", 0},
        {"--source-in-ptx", 0},
        {"-src-in-ptx", 0},
        {"--restrict", 0},
        {"-restrict", 0},
        {"--Wreorder", 0},
        {"-Wreorder", 0},
        {"--Wno-deprecated-declarations", 0},
        {"-Wno-deprecated-declarations", 0},
        {"--Wno-deprecated-gpu-targets", 0},
        {"-Wno-deprecated-gpu-targets", 0},
        {"--Werror", 1},
        {"-Werror", 1},
        {"--resource-usage", 0},
        {"-res-usage", 0},
        {"--extensible-whole-program", 0},
        {"-ewp", 0},
        {"--no-compress", 0},
        {"-no-compress", 0},
        {"--help", 0},
        {"-h", 0},
        {"--version", 0},
        {"-V", 0},
        {"--options-file", 0},
        {"-optf", 0},
};

void addDiretoryToDirIncludedSet(const std::string &Directory) {
  if (DirIncludedSet.find(Directory) == end(DirIncludedSet)) {
    DirIncludedSet.insert(Directory);
  }
}

void addMacroDefinedSet(const std::string &MacroDefined) {
  if (MacroDefinedSet.find(MacroDefined) == end(MacroDefinedSet)) {
    MacroDefinedSet.insert(MacroDefined);
  }
}

void addFilesSet(const std::string Compiler, std::string &File) {
  FilesSet[Compiler].push_back(File);
}

void updateOptionsMap(const std::string &Option, const std::string &Value) {
  OptionsMap[Option].push_back(Value);
}

void backslashToForwardslash(std::string &Str) {
  std::replace(Str.begin(), Str.end(), '\\', '/');
}

// Evaluate variable with its value in file path if possible.
void replaceVar(std::string &SubStr, size_t VarStart = 0) {
  VarStart = SubStr.find("$(", VarStart);
  while (VarStart != std::string::npos) {
    size_t VarEnd = SubStr.find(")", VarStart + 2);
    std::string Variable = SubStr.substr(VarStart + 2, VarEnd - VarStart - 2);
    if (VariablesMap.find(Variable) != VariablesMap.end()) {
      std::string ReplaceStr = VariablesMap[Variable];
      SubStr.replace(VarStart, VarEnd - VarStart + 1, ReplaceStr);
      replaceVar(SubStr, VarStart);
    } else {
      VarStart = SubStr.find("$(", VarEnd);
    }
  }
}

void processOptions(std::string &Output) {
  for (const auto &Entry : OptionsMap) {
    std::string Option = Entry.first;
    if (OptionsIgnored.find(Option) != OptionsIgnored.end()) {
      // Ingore option in OptionsIgnored
      continue;
    }

    auto Iter = OptionsMapped.find(Option);
    if (Iter != OptionsMapped.end()) {
      // Map option with new name in OptionsMapped
      Option = Iter->second;
    }

    // For options that have more than one values, only use their first
    // values.
    std::string Value = Entry.second.front();
    if (Option == "-m") {
      Output += Option + Value;
    } else {
      Option = Option + " " + Value;
      Output += Option;
    }
    Output += " ";
  }
}

void ProcessMacrosDefined(std::string &Output) {
  for (auto const &Macro : MacroDefinedSet) {
    if (Macro.find("%") != std::string::npos) {
      // Skip variables such as %(PreprocessorDefinitions)
      continue;
    }
    std::string MacroDefined = "-D" + Macro + " ";
    Output += MacroDefined;
  }
}

void ProcessDirectoriesIncluded(std::string &Output) {
  for (auto const &Dir : DirIncludedSet) {
    std::string DirectoryInclude = Dir;
    replaceVar(DirectoryInclude);
    backslashToForwardslash(DirectoryInclude);
    DirectoryInclude = "-I\\\"" + DirectoryInclude + "\\\" ";

    if (DirectoryInclude.find("$") != std::string::npos) {
      // Skip variables such as $(TMP) in $(TMP)/tmp.exe, that still could not
      // be replaced.
      std::string Warning =
          "Cannot evaluate variable in  \"" + DirectoryInclude + "\"\n";
      clang::dpct::PrintMsg(Warning);
      continue;
    }
    Output += DirectoryInclude;
  }
}

void generateCompilationDatabase(const std::string &BuildDir) {
  std::string Options;
  processOptions(Options);

  std::string MacrosDefined;
  ProcessMacrosDefined(MacrosDefined);

  std::string DirectoriesIncluded;
  ProcessDirectoriesIncluded(DirectoriesIncluded);

  size_t EntryCount = 0;
  size_t TotalCount = 0;
  for (auto const &Entry : FilesSet) {
    TotalCount += Entry.second.size();
  }

  std::string FilePath = BuildDir + "/compile_commands.json";
  std::ofstream OutFile(FilePath);
  if (!OutFile) {
    std::string ErrMsg =
        "Cannot create CompilationDatabase \"" + FilePath + "\"\n";
    clang::dpct::PrintMsg(ErrMsg);
    exit(VcxprojPaserCreateCompilationDBFail);
  }

  OutFile << "[\n";
  for (auto const &Entry : FilesSet) {
    const std::string Compiler = Entry.first + " ";
    auto FilesSet = Entry.second;
    for (auto const &File : FilesSet) {
      EntryCount++;
      std::string FileName = "\"file\":\"" + File + "\",";
      std::string Command = "\"command\":\"" + Compiler + Options +
                            MacrosDefined + DirectoriesIncluded + "\\\"" +
                            File + "\\\"\",";
      std::string Directory = "\"directory\":\"" + BuildDir + "\"";
      OutFile << "    {\n";
      OutFile << "        " << FileName << "\n";
      OutFile << "        " << Command << "\n";
      OutFile << "        " << Directory << "\n";
      if (EntryCount < TotalCount) {
        OutFile << "    },\n";
      } else {
        OutFile << "    }\n";
      }
    }
  }
  OutFile << "]\n";
}

void collectFiles(const std::string &Compiler, const std::string &Line) {
  size_t Pos = Line.find("Include=");
  if (Pos != std::string::npos) {
    size_t Start = Line.find("Include=") + sizeof("Include=\"") - 1;
    size_t End = Line.find("\"", Start + 1);
    std::string SubStr = Line.substr(Start, End - Start);
    backslashToForwardslash(SubStr);
    // Exclude *.txt files, i.e. <CustomBuild Include="/path/to/CMakeLists.txt">
    // Exclude *.rule files, i.e. <CustomBuild
    // Include="/path/to/name_intermediate_link.obj.rule">
    if (!endsWith(SubStr, ".txt") && !endsWith(SubStr, ".rule")) {
      addFilesSet(Compiler, SubStr);
    }
  }
}

void collectOtions(const std::string &Line) {
  size_t Pos = Line.find("<CodeGeneration>");
  if (Pos != std::string::npos) {
    size_t Start = Pos + sizeof("<CodeGeneration>") - 1;
    size_t End = Line.find("</CodeGeneration>");
    std::string SubStr = Line.substr(Start, End - Start);
    std::vector<std::string> CodeGenValues = split(SubStr, ';');
    for (auto const &CodeGenValue : CodeGenValues) {
      size_t Pos = CodeGenValue.find(',');
      std::string Arch = CodeGenValue.substr(0, Pos);
      updateOptionsMap("-gencode", Arch);
      updateOptionsMap("-code", CodeGenValue);
    }
  }

  Pos = Line.find("<TargetMachinePlatform>");
  if (Pos != std::string::npos) {
    size_t Start = Pos + sizeof("<TargetMachinePlatform>") - 1;
    size_t End = Line.find("</TargetMachinePlatform>");
    std::string SubStr = Line.substr(Start, End - Start);
    std::vector<std::string> PreprocessorMacros = split(SubStr, ';');
    for (auto const &PreprocessorMacro : PreprocessorMacros) {
      updateOptionsMap("-m", PreprocessorMacro);
    }
  }
}

using FunProcessBaseNode = void (*)(const std::string &Str);
void collectMacrosAndIncludingDIr(const std::string &&Node,
                                  const std::string &Line,
                                  FunProcessBaseNode FunPtr) {
  const std::string StartNode = "<" + Node + ">";
  const std::string EndNode = "</" + Node + ">";
  size_t Pos = Line.find(StartNode);

  if (Pos != std::string::npos) {
    size_t Start = Pos + StartNode.length();
    size_t End = Line.find(EndNode);
    std::string SubStr = Line.substr(Start, End - Start);
    std::vector<std::string> VecSet = split(SubStr, ';');
    for (auto const &Entry : VecSet) {
      // Skip CMAKE_INTDIR="Debug", CMAKE_INTDIR="MinSizeRel",
      // CMAKE_INTDIR="RelWithDebInfo", CMAKE_INTDIR="Release",
      // and skip the Entry that has the same name with Node, such as
      // "<AdditionalIncludeDirectories>/path/to/dir/;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>"
      if (Entry.find("Debug") != std::string::npos ||
          Entry.find("Release") != std::string::npos ||
          Entry.find("MinSizeRel") != std::string::npos ||
          Entry.find("RelWithDebInfo") != std::string::npos ||
          Entry.find(Node) != std::string::npos) {
        continue;
      }
      FunPtr(Entry);
    }
  }
}

void collectCompileNodeInfo(const std::string &Compiler,
                            const std::vector<std::string> &CompileNode) {
  for (auto const &Line : CompileNode) {
    collectOtions(Line);
    collectFiles(Compiler, Line);

    // Collect Macros defined.
    collectMacrosAndIncludingDIr("Defines", Line, addMacroDefinedSet);

    // Collect including directory
    collectMacrosAndIncludingDIr("Include", Line, addDiretoryToDirIncludedSet);
    collectMacrosAndIncludingDIr("AdditionalIncludeDirectories", Line,
                                 addDiretoryToDirIncludedSet);

    // Collect Macros defined
    collectMacrosAndIncludingDIr("PreprocessorDefinitions", Line,
                                 addMacroDefinedSet);
    // TODO:
    // Process AdditionalOptions node.
    // Need to support the case that "<>" and </>"" in multi lines.
  }
}

void parseVaribles(const std::string &VcxprojFile) {
  std::ifstream InFile(VcxprojFile);
  if (!InFile) {
    std::string ErrMsg = "Cannot Open VcxprojFile \"" + VcxprojFile + "\"\n";
    clang::dpct::PrintMsg(ErrMsg);
    exit(VcxprojPaserFileNotExist);
  }

  std::string Line;

  while (std::getline(InFile, Line)) {
    if (Line.find("<") != std::string::npos) {
      const size_t BeginNodeStart = Line.find("<");
      const size_t BeginNodeEnd = Line.find(">", BeginNodeStart + 1);
      const size_t EndNodeStart = Line.find("</", BeginNodeEnd + 1);
      const size_t EndNodeEnd = Line.find(">", EndNodeStart + 1);

      if (BeginNodeStart != std::string::npos &&
          BeginNodeEnd != std::string::npos &&
          EndNodeStart != std::string::npos &&
          EndNodeEnd != std::string::npos) {
        const std::string BeginVariableName =
            Line.substr(BeginNodeStart + 1, BeginNodeEnd - BeginNodeStart - 1);
        const std::string EndVariableName =
            Line.substr(EndNodeStart + 2, EndNodeEnd - EndNodeStart - 2);
        if (BeginVariableName == EndVariableName) {
          std::string Value =
              Line.substr(BeginNodeEnd + 1, EndNodeStart - BeginNodeEnd - 1);
          if (VariablesMap.find(BeginVariableName) == VariablesMap.end()) {
            VariablesMap[BeginVariableName] = Value;
          }
        }
      }
    }
  }
}

void processCompileNode(const std::string &&CompileNodeName,
                        std::ifstream &Infile, std::string &Line) {
  const std::string StartCompileNode = "<" + CompileNodeName + ">";
  const std::string EndCompileNode = "</" + CompileNodeName + ">";
  const std::string WholeCompileNode = "<" + CompileNodeName + " ";

  if (Line.find(StartCompileNode) != std::string::npos) {
    std::vector<std::string> CompileNode;
    CompileNode.push_back(Line);
    while (std::getline(Infile, Line)) {
      size_t pos = Line.find(EndCompileNode);
      CompileNode.push_back(Line);
      if (pos != std::string::npos) {
        break;
      }
    }
    collectCompileNodeInfo(CompileNodeName, CompileNode);
  } else if (Line.find(WholeCompileNode) != std::string::npos) {
    std::vector<std::string> CompileNode;
    CompileNode.push_back(Line);
    collectCompileNodeInfo(CompileNodeName, CompileNode);
  }
}

void parseVcxprojFile(const std::string &VcxprojFile) {

  std::ifstream Infile;
  Infile.open(VcxprojFile);

  if (!Infile) {
    std::string ErrMsg = "Cannot Open VcxprojFile \"" + VcxprojFile + "\"\n";
    clang::dpct::PrintMsg(ErrMsg);
    exit(VcxprojPaserFileNotExist);
  }

  std::string Line;

  while (std::getline(Infile, Line)) {
    processCompileNode("CudaCompile", Infile, Line);
    processCompileNode("ClCompile", Infile, Line);
    processCompileNode("CustomBuild", Infile, Line);
  }
}

void vcxprojParser(std::string &BuildPath, std::string &VcxprojFile) {
  backslashToForwardslash(VcxprojFile);
  backslashToForwardslash(BuildPath);

  parseVaribles(VcxprojFile);
  parseVcxprojFile(VcxprojFile);
  generateCompilationDatabase(BuildPath);
}
