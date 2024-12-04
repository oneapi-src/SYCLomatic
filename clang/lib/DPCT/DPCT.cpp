//===--------------- DPCT.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DPCT/DPCT.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "CommandOption/ValidateArguments.h"
#include "Config.h"
#include "ErrorHandle/CrashRecovery.h"
#include "ErrorHandle/Error.h"
#include "FileGenerator/GenFiles.h"
#include "FileGenerator/GenHelperFunction.h"
#include "IncMigration/ExternalReplacement.h"
#include "IncMigration/IncrementalMigrationUtility.h"
#include "Linux/AutoComplete.h"
#include "MigrateScript/GenMakefile.h"
#include "MigrateScript/MigrateCmakeScript.h"
#include "MigrateScript/MigratePythonBuildScript.h"
#include "MigrationAction.h"
#include "MigrationReport/Statics.h"
#include "QueryAPIMapping/QueryAPIMapping.h"
#include "RuleInfra/CallExprRewriter.h"
#include "RuleInfra/MemberExprRewriter.h"
#include "RuleInfra/TypeLocRewriters.h"
#include "RulesDNN/MapNamesDNN.h"
#include "RulesLang/MapNamesLang.h"
#include "RulesLangLib/MapNamesLangLib.h"
#include "RulesMathLib/MapNamesBlas.h"
#include "RulesMathLib/MapNamesRandom.h"
#include "UserDefinedRules/PatternRewriter.h"
#include "UserDefinedRules/UserDefinedRules.h"
#include "Utility.h"
#include "Windows/VcxprojParser.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Core/UnifiedPath.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"

#include <string>

#include "ToolChains/Cuda.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/DPCT/DpctOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::dpct;
using namespace clang::tooling;

using namespace llvm::cl;

namespace clang {
namespace tooling {
UnifiedPath getFormatSearchPath();
extern std::string ClangToolOutputMessage;
#ifdef _WIN32
extern UnifiedPath VcxprojFilePath;
#endif
} // namespace tooling
namespace dpct {
llvm::cl::OptionCategory &DPCTCat = llvm::cl::getDPCTCategory();
llvm::cl::OptionCategory &DPCTBasicCat = llvm::cl::getDPCTBasicCategory();
llvm::cl::OptionCategory &DPCTAdvancedCat = llvm::cl::getDPCTAdvancedCategory();
llvm::cl::OptionCategory &DPCTCodeGenCat = llvm::cl::getDPCTCodeGenCategory();
llvm::cl::OptionCategory &DPCTReportGenCat =
    llvm::cl::getDPCTReportGenCategory();
llvm::cl::OptionCategory &DPCTBuildScriptCat =
    llvm::cl::getDPCTBuildScriptCategory();
llvm::cl::OptionCategory &DPCTQueryAPICat = llvm::cl::getDPCTQueryAPICategory();
llvm::cl::OptionCategory &DPCTWarningsCat = llvm::cl::getDPCTWarningsCategory();
llvm::cl::OptionCategory &DPCTHelpInfoCat = llvm::cl::getDPCTHelpInfoCategory();
llvm::cl::OptionCategory &DPCTInterceptBuildCat =
    llvm::cl::getDPCTInterceptBuildCategory();
void initWarningIDs();
} // namespace dpct
} // namespace clang

// clang-format off
const char *const CtHelpMessage = DiagRef;

const char *const CtHelpHint =
    "  Warning: Please specify file(s) to be migrated.\n"
    "  To get help on the tool usage, run: dpct --help\n"
    "\n";

const char *const CmakeScriptMigrationHelpHint =
    "Warning: CMake build script file like CMakeLists.txt is not found, so no CMake build script file will be migrated.";

const char *const SetupScriptMigrationHelpHint =
    "Warning: No Python file is found, so no Python build script file will be migrated.";

const char *const BuildScriptMigrationHelpHint =
    "Warning: No CMake build script file (e.g., CMakeLists.txt or files with a .cmake suffix) or Python file was found, so no CMake or Python build script file will be migrated.";

static extrahelp CommonHelp(CtHelpMessage);

static std::string SuppressWarningsMessage = "A comma separated list of migration warnings to suppress. Valid "
                "warning IDs range\n"
                "from " + std::to_string(DiagnosticsMessage::MinID) + " to " +
                std::to_string(DiagnosticsMessage::MaxID) +
                ". Hyphen separated ranges are also allowed. For example:\n"
                "--suppress-warnings=1000-1010,1011.";

#define DPCT_OPTIONS_VAR 1
#define DPCT_OPTIONS_IN_CLANG_DPCT 1
#include "clang/DPCT/DPCTOptions.inc"

#ifdef __linux__
static AutoCompletePrinter AutoCompletePrinterInstance;
static llvm::cl::opt<AutoCompletePrinter, true, llvm::cl::parser<std::string>> AutoComplete(
  "autocomplete", llvm::cl::desc("List all options or enums which have the specified prefix.\n"),
  llvm::cl::cat(DPCTCat), llvm::cl::ReallyHidden, llvm::cl::location(AutoCompletePrinterInstance));
#endif
// clang-format on

// TODO: implement one of this for each source language.
UnifiedPath CudaPath;
UnifiedPath DpctInstallPath;

std::unordered_map<std::string, bool> ChildOrSameCache;
std::unordered_map<std::string, bool> ChildPathCache;
std::unordered_map<std::string, bool> IsDirectoryCache;
extern bool StopOnParseErrTooling;
extern UnifiedPath InRootTooling;

clang::tooling::UnifiedPath InRootPath;
clang::tooling::UnifiedPath OutRootPath;
clang::tooling::UnifiedPath CudaIncludePath;
clang::tooling::UnifiedPath SDKPath;
std::vector<clang::tooling::UnifiedPath> RuleFilePath;
std::vector<clang::tooling::UnifiedPath> AnalysisScope;

UnifiedPath getCudaInstallPath(int argc, const char **argv) {
  std::vector<const char *> Argv;
  Argv.reserve(argc);
  // do not copy "--" so the driver sees a possible SDK include path option
  std::copy_if(argv, argv + argc, back_inserter(Argv),
               [](const char *s) { return std::strcmp(s, "--"); });
  // Remove the redundant prefix "--extra-arg=" so that
  // SDK detector can find correct path.
  for (unsigned int i = 0; i < Argv.size(); i++) {
    if (strncmp(argv[i], "--extra-arg=--cuda-path", 23) == 0) {
      Argv[i] = argv[i] + 12;
    }
  }

  // Output parameters to indicate errors in parsing. Not checked here,
  // OptParser will handle errors.
  unsigned MissingArgIndex, MissingArgCount;
  MissingArgIndex = MissingArgCount = 0;
  auto &Opts = driver::getDriverOptTable();
  llvm::opt::InputArgList ParsedArgs =
      Opts.ParseArgs(Argv, MissingArgIndex, MissingArgCount);

  // Create minimalist CudaInstallationDetector and return the InstallPath.
  DiagnosticsEngine E(nullptr, nullptr, nullptr, false);
  driver::Driver Driver("", llvm::sys::getDefaultTargetTriple(), E);
  driver::CudaInstallationDetector CudaIncludeDetector(
      Driver, llvm::Triple(Driver.getTargetTriple()), ParsedArgs);

  UnifiedPath Path = CudaIncludeDetector.getIncludePath().str();
  dpct::DpctGlobalInfo::setSDKVersion(CudaIncludeDetector.version());
  if (!CudaIncludePath.getPath().empty()) {
    if (!CudaIncludeDetector.isIncludePathValid()) {
      ShowStatus(MigrationErrorInvalidCudaIncludePath);
      dpctExit(MigrationErrorInvalidCudaIncludePath);
    }
    if (!CudaIncludeDetector.isVersionSupported() &&
        !CudaIncludeDetector.isVersionPartSupported()) {
      ShowStatus(MigrationErrorCudaVersionUnsupported);
      dpctExit(MigrationErrorCudaVersionUnsupported);
    }
  } else if (!CudaIncludeDetector.isIncludePathValid()) {
    ShowStatus(MigrationErrorCannotDetectCudaPath);
    dpctExit(MigrationErrorCannotDetectCudaPath);
  } else if (!CudaIncludeDetector.isVersionSupported() &&
             !CudaIncludeDetector.isVersionPartSupported()) {
    ShowStatus(MigrationErrorDetectedCudaVersionUnsupported);
    dpctExit(MigrationErrorDetectedCudaVersionUnsupported);
  }

  if (Path.getCanonicalPath().empty()) {
    ShowStatus(MigrationErrorInvalidCudaIncludePath);
    dpctExit(MigrationErrorInvalidCudaIncludePath);
  }
  return Path;
}

static bool isCUDAHeaderRequired() { return !MigrateBuildScriptOnly; }

UnifiedPath getInstallPath(const char *invokeCommand) {
  SmallString<512> InstalledPathStr(invokeCommand);

  // Do a PATH lookup, if there are no directory components.
  if (llvm::sys::path::filename(InstalledPathStr) == InstalledPathStr) {
    if (llvm::ErrorOr<std::string> Tmp = llvm::sys::findProgramByName(
            llvm::sys::path::filename(InstalledPathStr.str()))) {
      InstalledPathStr = *Tmp;
    }
  }

  UnifiedPath InstalledPath(InstalledPathStr);
  StringRef InstalledPathParent(llvm::sys::path::parent_path(InstalledPath.getCanonicalPath()));
  // Move up to parent directory of bin directory
  InstalledPath = llvm::sys::path::parent_path(InstalledPathParent);
  return InstalledPath;
}



unsigned int GetLinesNumber(clang::tooling::RefactoringTool &Tool,
                            UnifiedPath Path) {
  // Set up Rewriter and to get source manager.
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);
  SourceManager Sources(Diagnostics, Tool.getFiles());
  Rewriter Rewrite(Sources, DefaultLangOptions);
  SourceManager &SM = Rewrite.getSourceMgr();

  auto Entry = SM.getFileManager().getOptionalFileRef(Path.getCanonicalPath());
  if (!Entry) {
    std::string ErrMsg = "FilePath Invalid...\n";
    PrintMsg(ErrMsg);
    dpctExit(MigrationErrorInvalidFilePath);
  }

  FileID FID = SM.getOrCreateFileID(*Entry, SrcMgr::C_User);

  SourceLocation EndOfFile = SM.getLocForEndOfFile(FID);
  unsigned int LineNumber = SM.getSpellingLineNumber(EndOfFile, nullptr);
  return LineNumber;
}

static void printMetrics(clang::tooling::RefactoringTool &Tool) {

  size_t Count = 0;
  for (const auto &Elem : LOCStaticsMap) {
    // Skip invalid file path.
    if (!llvm::sys::fs::exists(Elem.first))
      continue;
    unsigned TotalLines = GetLinesNumber(Tool, Elem.first);
    unsigned TransToAPI = Elem.second[0];
    unsigned TransToSYCL = Elem.second[1];
    unsigned NotTrans = TotalLines - TransToSYCL - TransToAPI;
    unsigned NotSupport = Elem.second[2];
    if (Count == 0) {
      DpctStats() << "\n";
      DpctStats() << "File name, LOC migrated to SYCL, LOC migrated to helper "
                     "functions, "
                     "LOC not needed to migrate, LOC not able to migrate";
      DpctStats() << "\n";
    }
    DpctStats() << Elem.first + ", " + std::to_string(TransToSYCL) + ", " +
                       std::to_string(TransToAPI) + ", " +
                       std::to_string(NotTrans) + ", " +
                       std::to_string(NotSupport);
    DpctStats() << "\n";
    Count++;
  }
}

static void saveApisReport(void) {
  if (ReportFilePrefix == "stdout") {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "------------------APIS report--------------------\n";
    OS << "API name\t\t\t\tFrequency";
    OS << "\n";

    for (const auto &Elem : SrcAPIStaticsMap) {
      std::string APIName = Elem.first;
      unsigned int Count = Elem.second;
      OS << llvm::format("%-30s%16u\n", APIName.c_str(), Count);
    }
    OS << "-------------------------------------------------\n";
    PrintMsg(OS.str());
  } else {
    std::string RFile = appendPath(
        OutRootPath.getCanonicalPath().str(),
        ReportFilePrefix + (ReportFormat.getValue() == ReportFormatEnum::RFE_CSV
                                ? ".apis.csv"
                                : ".apis.log"));

    createDirectories(llvm::sys::path::parent_path(RFile));
    RawFDOStream File(RFile);

    File << (ReportFormat.getValue() == ReportFormatEnum::RFE_CSV
                 ? " API name, Frequency "
                 : "API name\t\t\t\tFrequency");

    File << "\n";
    for (const auto &Elem : SrcAPIStaticsMap) {
      std::string APIName = Elem.first;
      unsigned int Count = Elem.second;
      if (ReportFormat.getValue() == ReportFormatEnum::RFE_CSV) {
        File << "\"" << APIName << "\"," << std::to_string(Count) << "\n";
      } else {
        File << llvm::format("%-30s%16u\n", APIName.c_str(), Count);
      }
    }
  }
}

static void saveStatsReport(clang::tooling::RefactoringTool &Tool,
                            double Duration) {

  printMetrics(Tool);
  DpctStats() << "\nTotal migration time: " + std::to_string(Duration) +
                     " ms\n";
  if (ReportFilePrefix == "stdout") {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "----------Stats report---------------\n";
    OS << getDpctStatsStr() << "\n";
    OS << "-------------------------------------\n";
    PrintMsg(OS.str());
  } else {
    std::string RFile = appendPath(
        OutRootPath.getCanonicalPath().str(),
        ReportFilePrefix + (ReportFormat.getValue() == ReportFormatEnum::RFE_CSV
                                ? ".stats.csv"
                                : ".stats.log"));
    createDirectories(llvm::sys::path::parent_path(RFile));
    writeDataToFile(RFile, getDpctStatsStr() + "\n");
  }
}

static void saveDiagsReport() {

  // DpctDiags() << "\n";
  if (ReportFilePrefix == "stdout") {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "--------Diags message----------------\n";
    OS << getDpctDiagsStr() << "\n";
    OS << "-------------------------------------\n";
    PrintMsg(OS.str());
  } else {
    std::string RFile = appendPath(OutRootPath.getCanonicalPath().str(),
                                   ReportFilePrefix + ".diags.log");
    createDirectories(llvm::sys::path::parent_path(RFile));
    writeDataToFile(RFile, getDpctStatsStr() + "\n");
  }
}

std::string printCTVersion() {

  std::string buf;
  llvm::raw_string_ostream OS(buf);

  OS << "\n"
     << TOOL_NAME << " version " << getDpctVersionStr() << "."
     << " Codebase:";
  std::string Revision = getClangRevision();
  if (!Revision.empty()) {
    OS << '(';
    if (!Revision.empty()) {
      OS << Revision;
    }
    OS << ").";
  }

  OS << " clang version " << CLANG_VERSION_MAJOR << "." << CLANG_VERSION_MINOR
     << "." << CLANG_VERSION_PATCHLEVEL << "\n";

  return OS.str();
}

static void DumpOutputFile(void) {
  // Redirect stdout/stderr output to <file> if option "-output-file" is set
  if (!OutputFile.empty()) {
    std::string FilePath =
        appendPath(OutRootPath.getCanonicalPath().str(), OutputFile);
    createDirectories(llvm::sys::path::parent_path(FilePath));
    writeDataToFile(FilePath, getDpctTermStr() + "\n");
  }
}

void PrintReportOnFault(const std::string &FaultMsg) {
  PrintMsg(FaultMsg);
  saveApisReport();
  saveDiagsReport();

  if (ReportFilePrefix == "stdcout")
    return;

  std::string FileApis = appendPath(
      OutRootPath.getCanonicalPath().str(),
      ReportFilePrefix + (ReportFormat.getValue() == ReportFormatEnum::RFE_CSV
                              ? ".apis.csv"
                              : ".apis.log"));
  std::string FileDiags = appendPath(OutRootPath.getCanonicalPath().str(),
                                     ReportFilePrefix + ".diags.log");

  appendDataToFile(FileApis, FaultMsg);
  appendDataToFile(FileDiags, FaultMsg);
  DumpOutputFile();
}

void parseFormatStyle() {
  StringRef StyleStr = "file"; // DPCTFormatStyle::Custom
  if (clang::dpct::DpctGlobalInfo::getFormatStyle() ==
      DPCTFormatStyle::FS_Google) {
    StyleStr = "google";
  } else if (clang::dpct::DpctGlobalInfo::getFormatStyle() ==
             DPCTFormatStyle::FS_LLVM) {
    StyleStr = "llvm";
  }
  UnifiedPath StyleSearchPath =
      clang::tooling::getFormatSearchPath().getCanonicalPath().empty()
          ? clang::dpct::DpctGlobalInfo::getInRoot()
          : clang::tooling::getFormatSearchPath();
  llvm::Expected<clang::format::FormatStyle> StyleOrErr =
      clang::format::getStyle(StyleStr, StyleSearchPath.getCanonicalPath(),
                              "llvm");
  clang::format::FormatStyle Style;
  if (!StyleOrErr) {
    PrintMsg(llvm::toString(StyleOrErr.takeError()) + "\n");
    PrintMsg("Using LLVM style as fallback formatting style.\n");
    clang::format::FormatStyle FallbackStyle = clang::format::getNoStyle();
    getPredefinedStyle("llvm", clang::format::FormatStyle::LanguageKind::LK_Cpp,
                       &FallbackStyle);
    Style = FallbackStyle;
  } else {
    Style = StyleOrErr.get();
  }

  DpctGlobalInfo::setCodeFormatStyle(Style);
}

static void loadMainSrcFileInfo(clang::tooling::UnifiedPath OutRoot) {
  std::string YamlFilePath = appendPath(OutRoot.getCanonicalPath().str(),
                                        DpctGlobalInfo::getYamlFileName());
  auto PreTU = std::make_shared<clang::tooling::TranslationUnitReplacements>();
  if (llvm::sys::fs::exists(YamlFilePath)) {
    if (loadFromYaml(YamlFilePath, *PreTU) != 0) {
      llvm::errs() << getLoadYamlFailWarning(YamlFilePath);
    }
  }
  for (auto &Entry : PreTU->MainSourceFilesDigest) {
    if (Entry.HasCUDASyntax)
      MainSrcFilesHasCudaSyntex.insert(Entry.MainSourceFile);
  }

  // Currently, when "--use-experimental-features=device_global" and
  // "--use-experimental-features=all" are specified, the migrated code should
  // be compiled with C++20 or later.
  auto Iter = PreTU->OptionMap.find("ExperimentalFlag");
  if (Iter != PreTU->OptionMap.end()) {
    if (Iter->second.Specified) {
      const std::string Value = Iter->second.Value;
      unsigned int UValue = std::stoul(Value);
      if (UValue & (1 << static_cast<unsigned>(
                        ExperimentalFeatures::Exp_DeviceGlobal))) {
        LANG_Cplusplus_20_Used = true;
      }
    }
  }
}

void processPathToHelperFunctionAndExit(const char **argv) {
  auto FindHelperPath = [&](const char *Cmd) {
      SmallString<512> Path;
      Path = getInstallPath(Cmd).getCanonicalPath();
      llvm::sys::path::append(Path, "include");
      if (!llvm::sys::fs::exists(Path))
        return false;
      else if (UseSYCLCompat) {
        auto CompatPath = Path;
        llvm::sys::path::append(CompatPath, "syclcompat");
        if (!llvm::sys::fs::exists(CompatPath))
          return false;
      }
      DpctLog() << Path << '\n';
      return true;
    };
    auto Ret = MigrationSucceeded;
    if (UseSYCLCompat) {
      auto Success = FindHelperPath("clang");
      Success |= FindHelperPath("icpx");
      if (!Success) {
        DpctLog() << "SYCLcompat is usually installed in include folder of "
                     "SYCL compiler.\n";
        Ret = MigrationErrorInvalidInstallPath;
      }
    } else if (!FindHelperPath(argv[0])) {
      Ret = MigrationErrorInvalidInstallPath;
    }
    ShowStatus(Ret, "Helper functions");
    dpctExit(Ret);
}

void callIndependentToolAndExit(const std::string IndependentTool, int argc, const char **argv) {
  SmallString<512> ExecutableScriptPath(DpctInstallPath.getCanonicalPath());
  llvm::sys::path::append(ExecutableScriptPath, "bin", IndependentTool);
  if (!llvm::sys::fs::exists(ExecutableScriptPath)) {
    ShowStatus(MigrationErrorInvalidInstallPath, IndependentTool + " tool");
    dpctExit(MigrationErrorInvalidInstallPath);
  }
  std::string Python = GetPython();
  if (Python.empty()) {
    ShowStatus(CallIndependentToolError, "python");
    dpctExit(CallIndependentToolError);
  }
  std::string SystemCallCommand =
      Python + " " + std::string(ExecutableScriptPath.str());
  for (int Index = 2; Index < argc; Index++) {
    SystemCallCommand.append(" ");
    SystemCallCommand.append(std::string(argv[Index]));
  }
  int ProcessExitCode = system(SystemCallCommand.c_str());
  if (ProcessExitCode) {
    ShowStatus(CallIndependentToolError, std::move(IndependentTool));
    dpctExit(CallIndependentToolError);
  }
  dpctExit(CallIndependentToolSucceeded);
}

void showReportHeader() {
  std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "Generate report: "
       << "report-type:"
       << (ReportType.getValue() == ReportTypeEnum::RTE_All
               ? "all"
               : (ReportType.getValue() == ReportTypeEnum::RTE_APIs
                      ? "apis"
                      : (ReportType.getValue() == ReportTypeEnum::RTE_Stats
                             ? "stats"
                             : "diags")))
       << ", report-format:"
       << (ReportFormat.getValue() == ReportFormatEnum::RFE_CSV ? "csv"
                                                                : "formatted")
       << ", report-file-prefix:" << ReportFilePrefix << "\n";

    PrintMsg(OS.str());
}

void checkIncMigrationOrExit() {
  if (!MigrateBuildScriptOnly &&
      clang::dpct::DpctGlobalInfo::isIncMigration()) {
    std::string Msg;
    if (!canContinueMigration(Msg)) {
      ShowStatus(MigrationErrorDifferentOptSet, Msg);
      dpctExit(MigrationErrorDifferentOptSet, false);
    }
  }
}

int migrateCmakeScript(const clang::tooling::UnifiedPath &InRoot,
                       const clang::tooling::UnifiedPath &OutRoot) {
  if (!cmakeScriptNotFound()) {
    runWithCrashGuard(
        [&]() { doCmakeScriptMigration(InRoot, OutRoot); },
        "Error: dpct internal error. Migrating CMake scripts in \"" +
            InRootPath.getCanonicalPath().str() +
            "\" causing the error skipped. Migration continues.\n");
  }
  return MigrationSucceeded;
}

int migratePythonScript(const clang::tooling::UnifiedPath &InRoot,
                        const clang::tooling::UnifiedPath &OutRoot) {
  if (!pythonBuildScriptNotFound()) {
    runWithCrashGuard(
        [&]() { doPythonBuildScriptMigration(InRoot, OutRoot); },
        "Error: dpct internal error. Migrating Python build scripts in \"" +
            InRoot.getCanonicalPath().str() +
            "\" causing the error skipped. Migration continues.\n");
  }
  return MigrationSucceeded;
}

// print APIMapping of Query
int showAPIMapping(std::string SrcAPI, std::string Option,
                   RefactoringTool &Tool, ReplTy &ReplSYCL) {
  llvm::outs() << "CUDA API:" << llvm::raw_ostream::GREEN << SrcAPI
               << llvm::raw_ostream::RESET;
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      IntrusiveRefCntPtr<DiagnosticOptions>(new DiagnosticOptions()));
  SourceManager Sources(Diagnostics, Tool.getFiles());
  LangOptions DefaultLangOptions;
  Rewriter Rewrite(Sources, DefaultLangOptions);
  // Must be only 1 file.
  tooling::applyAllReplacements(ReplSYCL.begin()->second, Rewrite);
  const auto &RewriteBuffer = Rewrite.buffer_begin()->second;
  static const std::string StartStr{"// Start"};
  static const std::string EndStr{"// End"};
  std::string MigratedStr{""};
  bool Flag = false;
  for (auto I = RewriteBuffer.begin(), E = RewriteBuffer.end(); I != E;
       I.MoveToNextPiece()) {
    size_t StartPos = 0;
    if (!Flag) {
      if (auto It = I.piece().find(StartStr); It != StringRef::npos) {
        StartPos = It + StartStr.length();
        Flag = true;
      }
    }
    if (Flag) {
      size_t EndPos = I.piece().size();
      if (auto It = I.piece().find(EndStr); It != StringRef::npos) {
        auto TempStr = I.piece().substr(0, It);
        EndPos = TempStr.find_last_of('\n') + 1;
        Flag = false;
      }
      MigratedStr += I.piece().substr(StartPos, EndPos - StartPos);
    }
  }

  // For some cuda error handling APIs (currently only cudaGetErrorString), we
  // use NoRewriteRewriter to do migration. So the comment in the argument
  // part is kept in the migrated code. We remove those comments here.
  // ATTENTION: There is A SPACE at the beginning of each comment.
  static const std::unordered_set<std::string> CommentsNeedBeRemoved = {
      " /*cudaError_t*/"};
  for (const auto &Comment : CommentsNeedBeRemoved) {
    size_t RemoveStartPos = MigratedStr.find(Comment);
    if (RemoveStartPos != std::string::npos) {
      MigratedStr.erase(RemoveStartPos, Comment.length());
      break;
    }
  }

  if (MigratedStr.find_first_not_of(" \n") == std::string::npos) {
    llvm::outs() << "The API is Removed.\n";
  } else {
    llvm::outs() << "Is migrated to" << Option << ":" << llvm::raw_ostream::BLUE
                 << MigratedStr << llvm::raw_ostream::RESET;
  }
  return MigrationSucceeded;
}

int runDPCT(int argc, const char **argv) {

  if (argc < 2) {
    std::cout << CtHelpHint;
    return MigrationErrorShowHelp;
  }
  clang::dpct::initCrashRecovery();

  clang::dpct::DpctOptionBase::init();

#if defined(_WIN32)
  // To support wildcard "*" in source file name in windows.
  llvm::InitLLVM X(argc, argv);
#endif
  // Set handle for libclangTooling to process message for dpct
  clang::tooling::SetPrintHandle(PrintMsg);
  clang::tooling::SetFileSetInCompilationDB(
      dpct::DpctGlobalInfo::getFileSetInCompilationDB());

  // CommonOptionsParser will adjust argc to the index of "--"
  int OriginalArgc = argc;
  clang::tooling::SetModuleFiles(dpct::DpctGlobalInfo::getModuleFiles());
#ifdef _WIN32
  // Set function handle for libclangTooling to parse vcxproj file.
  clang::tooling::SetParserHandle(vcxprojParser);
#endif
  llvm::cl::SetVersionPrinter(
      [](llvm::raw_ostream &OS) { OS << printCTVersion() << "\n"; });
  auto OptParser =
      CommonOptionsParser::create(argc, argv, DPCTCat, llvm::cl::OneOrMore);
  if (!OptParser) {
    if (OptParser.errorIsA<DPCTError>()) {
      llvm::Error NewE =
          handleErrors(OptParser.takeError(), [](const DPCTError &DE) {
            if (DE.EC == -101) {
              ShowStatus(MigrationErrorCannotParseDatabase);
              dpctExit(MigrationErrorCannotParseDatabase);
            } else if (DE.EC == -102) {
              ShowStatus(MigrationErrorCannotFindDatabase);
              dpctExit(MigrationErrorCannotFindDatabase);
            } else {
              ShowStatus(MigrationError);
              dpctExit(MigrationError);
            }
          });
    }
    // Filter and output error messages emitted by clang
    auto E =
        handleErrors(OptParser.takeError(), [](const llvm::StringError &E) {
          DpctLog() << E.getMessage();
        });
    dpct::ShowStatus(MigrationOptionParsingError);
    dpctExit(MigrationOptionParsingError);
  }
  // Option check: like conflict
  DpctOptionBase::check();
  if (UseSYCLCompat && USMLevel.getValue() == UsmLevel::UL_None) {
    llvm::errs()
        << "Currently SYCLcompat header-only library (syclcompat:: namespace) "
           "doesn't support buffer and accessor data management..\n";
    ShowStatus(MigrationErrorConflictOptions);
    dpctExit(MigrationErrorConflictOptions);
  }

  DpctInstallPath = getInstallPath(argv[0]);

  InRootPath = InRoot;
  OutRootPath = OutRoot;
  std::string OutRootPathCUDACodepin = "";
  CudaIncludePath = CudaInclude;
  SDKPath = SDKPathOpt;
  std::transform(
      RuleFile.begin(), RuleFile.end(),
      std::back_insert_iterator<std::vector<clang::tooling::UnifiedPath>>(
          RuleFilePath),
      [](const std::string &Str) { return clang::tooling::UnifiedPath(Str); });
  std::transform(
      AnalysisScopeOpt.begin(), AnalysisScopeOpt.end(),
      std::back_insert_iterator<std::vector<clang::tooling::UnifiedPath>>(
          AnalysisScope),
      [](const std::string &Str) { return clang::tooling::UnifiedPath(Str); });

  // Action: just show -- --help information and then exit
  if (CommonOptionsParser::hasHelpOption(OriginalArgc, argv))
    dpctExit(MigrationSucceeded);

  // OC_Action
  if (PathToHelperFunction) {
    processPathToHelperFunctionAndExit(argv);
  }

  if (!OutputFile.empty()) {
    // Set handle for libclangTooling to redirect warning message to DpctTerm
    clang::tooling::SetDiagnosticOutput(DpctTerm());
  }
  initWarningIDs();

#ifndef _WIN32
  // OC_Action
  if (InterceptBuildCommand)
    callIndependentToolAndExit("intercept-build", argc, argv);
#endif
  // OC_Action
  if (CodePinReport)
    callIndependentToolAndExit("codepin-report.py", argc, argv);

  if (AnalysisMode)
    DpctGlobalInfo::enableAnalysisMode();

  // Check Option Values...
  validateInputDirectoryLengthOrExit("--in-root", InRootPath);
  validateInputDirectoryLengthOrExit("--out-root", OutRootPath);
  std::for_each(AnalysisScope.begin(), AnalysisScope.end(),
                [](const clang::tooling::UnifiedPath &P) {
                  validateInputDirectoryLengthOrExit("--analysis-scope-path",
                                                     P);
                });
  validateInputDirectoryLengthOrExit("--cuda-include-path", CudaIncludePath);
  validateInputDirectoryLengthOrExit("--output-file", OutputFile);
  // Report file prefix is limited to 128, so that <report-type> and
  // <report-format> can be extended later
  checkOptionLengthLimitOrExit("--report-file-prefix", ReportFilePrefix);
  checkSpecialCharsOrExit("--report-file-prefix", ReportFilePrefix);

  clock_t StartTime = clock();

  if (LimitChangeExtension) {
    DpctGlobalInfo::addChangeExtensions(".cu");
    DpctGlobalInfo::addChangeExtensions(".cuh");
  }

  if (InRootPath.getPath().empty() && ProcessAll) {
    ShowStatus(MigrationErrorNoExplicitInRoot);
    dpctExit(MigrationErrorNoExplicitInRoot);
  }

  if (MigrateBuildScriptOnly) {
    if (InRootPath.getPath().empty() &&
        !buildScriptFileSpecified(OptParser->getSourcePathList())) {
      ShowStatus(MigrationErrorNoExplicitInRootAndBuildScript);
      dpctExit(MigrationErrorNoExplicitInRootAndBuildScript);
    }
  }

  if (!makeInRootCanonicalOrSetDefaults(InRootPath,
                                        OptParser->getSourcePathList())) {
    ShowStatus(MigrationErrorInvalidInRootOrOutRoot);
    dpctExit(MigrationErrorInvalidInRootOrOutRoot);
  }

  if (!MigrateBuildScriptOnly) {
    int ValidPath = validatePaths(InRootPath, OptParser->getSourcePathList());
    if (ValidPath == -1) {
      ShowStatus(MigrationErrorInvalidInRootPath);
      dpctExit(MigrationErrorInvalidInRootPath);
    } else if (ValidPath == -2) {
      ShowStatus(MigrationErrorNoFileTypeAvail);
      dpctExit(MigrationErrorNoFileTypeAvail);
    }

    if (buildScriptFileSpecified(OptParser->getSourcePathList())) {
      ShowStatus(MigrateBuildScriptOnlyNotSpecifed);
      dpctExit(MigrateBuildScriptOnlyNotSpecifed);
    }

  } else {
    // To validate the path of CMake or Python build script file or directory
    int ValidPath =
        validateBuildScriptPaths(InRootPath, OptParser->getSourcePathList());
    if (ValidPath == -1) {
      ShowStatus(MigrationErrorInvalidInRootPath);
      dpctExit(MigrationErrorInvalidInRootPath);
    } else if (ValidPath < -1) {
      ShowStatus(MigrationErrorBuildScriptPathInvalid);
      dpctExit(MigrationErrorBuildScriptPathInvalid);
    }
  }

  if ((BuildScript == BuildScriptKind::BS_Cmake ||
       BuildScript == BuildScriptKind::BS_Python) &&
      !OptParser->getSourcePathList().empty()) {
    ShowStatus(MigrateBuildScriptIncorrectUse);
    dpctExit(MigrateBuildScriptIncorrectUse);
  }
  if ((BuildScript == BuildScriptKind::BS_Cmake ||
       BuildScript == BuildScriptKind::BS_Python) &&
      MigrateBuildScriptOnly) {
    ShowStatus(MigrateBuildScriptAndMigrateBuildScriptOnlyBothUse);
    dpctExit(MigrateBuildScriptAndMigrateBuildScriptOnlyBothUse);
  }

  int SDKIncPathRes = checkSDKPathOrIncludePath(CudaIncludePath);
  if (SDKIncPathRes == -1) {
    ShowStatus(MigrationErrorInvalidCudaIncludePath);
    dpctExit(MigrationErrorInvalidCudaIncludePath);
  } else if (SDKIncPathRes == 0) {
    RealSDKIncludePath = CudaIncludePath.getCanonicalPath();
    HasSDKIncludeOption = true;
  }

  int SDKPathRes = checkSDKPathOrIncludePath(SDKPath);
  if (SDKPathRes == -1) {
    ShowStatus(MigrationErrorInvalidCudaIncludePath);
    dpctExit(MigrationErrorInvalidCudaIncludePath);
  } else if (SDKPathRes == 0) {
    RealSDKPath = SDKPath.getCanonicalPath();
    HasSDKPathOption = true;
  }

  bool GenReport = false;
#ifdef DPCT_DEBUG_BUILD
  std::string &DVerbose = DiagsContent;
#else
  std::string DVerbose = "";
#endif
  if (!checkReportArgs(ReportType.getValue(), ReportFormat.getValue(),
                       ReportFilePrefix, ReportOnly, GenReport, DVerbose)) {
    ShowStatus(MigrationErrorInvalidReportArgs);
    dpctExit(MigrationErrorInvalidReportArgs);
  }

  if (GenReport)
    showReportHeader();
  
  ExtraIncPaths = OptParser->getExtraIncPathList();

  if (isCUDAHeaderRequired()) {
    // TODO: implement one of this for each source language.
    CudaPath = getCudaInstallPath(OriginalArgc, argv);
    DpctDiags() << "Cuda Include Path found: " << CudaPath.getCanonicalPath()
                << "\n";
  }
  // set source code
  std::vector<std::string> SourcePathList;
  if (QueryAPIMapping.getNumOccurrences()) {
    if (QueryAPIMapping.getNumOccurrences() > 1) {
      llvm::outs()
          << "Warning: Option --query-api-mapping is specified multi times, "
             "only the last one is used, all other are ignored.\n";
    }
    // Set a virtual file for --query-api-mapping.
    llvm::SmallString<16> VirtFolderSS;
    llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, VirtFolderSS);
    UnifiedPath VirtFolderPath(VirtFolderSS);

    // Need set a virtual path and it will used by AnalysisScope.
    InRootPath = VirtFolderPath;

    llvm::SmallString<16> VirtFileSS(VirtFolderPath.getCanonicalPath());
    llvm::sys::path::append(VirtFileSS, "temp.cu");
    SourcePathList.emplace_back(VirtFileSS);
    DpctGlobalInfo::setIsQueryAPIMapping(true);
  } else {
    SourcePathList = OptParser->getSourcePathList();
  }

  // Refactoring Tool
  RefactoringTool Tool(OptParser->getCompilations(), SourcePathList);
  std::string QueryAPIMappingSrc;
  std::string QueryAPIMappingOpt;
  if (DpctGlobalInfo::isQueryAPIMapping()) {
    APIMapping::setPrintAll(QueryAPIMapping == "-");
    APIMapping::initEntryMap();
    if (APIMapping::getPrintAll()) {
      APIMapping::printAll();
      dpctExit(MigrationSucceeded);
    }
    auto SourceCode = APIMapping::getAPISourceCode(QueryAPIMapping);
    if (SourceCode.empty()) {
      ShowStatus(MigrationErrorNoAPIMapping);
      dpctExit(MigrationErrorNoAPIMapping);
    }

    Tool.mapVirtualFile(SourcePathList[0], SourceCode);

    static const std::string OptionStr{"// Option:"};
    if (SourceCode.starts_with(OptionStr)) {
      QueryAPIMappingOpt += " (with the option";
      while (SourceCode.consume_front(OptionStr)) {
        auto Option = SourceCode.substr(0, SourceCode.find_first_of('\n'));
        Option = Option.trim(' ');
        SourceCode = SourceCode.substr(SourceCode.find_first_of('\n') + 1);
        QueryAPIMappingOpt += " ";
        QueryAPIMappingOpt += Option.str();
        if (Option.starts_with("--use-dpcpp-extensions")) {
          if (Option.ends_with("intel_device_math"))
            UseDPCPPExtensions.addValue(
                DPCPPExtensionsDefaultDisabled::ExtDD_IntelDeviceMath);
        } else if (Option.starts_with("--use-experimental-features")) {
          if (Option.ends_with("bfloat16_math_functions"))
            Experimentals.addValue(ExperimentalFeatures::Exp_BFloat16Math);
          else if (Option.ends_with("occupancy-calculation"))
            Experimentals.addValue(
                ExperimentalFeatures::Exp_OccupancyCalculation);
          else if (Option.ends_with("free-function-queries"))
            Experimentals.addValue(ExperimentalFeatures::Exp_FreeQueries);
          else if (Option.ends_with("logical-group"))
            Experimentals.addValue(ExperimentalFeatures::Exp_LogicalGroup);
          else if (Option.ends_with("root-group"))
            Experimentals.addValue(ExperimentalFeatures::Exp_RootGroup);
          else if (Option.ends_with("masked-sub-group-operation"))
            Experimentals.addValue(
                ExperimentalFeatures::Exp_MaskedSubGroupFunction);
          else if (Option.ends_with("bindless_images"))
            Experimentals.addValue(ExperimentalFeatures::Exp_BindlessImages);
          else if (Option.ends_with("graph"))
            Experimentals.addValue(ExperimentalFeatures::Exp_Graph);
        } else if (Option == "--no-dry-pattern") {
          NoDRYPattern.setValue(true);
        }
        // Need add more option.
      }
      QueryAPIMappingOpt += ")";
    }

    static const std::string StartStr{"// Start"};
    static const std::string EndStr{"// End"};
    auto StartPos = SourceCode.find(StartStr);
    auto EndPos = SourceCode.find(EndStr);
    if (StartPos == StringRef::npos || EndPos == StringRef::npos) {
      dpctExit(MigrationErrorNoAPIMapping);
    }
    StartPos = StartPos + StartStr.length();
    EndPos = SourceCode.find_last_of('\n', EndPos);
    QueryAPIMappingSrc =
        SourceCode.substr(StartPos, EndPos - StartPos + 1).str();
    // Print static migration info for cudaGraphicsD3D11RegisterResource
    // on linux, as the API is unavailable on Linux and hence, no API mapping.
#if defined(__linux__)
    if (QueryAPIMapping == "cudaGraphicsD3D11RegisterResource") {
      const std::string MigratedToStr =
          "r = new dpct::experimental::external_mem_wrapper(pD3Dr, f);";
      llvm::outs() << "CUDA API:" << llvm::raw_ostream::GREEN
                   << QueryAPIMappingSrc << llvm::raw_ostream::RESET;
      llvm::outs() << "On Windows, is migrated to" << QueryAPIMappingOpt << ":"
                   << llvm::raw_ostream::BLUE << "\n  " << MigratedToStr << "\n"
                   << llvm::raw_ostream::RESET;
      dpctExit(MigrationSucceeded);
    }
#endif // __linux__
    static const std::string MigrateDesc{"// Migration desc: "};
    auto MigrateDescPos = SourceCode.find(MigrateDesc);
    if (MigrateDescPos != StringRef::npos) {
      auto MigrateDescBegin = MigrateDescPos + MigrateDesc.length();
      auto MigrateDescEnd = SourceCode.find_first_of('\n', MigrateDescPos);
      llvm::outs() << "CUDA API:" << llvm::raw_ostream::GREEN
                   << QueryAPIMappingSrc << llvm::raw_ostream::RESET
                   << SourceCode.substr(MigrateDescBegin,
                                        MigrateDescEnd - MigrateDescBegin + 1);
      dpctExit(MigrationSucceeded);
    }

    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-w"));
    NoIncrementalMigration.setValue(true);
    StopOnParseErr.setValue(true);
    Tool.setPrintErrorMessage(false);
  } else {
    IsUsingDefaultOutRoot = OutRootPath.getPath().empty();
    bool NeedCheckOutRootEmpty = !(BuildScript == BuildScriptKind::BS_Cmake ||
                                   BuildScript == BuildScriptKind::BS_Python) &&
                                 !MigrateBuildScriptOnly;
    if (!DpctGlobalInfo::isAnalysisModeEnabled() && IsUsingDefaultOutRoot &&
        !getDefaultOutRoot(OutRootPath, NeedCheckOutRootEmpty) && !EnableCodePin) {
      ShowStatus(MigrationErrorInvalidInRootOrOutRoot);
      dpctExit(MigrationErrorInvalidInRootOrOutRoot, false);
    }
    if (EnableCodePin) {
      OutRootPathCUDACodepin =  OutRootPath.getPath().str()  + "_codepin_cuda";
      OutRootPath = OutRootPath.getPath().str() + "_codepin_sycl";
    }

    dpct::DpctGlobalInfo::setOutRoot(OutRootPath);
  }

  if (GenBuildScript) {
    clang::tooling::SetCompileTargetsMap(CompileTargetsMap);
  }

  UnifiedPath CompilationsDir(OptParser->getCompilationsDir());

  Tool.setCompilationDatabaseDir(CompilationsDir.getCanonicalPath().str());

  if (isCUDAHeaderRequired())
    validateInputDirectory(InRootPath);

  // AnalysisScope defaults to the value of InRoot
  // InRoot must be the same as or child of AnalysisScope
  if (AnalysisScope.empty())
    AnalysisScope.push_back(InRootPath);
  if (!std::all_of(AnalysisScope.begin(), AnalysisScope.end(),
                   [](clang::tooling::UnifiedPath &P) {
                     return makeAnalysisScopeCanonicalOrSetDefaults(P,
                                                                    InRootPath);
                   }) ||
      (!InRootPath.getPath().empty() &&
       !std::any_of(AnalysisScope.begin(), AnalysisScope.end(),
                    [](const clang::tooling::UnifiedPath &P) {
                      return isChildOrSamePath(P, InRootPath);
                    }))) {
    ShowStatus(MigrationErrorInvalidAnalysisScope);
    dpctExit(MigrationErrorInvalidAnalysisScope);
  }

  if (isCUDAHeaderRequired())
    std::for_each(AnalysisScope.begin(), AnalysisScope.end(),
                  validateInputDirectory);

  if (GenHelperFunction.getValue()) {
    dpct::genHelperFunction(dpct::DpctGlobalInfo::getOutRoot());
  }

  // Add extra parser options
  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-nocudalib", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "--cuda-host-only", ArgumentInsertPosition::BEGIN));

  if (DefineCUDAVerMajorMinor) {
    std::string CUDAVerMajor =
        "-D__CUDACC_VER_MAJOR__=" + std::to_string(SDKVersionMajor);
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
        CUDAVerMajor.c_str(), ArgumentInsertPosition::BEGIN));
    std::string CUDAVerMinor =
        "-D__CUDACC_VER_MINOR__=" + std::to_string(SDKVersionMinor);
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
        CUDAVerMinor.c_str(), ArgumentInsertPosition::BEGIN));
  }
  std::string CUDADotHFilePathMacro =
      "-D__CUDA_DOT_H_FILE_PATH__=\"" +
      appendPath(CudaPath.getCanonicalPath().str(), "cuda.h") + "\"";
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      CUDADotHFilePathMacro.c_str(), ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "-fno-delayed-template-parsing", ArgumentInsertPosition::END));

  SetSDKIncludePath(CudaPath.getCanonicalPath().str());

#ifdef _WIN32
  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
                                ArgumentInsertPosition::BEGIN));
#endif
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "-fcuda-allow-variadic-functions", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-Xclang", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "-Wno-c++11-narrowing", ArgumentInsertPosition::BEGIN));

  // Init Global analysis info
  DpctGlobalInfo::setInRoot(InRootPath);
  DpctGlobalInfo::setOutRoot(OutRootPath);
  DpctGlobalInfo::setAnalysisScope(AnalysisScope);
  DpctGlobalInfo::setCudaPath(CudaPath);
  DpctGlobalInfo::setKeepOriginCode(KeepOriginalCode);
  DpctGlobalInfo::setSyclNamedLambda(SyclNamedLambda);
  DpctGlobalInfo::setUsmLevel(USMLevel);
  DpctGlobalInfo::setBuildScript(BuildScript);
  // When enable codepin feature, the incremental migration will be disabled.
  DpctGlobalInfo::setIsIncMigration(!NoIncrementalMigration && !EnableCodePin &&
                                    !MigrateBuildScriptOnly);
  DpctGlobalInfo::setCheckUnicodeSecurityFlag(CheckUnicodeSecurity);
  DpctGlobalInfo::setEnablepProfilingFlag(EnablepProfiling);
  DpctGlobalInfo::setFormatRange(FormatRng);
  DpctGlobalInfo::setFormatStyle(FormatST);
  DpctGlobalInfo::setCtadEnabled(EnableCTAD);
  DpctGlobalInfo::setCodePinEnabled(EnableCodePin);
  DpctGlobalInfo::setGenBuildScriptEnabled(GenBuildScript);
  DpctGlobalInfo::setMigrateBuildScriptOnlyEnabled(MigrateBuildScriptOnly);
  DpctGlobalInfo::setCommentsEnabled(EnableComments);
  DpctGlobalInfo::setHelperFuncPreferenceFlag(Preferences.getBits());
  DpctGlobalInfo::setUsingDRYPattern(!NoDRYPattern);
  DpctGlobalInfo::setExperimentalFlag(Experimentals.getBits());
  DpctGlobalInfo::setExtensionDEFlag(~(NoDPCPPExtensions.getBits()));
  DpctGlobalInfo::setExtensionDDFlag(UseDPCPPExtensions.getBits());
  DpctGlobalInfo::setAssumedNDRangeDim(
      (NDRangeDim == AssumedNDRangeDimEnum::ARE_Dim1) ? 1 : 3);
  DpctGlobalInfo::setOptimizeMigrationFlag(OptimizeMigration.getValue());
  DpctGlobalInfo::setSYCLFileExtension(SYCLFileExtension);
  DpctGlobalInfo::setUseSYCLCompat(UseSYCLCompat);
  StopOnParseErrTooling = StopOnParseErr;
  InRootTooling = InRootPath;

  if (ExcludePathList.getNumOccurrences()) {
    DpctGlobalInfo::setExcludePath(ExcludePathList);
  }

  std::set<clang::dpct::ExplicitNamespace> ExplicitNamespaces;
  if (UseExplicitNamespace.getNumOccurrences()) {
    ExplicitNamespaces.insert(UseExplicitNamespace.begin(),
                              UseExplicitNamespace.end());
  } else {
    ExplicitNamespaces.insert({UseSYCLCompat ? ExplicitNamespace::EN_SYCLCompat
                                             : ExplicitNamespace::EN_DPCT,
                               ExplicitNamespace::EN_SYCL});
  }
  MapNames::setExplicitNamespaceMap(ExplicitNamespaces);
  MapNamesLang::setExplicitNamespaceMap(ExplicitNamespaces);
  MapNamesBlas::setExplicitNamespaceMap(ExplicitNamespaces);
  MapNamesDNN::setExplicitNamespaceMap(ExplicitNamespaces);
  MapNamesLangLib::setExplicitNamespaceMap(ExplicitNamespaces);
  MapNamesRandom::setExplicitNamespaceMap(ExplicitNamespaces);

  // Init migration rules infrasturecture.
  CallExprRewriterFactoryBase::initRewriterMap();
  TypeLocRewriterFactoryBase::initTypeLocRewriterMap();
  MemberExprRewriterFactoryBase::initMemberExprRewriterMap();
  clang::dpct::initHeaderSpellings();

  // load user defind rules in case.
  if (MigrateBuildScriptOnly ||
      DpctGlobalInfo::getBuildScript() == BuildScriptKind::BS_Cmake) {
    SmallString<128> FilePath1(DpctInstallPath.getCanonicalPath());
    llvm::sys::path::append(FilePath1,
                            Twine("extensions/cmake_rules/"
                                  "cmake_script_migration_rule.yaml"));
    SmallString<128> FilePath2(DpctInstallPath.getCanonicalPath());
    llvm::sys::path::append(FilePath2,
                            Twine("opt/dpct/extensions/cmake_rules/"
                                  "cmake_script_migration_rule.yaml"));

    std::vector<clang::tooling::UnifiedPath> CmakeRuleFiles{
        llvm::sys::fs::exists(FilePath1) ? FilePath1.c_str()
                                         : FilePath2.c_str()};
    importRules(CmakeRuleFiles);
    dpct::genCmakeHelperFunction(dpct::DpctGlobalInfo::getOutRoot());
  }

  if (MigrateBuildScriptOnly ||
      DpctGlobalInfo::getBuildScript() == BuildScriptKind::BS_Python) {
    SmallString<128> PythonRuleFilePath(DpctInstallPath.getCanonicalPath());
    llvm::sys::path::append(
        PythonRuleFilePath,
        Twine("extensions/python_rules/"
              "python_build_script_migration_rule_ipex.yaml"));
    if (llvm::sys::fs::exists(PythonRuleFilePath)) {
      std::vector<clang::tooling::UnifiedPath> PythonRuleFiles{
          PythonRuleFilePath};
      importRules(PythonRuleFiles);
      // generage helper functions file in the outroot dir here
    }
  }

  if (!RuleFilePath.empty()) {
    importRules(RuleFilePath);
  }

  {
    // init globalOptionMap
    setValueToOptMap(clang::dpct::OPTION_AsyncHandler, AsyncHandler.getValue(),
                     AsyncHandler.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_NDRangeDim,
                     static_cast<unsigned int>(NDRangeDim.getValue()),
                     NDRangeDim.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_CommentsEnabled,
                     DpctGlobalInfo::isCommentsEnabled(),
                     EnableComments.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_CtadEnabled,
                     DpctGlobalInfo::isCtadEnabled(),
                     EnableCTAD.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_CodePinEnabled,
                     DpctGlobalInfo::isCodePinEnabled(),
                     EnableCodePin.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_ExtensionDEFlag,
                     DpctGlobalInfo::getExtensionDEFlag(),
                     NoDPCPPExtensions.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_ExtensionDDFlag,
                     DpctGlobalInfo::getExtensionDDFlag(),
                     UseDPCPPExtensions.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_NoDRYPattern, NoDRYPattern.getValue(),
                     NoDRYPattern.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_CompilationsDir, CompilationsDir,
                     OptParser->isPSpecified());
#ifdef _WIN32
    if (!VcxprojFilePath.getPath().empty()) {
      setValueToOptMap(clang::dpct::OPTION_VcxprojFile,
                       VcxprojFilePath.getCanonicalPath().str(),
                       OptParser->isVcxprojfileSpecified());
    } else {
      setValueToOptMap(clang::dpct::OPTION_VcxprojFile,
                       VcxprojFilePath.getPath().str(),
                       OptParser->isVcxprojfileSpecified());
    }
#endif
    setValueToOptMap(clang::dpct::OPTION_ProcessAll, ProcessAll.getValue(),
                     ProcessAll.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_SyclNamedLambda, SyclNamedLambda.getValue(),
                     SyclNamedLambda.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_ExperimentalFlag,
                     DpctGlobalInfo::getExperimentalFlag(),
                     Experimentals.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_HelperFuncPreferenceFlag,
                     DpctGlobalInfo::getHelperFuncPreferenceFlag(),
                     Preferences.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_ExplicitNamespace,
                     ExplicitNamespaces,
                     UseExplicitNamespace.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_UsmLevel,
                     static_cast<unsigned int>(DpctGlobalInfo::getUsmLevel()),
                     USMLevel.getNumOccurrences());
    setValueToOptMap(
        clang::dpct::OPTION_BuildScript,
        static_cast<unsigned int>(DpctGlobalInfo::getBuildScript()),
        BuildScript.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_OptimizeMigration,
                     OptimizeMigration.getValue(),
                     OptimizeMigration.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_EnablepProfiling, EnablepProfiling.getValue(),
                     EnablepProfiling.getValue());
    setValueToOptMap(clang::dpct::OPTION_RuleFile, MetaRuleObject::RuleFiles,
                     RuleFile.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_AnalysisScopePath,
                     DpctGlobalInfo::getAnalysisScope(),
                     AnalysisScopeOpt.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_UseSYCLCompat, UseSYCLCompat.getValue(),
                     UseSYCLCompat.getNumOccurrences());

    checkIncMigrationOrExit();
  }

  if (DpctGlobalInfo::getFormatRange() != clang::format::FormatRange::none) {
    parseFormatStyle();
  }
  // OC_Action: only migrate Build scripts.
  if (MigrateBuildScriptOnly) {
    loadMainSrcFileInfo(OutRootPath);
    collectCmakeScriptsSpecified(OptParser, InRootPath, OutRootPath);
    collectPythonBuildScriptsSpecified(OptParser, InRootPath, OutRootPath);
    migrateCmakeScript(InRootPath, OutRootPath);
    migratePythonScript(InRootPath, OutRootPath);
    if (cmakeScriptNotFound() && pythonBuildScriptNotFound()) {
      std::cout << BuildScriptMigrationHelpHint << "\n";
    }
    ShowStatus(MigrationBuildScriptCompleted);
    dpctExit(MigrationSucceeded, false);
  }

  ReplTy ReplCUDA, ReplSYCL;
  volatile int RunCount = 0;
  do {
    if (RunCount == 1) {
      // Currently, we just need maximum two passes
      DpctGlobalInfo::setNeedRunAgain(false);
      DpctGlobalInfo::getInstance().resetInfo();
      DeviceFunctionDecl::reset();
    }
    DpctGlobalInfo::setRunRound(RunCount++);
    // DPCT Action
    DpctToolAction Action(OutputFile.empty() &&
                                  !DpctGlobalInfo::isQueryAPIMapping()
                              ? llvm::errs()
                              : DpctTerm(),
                          ReplCUDA, ReplSYCL, Passes,
                          {PassKind::PK_Analysis, PassKind::PK_Migration},
                          Tool.getFiles().getVirtualFileSystemPtr());

    if (ProcessAll) {
      clang::tooling::SetFileProcessHandle(InRootPath.getCanonicalPath(),
                                           OutRootPath.getCanonicalPath(),
                                           processAllFiles);
    }
    // Migrate Action: Parse code to Translation unit or cache the Invocation
    int RunResult = Tool.run(&Action);
    if (RunResult == MigrationErrorCannotAccessDirInDatabase) {
      ShowStatus(MigrationErrorCannotAccessDirInDatabase,
                 ClangToolOutputMessage);
      return MigrationErrorCannotAccessDirInDatabase;
    } else if (RunResult == MigrationErrorInconsistentFileInDatabase) {
      ShowStatus(MigrationErrorInconsistentFileInDatabase,
                 ClangToolOutputMessage);
      return MigrationErrorInconsistentFileInDatabase;
    }

    do {
      if (RunResult && StopOnParseErr) {
        DumpOutputFile();
        if (RunResult == 1) {
          if (DpctGlobalInfo::isQueryAPIMapping()) {
            std::string Err = getDpctTermStr();
            StringRef ErrStr = Err;
            // Avoid the "Visual Studio version" error on windows platform.
            if (ErrStr.find("error:") == ErrStr.rfind("error:") &&
                ErrStr.contains(
                    "error -- unsupported Microsoft Visual Studio version")) {
              break;
            }
            if (ErrStr.contains("use of undeclared identifier")) {
              ShowStatus(MigrationErrorAPIMappingWrongCUDAHeader,
                         QueryAPIMapping);
              return MigrationErrorAPIMappingWrongCUDAHeader;
            } else if (ErrStr.contains("file not found")) {
              ShowStatus(MigrationErrorAPIMappingNoCUDAHeader, QueryAPIMapping);
              return MigrationErrorAPIMappingNoCUDAHeader;
            }
            ShowStatus(MigrationErrorNoAPIMapping);
            dpctExit(MigrationErrorNoAPIMapping);
          }
          ShowStatus(MigrationErrorFileParseError);
          return MigrationErrorFileParseError;
        } else {
          // When RunResult equals to 2, it means no error, but some files are
          // skipped due to missing compile commands.
          // And clang::tooling::ReFactoryTool will emit error message.
          return MigrationSKIPForMissingCompileCommand;
        }
      }
    } while (0);

    Action.runPasses();
  } while (DpctGlobalInfo::isNeedRunAgain());

  // OC_Action: QueryAPI mapping: show mapping result
  if (DpctGlobalInfo::isQueryAPIMapping()) {
    return showAPIMapping(QueryAPIMappingSrc, QueryAPIMappingOpt, Tool,
                          ReplSYCL);
  }

  // OC_Action: Analysis mode
  if (DpctGlobalInfo::isAnalysisModeEnabled()) {
    if (AnalysisModeOutputFile.getValue().empty()) {
      dumpAnalysisModeStatics(llvm::outs());
    } else {
      dpct::RawFDOStream Out(AnalysisModeOutputFile);
      Out.enable_colors(false);
      dumpAnalysisModeStatics(Out);
    }
    return MigrationSucceeded;
  }

  // OC_Action: Migrate code : Migration report
  if (GenReport) {
    // report: apis, stats, all, diags
    if (ReportType.getValue() == ReportTypeEnum::RTE_All ||
        ReportType.getValue() == ReportTypeEnum::RTE_APIs)
      saveApisReport();

    if (ReportType.getValue() == ReportTypeEnum::RTE_All ||
        ReportType.getValue() == ReportTypeEnum::RTE_Stats) {
      clock_t EndTime = clock();
      double Duration = (double)(EndTime - StartTime) / (CLOCKS_PER_SEC / 1000);
      saveStatsReport(Tool, Duration);
    }
    // all doesn't include diags.
    if (ReportType.getValue() == ReportTypeEnum::RTE_Diags) {
      saveDiagsReport();
    }
    if (ReportOnly) {
      DumpOutputFile();
      return MigrationSucceeded;
    }
  }

  // OC_Action: Migrate code : Generate migrated src files
  int Status = 0;
  runWithCrashGuard(
      [&]() {
        Status = saveNewFiles(Tool, InRootPath, OutRootPath,
                              OutRootPathCUDACodepin, ReplCUDA, ReplSYCL);
      },
      "Error: dpct internal error. Saving in \"" +
          OutRootPath.getCanonicalPath().str() +
          "\" causing the error skipped. Migration continues.\n");

  // OC_Action: Migrate CMake scripts after Code Migration
  if (DpctGlobalInfo::getBuildScript() == BuildScriptKind::BS_Cmake) {
    loadMainSrcFileInfo(OutRootPath);
    collectCmakeScripts(InRootPath, OutRootPath);
    migrateCmakeScript(InRootPath, OutRootPath);
    if (cmakeScriptNotFound()) {
      std::cout << CmakeScriptMigrationHelpHint << "\n";
    }
  }

  if (DpctGlobalInfo::getBuildScript() == BuildScriptKind::BS_Python) {
    loadMainSrcFileInfo(OutRootPath);
    collectPythonBuildScripts(InRootPath, OutRootPath);
    migratePythonScript(InRootPath, OutRootPath);
    if (pythonBuildScriptNotFound()) {
      std::cout << SetupScriptMigrationHelpHint << "\n";
    }
  }

  ShowStatus(Status);
  DumpOutputFile();
  return Status;
}

int run(int argc, const char **argv) {
  int Status = runDPCT(argc, argv);
  if (IsUsingDefaultOutRoot) {
    removeDefaultOutRootFolder(OutRootPath.getCanonicalPath());
  }
  return Status;
}
