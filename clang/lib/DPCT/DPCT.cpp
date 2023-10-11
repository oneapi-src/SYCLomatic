//===--------------- DPCT.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DPCT/DPCT.h"
#include "APIMapping/QueryAPIMapping.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "AutoComplete.h"
#include "CallExprRewriter.h"
#include "Config.h"
#include "CrashRecovery.h"
#include "ExternalReplacement.h"
#include "GenHelperFunction.h"
#include "GenMakefile.h"
#include "IncrementalMigrationUtility.h"
#include "MemberExprRewriter.h"
#include "MigrationAction.h"
#include "MisleadingBidirectional.h"
#include "Rules.h"
#include "SaveNewFiles.h"
#include "Statics.h"
#include "TypeLocRewriters.h"
#include "Utility.h"
#include "ValidateArguments.h"
#include "VcxprojParser.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
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
#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include <signal.h>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::dpct;
using namespace clang::tooling;

using namespace llvm::cl;

namespace clang {
namespace tooling {
std::string getFormatSearchPath();
extern std::string ClangToolOutputMessage;
#ifdef _WIN32
extern std::string VcxprojFilePath;
#endif
} // namespace tooling
namespace dpct {
llvm::cl::OptionCategory &DPCTCat = llvm::cl::getDPCTCategory();
void initWarningIDs();
} // namespace dpct
} // namespace clang

// clang-format off
const char *const CtHelpMessage =
    "\n"
    "<source0> ... Paths of input source files. These paths are looked up in "
    "the compilation database.\n\n"
    "EXAMPLES:\n\n"
    "Migrate single source file:\n\n"
    "  dpct source.cpp\n\n"
    "Migrate single source file with C++11 features:\n\n"
    "  dpct --extra-arg=\"-std=c++11\" source.cpp\n\n"
    "Migrate all files available in compilation database:\n\n"
    "  dpct --compilation-database=<path to location of compilation database file>\n\n"
    "Migrate one file in compilation database:\n\n"
    "  dpct --compilation-database=<path to location of compilation database file>  source.cpp\n\n"
#if defined(_WIN32)
    "Migrate all files available in vcxprojfile:\n\n"
    "  dpct --vcxprojfile=path/to/vcxprojfile.vcxproj\n"
#endif
    DiagRef
    ;

const char *const CtHelpHint =
    "  Warning: Please specify file(s) to be migrated.\n"
    "  To get help on the tool usage, run: dpct --help\n"
    "\n";

static extrahelp CommonHelp(CtHelpMessage);

bool ReportOnlyFlag = false;
bool KeepOriginalCodeFlag = false;
bool SuppressWarningsAllFlag = false;
bool StopOnParseErr = false;
bool CheckUnicodeSecurityFlag = false;
bool EnablepProfilingFlag = false;
bool SyclNamedLambdaFlag = false;
bool ExplicitClNamespace = false;
bool NoDRYPatternFlag = false;
bool ProcessAllFlag = false;
bool AsyncHandlerFlag = false;
static std::string SuppressWarningsMessage = "A comma separated list of migration warnings to suppress. Valid "
                "warning IDs range\n"
                "from " + std::to_string(DiagnosticsMessage::MinID) + " to " +
                std::to_string(DiagnosticsMessage::MaxID) +
                ". Hyphen separated ranges are also allowed. For example:\n"
                "--suppress-warnings=1000-1010,1011.";

#define DPCT_OPTIONS_IN_CLANG_DPCT
#define DPCT_OPT_TYPE(...) __VA_ARGS__
#define DPCT_OPT_ENUM(NAME, ...)                                   \
llvm::cl::OptionEnumValue{NAME, __VA_ARGS__}
#define DPCT_OPTION_VALUES(...)                                    \
llvm::cl::values(__VA_ARGS__)
#define DPCT_NON_ENUM_OPTION(OPT_TYPE, OPT_VAR, OPTION_NAME, ...)  \
OPT_TYPE OPT_VAR(OPTION_NAME, __VA_ARGS__);
#define DPCT_ENUM_OPTION(OPT_TYPE, OPT_VAR, OPTION_NAME, ...)      \
OPT_TYPE OPT_VAR(OPTION_NAME, __VA_ARGS__);
#include "clang/DPCT/DPCTOptions.inc"
#undef DPCT_ENUM_OPTION
#undef DPCT_NON_ENUM_OPTION
#undef DPCT_OPTION_VALUES
#undef DPCT_OPT_ENUM
#undef DPCT_OPT_TYPE
#undef DPCT_OPTIONS_IN_CLANG_DPCT

static llvm::cl::opt<std::string> SDKPath("cuda-path", desc("Directory path of SDK.\n"),
                                llvm::cl::value_desc("dir"), llvm::cl::cat(DPCTCat),
                                llvm::cl::Optional, llvm::cl::Hidden);
static llvm::cl::opt<std::string> Passes(
    "passes",
    llvm::cl::desc("Comma separated list of migration passes, which will be applied.\n"
         "Only the specified passes are applied."),
    llvm::cl::value_desc("IterationSpaceBuiltinRule,..."), llvm::cl::cat(DPCTCat),
               llvm::cl::Hidden);
#ifdef DPCT_DEBUG_BUILD
static llvm::cl::opt<std::string>
    DiagsContent("report-diags-content",
                 llvm::cl::desc("Diagnostics verbosity level. \"pass\": Basic migration "
                      "pass information. "
                      "\"transformation\": Detailed migration pass "
                      "transformation information."),
                 llvm::cl::value_desc("[pass|transformation]"), llvm::cl::cat(DPCTCat),
                 llvm::cl::Optional, llvm::cl::Hidden);
#endif
#ifdef __linux__
static AutoCompletePrinter AutoCompletePrinterInstance;
static llvm::cl::opt<AutoCompletePrinter, true, llvm::cl::parser<std::string>> AutoComplete(
  "autocomplete", llvm::cl::desc("List all options or enums which have the specified prefix.\n"),
  llvm::cl::cat(DPCTCat), llvm::cl::ReallyHidden, llvm::cl::location(AutoCompletePrinterInstance));
#endif
// clang-format on

// TODO: implement one of this for each source language.
std::string CudaPath;
std::string DpctInstallPath;
std::unordered_map<std::string, bool> ChildOrSameCache;
std::unordered_map<std::string, bool> ChildPathCache;
std::unordered_map<std::string, llvm::SmallString<256>> RealPathCache;
std::unordered_map<std::string, bool> IsDirectoryCache;
extern bool StopOnParseErrTooling;
extern std::string InRootTooling;

std::string getCudaInstallPath(int argc, const char **argv) {
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

  std::string Path = CudaIncludeDetector.getInstallPath().str();
  dpct::DpctGlobalInfo::setSDKVersion(CudaIncludeDetector.version());

  if (!CudaIncludePath.empty()) {
    if (!CudaIncludeDetector.isIncludePathValid()) {
      ShowStatus(MigrationErrorInvalidCudaIncludePath);
      dpctExit(MigrationErrorInvalidCudaIncludePath);
    }

    if (!CudaIncludeDetector.isVersionSupported()) {
      ShowStatus(MigrationErrorCudaVersionUnsupported);
      dpctExit(MigrationErrorCudaVersionUnsupported);
    }
  } else if (!CudaIncludeDetector.isIncludePathValid()) {
    ShowStatus(MigrationErrorCannotDetectCudaPath);
    dpctExit(MigrationErrorCannotDetectCudaPath);
  } else if (!CudaIncludeDetector.isVersionSupported()) {
    ShowStatus(MigrationErrorDetectedCudaVersionUnsupported);
    dpctExit(MigrationErrorDetectedCudaVersionUnsupported);

  }

  makeCanonical(Path);

  SmallString<512> CudaPathAbs;
  std::error_code EC = llvm::sys::fs::real_path(Path, CudaPathAbs, true);
  if ((bool)EC) {
    ShowStatus(MigrationErrorInvalidCudaIncludePath);
    dpctExit(MigrationErrorInvalidCudaIncludePath);
  }
  return CudaPathAbs.str().str();
}

std::string getInstallPath(const char *invokeCommand) {
  SmallString<512> InstalledPath(invokeCommand);

  // Do a PATH lookup, if there are no directory components.
  if (llvm::sys::path::filename(InstalledPath) == InstalledPath) {
    if (llvm::ErrorOr<std::string> Tmp = llvm::sys::findProgramByName(
            llvm::sys::path::filename(InstalledPath.str()))) {
      InstalledPath = *Tmp;
    }
  }

  makeCanonical(InstalledPath);
  llvm::sys::fs::real_path(InstalledPath, InstalledPath, true);
  StringRef InstalledPathParent(llvm::sys::path::parent_path(InstalledPath));
  // Move up to parent directory of bin directory
  StringRef InstallPath = llvm::sys::path::parent_path(InstalledPathParent);

  SmallString<512> InstallPathAbs;
  std::error_code EC =
      llvm::sys::fs::real_path(InstallPath, InstallPathAbs, true);
  if ((bool)EC) {
    ShowStatus(MigrationErrorInvalidInstallPath);
    dpctExit(MigrationErrorInvalidInstallPath);
  }
  return InstallPathAbs.str().str();
}

// To validate the root path of the project to be migrated.
void ValidateInputDirectory(std::string &InRoot) {

  if (isChildOrSamePath(CudaPath, InRoot)) {
    ShowStatus(MigrationErrorRunFromSDKFolder);
    dpctExit(MigrationErrorRunFromSDKFolder);
  }

  if (isChildOrSamePath(InRoot, CudaPath)) {
    ShowStatus(MigrationErrorInputDirContainSDKFolder);
    dpctExit(MigrationErrorInputDirContainSDKFolder);
  }

  if (isChildOrSamePath(InRoot, DpctInstallPath)) {
    ShowStatus(MigrationErrorInputDirContainCTTool);
    dpctExit(MigrationErrorInputDirContainCTTool);
  }
}

unsigned int GetLinesNumber(clang::tooling::RefactoringTool &Tool,
                            StringRef Path) {
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

  const FileEntry *Entry = SM.getFileManager().getFile(Path).get();
  if (!Entry) {
    std::string ErrMsg = "FilePath Invalid...\n";
    PrintMsg(ErrMsg);
    dpctExit(MigrationErrorInvalidFilePath);
  }

  FileID FID = SM.getOrCreateFileID(Entry, SrcMgr::C_User);

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
    std::string RFile =
        OutRoot + "/" + ReportFilePrefix +
        (ReportFormat.getValue() == ReportFormatEnum::RFE_CSV ? ".apis.csv"
                                                              : ".apis.log");
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(RFile, std::ios::binary);

    std::string Str;
    llvm::raw_string_ostream Title(Str);
    Title << (ReportFormat.getValue() == ReportFormatEnum::RFE_CSV
                  ? " API name, Frequency "
                  : "API name\t\t\t\tFrequency");

    File << Title.str() << std::endl;
    for (const auto &Elem : SrcAPIStaticsMap) {
      std::string APIName = Elem.first;
      unsigned int Count = Elem.second;
      if (ReportFormat.getValue() == ReportFormatEnum::RFE_CSV) {
        File << "\"" << APIName << "\"," << std::to_string(Count) << std::endl;
      } else {
        std::string Str;
        llvm::raw_string_ostream OS(Str);
        OS << llvm::format("%-30s%16u\n", APIName.c_str(), Count);
        File << OS.str();
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
    std::string RFile =
        OutRoot + "/" + ReportFilePrefix +
        (ReportFormat.getValue() == ReportFormatEnum::RFE_CSV ? ".stats.csv"
                                                              : ".stats.log");
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(RFile, std::ios::binary);
    File << getDpctStatsStr() << "\n";
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
    std::string RFile = OutRoot + "/" + ReportFilePrefix + ".diags.log";
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(RFile, std::ios::binary);
    File << getDpctDiagsStr() << "\n";
  }
}

std::string printCTVersion() {

  std::string buf;
  llvm::raw_string_ostream OS(buf);

  OS << "\n"
     << TOOL_NAME << " version " << DPCT_VERSION_MAJOR << "."
     << DPCT_VERSION_MINOR << "." << DPCT_VERSION_PATCH << "."
     << " Codebase:";
  // getClangRepositoryPath() export the machine name of repo in release build.
  // so skip the repo name.
  std::string Path = "";
  std::string Revision = getClangRevision();
  if (!Path.empty() || !Revision.empty()) {
    OS << '(';
    if (!Path.empty())
      OS << Path;
    if (!Revision.empty()) {
      if (!Path.empty())
        OS << ' ';
      OS << Revision;
    }
    OS << ')';
  }

  OS << "\n";
  return OS.str();
}

static void DumpOutputFile(void) {
  // Redirect stdout/stderr output to <file> if option "-output-file" is set
  if (!OutputFile.empty()) {
    std::string FilePath = OutRoot + "/" + OutputFile;
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(FilePath));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(FilePath, std::ios::binary);
    File << getDpctTermStr() << "\n";
  }
}

void PrintReportOnFault(const std::string &FaultMsg) {
  PrintMsg(FaultMsg);
  saveApisReport();
  saveDiagsReport();

  if (ReportFilePrefix == "stdcout")
    return;

  std::string FileApis =
      OutRoot + "/" + ReportFilePrefix +
      (ReportFormat.getValue() == ReportFormatEnum::RFE_CSV ? ".apis.csv"
                                                            : ".apis.log");
  std::string FileDiags = OutRoot + "/" + ReportFilePrefix + ".diags.log";

  std::ofstream File;
  File.open(FileApis, std::ios::app);
  if (File) {
    File << FaultMsg;
    File.close();
  }

  File.open(FileDiags, std::ios::app);
  if (File) {
    File << FaultMsg;
    File.close();
  }

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
  std::string StyleSearchPath = clang::tooling::getFormatSearchPath().empty()
                                    ? clang::dpct::DpctGlobalInfo::getInRoot()
                                    : clang::tooling::getFormatSearchPath();
  llvm::Expected<clang::format::FormatStyle> StyleOrErr =
      clang::format::getStyle(StyleStr, StyleSearchPath, "llvm");
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

int runDPCT(int argc, const char **argv) {

  if (argc < 2) {
    std::cout << CtHelpHint;
    return MigrationErrorShowHelp;
  }
  clang::dpct::initCrashRecovery();

#if defined(_WIN32)
  // To support wildcard "*" in source file name in windows.
  llvm::InitLLVM X(argc, argv);
#endif

  // Set handle for libclangTooling to process message for dpct
  clang::tooling::SetPrintHandle(PrintMsg);
  clang::tooling::SetFileSetInCompiationDB(
      dpct::DpctGlobalInfo::getFileSetInCompiationDB());

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

  if (!OutputFile.empty()) {
    // Set handle for libclangTooling to redirect warning message to DpctTerm
    clang::tooling::SetDiagnosticOutput(DpctTerm());
  }

  initWarningIDs();

  DpctInstallPath = getInstallPath(argv[0]);

  if (PathToHelperFunction) {
    SmallString<512> pathToHelperFunction(DpctInstallPath);
    llvm::sys::path::append(pathToHelperFunction, "include");
    if (!llvm::sys::fs::exists(pathToHelperFunction)) {
      DpctLog() << "Error: Helper functions not found"
                << "/n";
      ShowStatus(MigrationErrorInvalidInstallPath);
      dpctExit(MigrationErrorInvalidInstallPath);
    }
    std::cout << pathToHelperFunction.c_str() << "\n";
    ShowStatus(MigrationSucceeded);
    dpctExit(MigrationSucceeded);
  }

  if (InRoot.size() >= MAX_PATH_LEN - 1) {
    DpctLog() << "Error: --in-root '" << InRoot << "' is too long\n";
    ShowStatus(MigrationErrorPathTooLong);
    dpctExit(MigrationErrorPathTooLong);
  }
  if (OutRoot.size() >= MAX_PATH_LEN - 1) {
    DpctLog() << "Error: --out-root '" << OutRoot << "' is too long\n";
    ShowStatus(MigrationErrorPathTooLong);
    dpctExit(MigrationErrorPathTooLong);
  }
  if (AnalysisScope.size() >= MAX_PATH_LEN - 1) {
    DpctLog() << "Error: --analysis-scope-path '" << AnalysisScope << "' is too long\n";
    ShowStatus(MigrationErrorPathTooLong);
    dpctExit(MigrationErrorPathTooLong);
  }
  if (CudaIncludePath.size() >= MAX_PATH_LEN - 1) {
    DpctLog() << "Error: --cuda-include-path '" << CudaIncludePath
              << "' is too long\n";
    ShowStatus(MigrationErrorPathTooLong);
    dpctExit(MigrationErrorPathTooLong);
  }
  if (OutputFile.size() >= MAX_PATH_LEN - 1) {
    DpctLog() << "Error: --output-file '" << OutputFile << "' is too long\n";
    ShowStatus(MigrationErrorPathTooLong);
    dpctExit(MigrationErrorPathTooLong);
  }
  // Report file prefix is limited to 128, so that <report-type> and
  // <report-format> can be extended later
  if (ReportFilePrefix.size() >= 128) {
    DpctLog() << "Error: --report-file-prefix '" << ReportFilePrefix
              << "' is too long\n";
    ShowStatus(MigrationErrorPrefixTooLong);
    dpctExit(MigrationErrorPrefixTooLong);
  }
  auto P = std::find_if_not(
      ReportFilePrefix.begin(), ReportFilePrefix.end(),
      [](char C) { return ::isalpha(C) || ::isdigit(C) || C == '_'; });
  if (P != ReportFilePrefix.end()) {
    DpctLog() << "Error: --report-file-prefix contains special character '"
              << *P << "' \n";
    ShowStatus(MigrationErrorSpecialCharacter);
    dpctExit(MigrationErrorSpecialCharacter);
  }
  clock_t StartTime = clock();
  // just show -- --help information and then exit
  if (CommonOptionsParser::hasHelpOption(OriginalArgc, argv))
    dpctExit(MigrationSucceeded);

  if (LimitChangeExtension) {
    DpctGlobalInfo::addChangeExtensions(".cu");
    DpctGlobalInfo::addChangeExtensions(".cuh");
  }

  if (InRoot.empty() && ProcessAllFlag) {
    ShowStatus(MigrationErrorNoExplicitInRoot);
    dpctExit(MigrationErrorNoExplicitInRoot);
  }

  if (!makeInRootCanonicalOrSetDefaults(InRoot,
                                        OptParser->getSourcePathList())) {
    ShowStatus(MigrationErrorInvalidInRootOrOutRoot);
    dpctExit(MigrationErrorInvalidInRootOrOutRoot);
  }

  int ValidPath = validatePaths(InRoot, OptParser->getSourcePathList());
  if (ValidPath == -1) {
    ShowStatus(MigrationErrorInvalidInRootPath);
    dpctExit(MigrationErrorInvalidInRootPath);
  } else if (ValidPath == -2) {
    ShowStatus(MigrationErrorNoFileTypeAvail);
    dpctExit(MigrationErrorNoFileTypeAvail);
  }

  int SDKIncPathRes =
      checkSDKPathOrIncludePath(CudaIncludePath, RealSDKIncludePath);
  if (SDKIncPathRes == -1) {
    ShowStatus(MigrationErrorInvalidCudaIncludePath);
    dpctExit(MigrationErrorInvalidCudaIncludePath);
  } else if (SDKIncPathRes == 0) {
    HasSDKIncludeOption = true;
  }

  int SDKPathRes = checkSDKPathOrIncludePath(SDKPath, RealSDKPath);
  if (SDKPathRes == -1) {
    ShowStatus(MigrationErrorInvalidCudaIncludePath);
    dpctExit(MigrationErrorInvalidCudaIncludePath);
  } else if (SDKPathRes == 0) {
    HasSDKPathOption = true;
  }

  bool GenReport = false;
#ifdef DPCT_DEBUG_BUILD
  std::string &DVerbose = DiagsContent;
#else
  std::string DVerbose = "";
#endif
  if (checkReportArgs(ReportType.getValue(), ReportFormat.getValue(),
                      ReportFilePrefix, ReportOnlyFlag, GenReport,
                      DVerbose) == false) {
    ShowStatus(MigrationErrorInvalidReportArgs);
    dpctExit(MigrationErrorInvalidReportArgs);
  }

  if (GenReport) {
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

  ExtraIncPaths = OptParser->getExtraIncPathList();

  // TODO: implement one of this for each source language.
  CudaPath = getCudaInstallPath(OriginalArgc, argv);
  DpctDiags() << "Cuda Include Path found: " << CudaPath << "\n";

  std::vector<std::string> SourcePathList;
  if (QueryAPIMapping.getNumOccurrences()) {
    // Set a virtual file for --query-api-mapping.
    llvm::SmallString<16> VirtFile;
    llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, VirtFile);
    // Need set a virtual path and it will used by AnalysisScope.
    InRoot = VirtFile.str().str();
    makeInRootCanonicalOrSetDefaults(InRoot, {});
    VirtFile = InRoot;
    llvm::sys::path::append(VirtFile, "temp.cu");
    SourcePathList.emplace_back(VirtFile);
    DpctGlobalInfo::setIsQueryAPIMapping(true);
  } else {
    SourcePathList = OptParser->getSourcePathList();
  }
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
        } else if (Option == "--no-dry-pattern") {
          NoDRYPatternFlag = true;
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
    NoIncrementalMigration = true;
    StopOnParseErr = true;
    Tool.setPrintErrorMessage(false);
  } else {
    IsUsingDefaultOutRoot = OutRoot.empty();
    if (!makeOutRootCanonicalOrSetDefaults(OutRoot)) {
      ShowStatus(MigrationErrorInvalidInRootOrOutRoot);
      dpctExit(MigrationErrorInvalidInRootOrOutRoot, false);
    }
    dpct::DpctGlobalInfo::setOutRoot(OutRoot);
  }

  if (GenBuildScript) {
    clang::tooling::SetCompileTargetsMap(CompileTargetsMap);
  }

  std::string CompilationsDir = OptParser->getCompilationsDir();
  if (!CompilationsDir.empty()) {
    // To convert the relative path to absolute path.
    llvm::SmallString<128> AbsPath(CompilationsDir);
    llvm::sys::fs::make_absolute(AbsPath);
    llvm::sys::path::remove_dots(AbsPath, /*remove_dot_dot=*/true);
    CompilationsDir = std::string(AbsPath.str());
  }

  Tool.setCompilationDatabaseDir(CompilationsDir);
  ValidateInputDirectory(InRoot);

  // AnalysisScope defaults to the value of InRoot
  // InRoot must be the same as or child of AnalysisScope
  if (!makeAnalysisScopeCanonicalOrSetDefaults(AnalysisScope, InRoot) ||
      (!InRoot.empty() && !isChildOrSamePath(AnalysisScope, InRoot))) {
    ShowStatus(MigrationErrorInvalidAnalysisScope);
    dpctExit(MigrationErrorInvalidAnalysisScope);
  }
  ValidateInputDirectory(AnalysisScope);

  if (GenHelperFunction.getValue()) {
    dpct::genHelperFunction(dpct::DpctGlobalInfo::getOutRoot());
  }

  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-nocudalib", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "--cuda-host-only", ArgumentInsertPosition::BEGIN));

  std::string CUDAVerMajor =
      "-D__CUDACC_VER_MAJOR__=" + std::to_string(SDKVersionMajor);
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      CUDAVerMajor.c_str(), ArgumentInsertPosition::BEGIN));

  std::string CUDAVerMinor =
      "-D__CUDACC_VER_MINOR__=" + std::to_string(SDKVersionMinor);
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      CUDAVerMinor.c_str(), ArgumentInsertPosition::BEGIN));
  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-D__NVCC__", ArgumentInsertPosition::BEGIN));

  SetSDKIncludePath(CudaPath);

#ifdef _WIN32
  if ((SDKVersionMajor == 11 && SDKVersionMinor == 2) ||
      (SDKVersionMajor == 11 && SDKVersionMinor == 3) ||
      (SDKVersionMajor == 11 && SDKVersionMinor == 4) ||
      (SDKVersionMajor == 11 && SDKVersionMinor == 5) ||
      (SDKVersionMajor == 11 && SDKVersionMinor == 6) ||
      (SDKVersionMajor == 11 && SDKVersionMinor == 7) ||
      (SDKVersionMajor == 11 && SDKVersionMinor == 8) ||
      (SDKVersionMajor == 12 && SDKVersionMinor == 0) ||
      (SDKVersionMajor == 12 && SDKVersionMinor == 1) ||
      (SDKVersionMajor == 12 && SDKVersionMinor == 2)) {
    Tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster("-fms-compatibility-version=19.21.27702.0",
                                  ArgumentInsertPosition::BEGIN));
  } else {
    Tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster("-fms-compatibility-version=19.00.24215.1",
                                  ArgumentInsertPosition::BEGIN));
  }
#endif
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "-fcuda-allow-variadic-functions", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-Xclang", ArgumentInsertPosition::BEGIN));

  DpctGlobalInfo::setInRoot(InRoot);
  DpctGlobalInfo::setOutRoot(OutRoot);
  DpctGlobalInfo::setAnalysisScope(AnalysisScope);
  DpctGlobalInfo::setCudaPath(CudaPath);
  DpctGlobalInfo::setKeepOriginCode(KeepOriginalCodeFlag);
  DpctGlobalInfo::setSyclNamedLambda(SyclNamedLambdaFlag);
  DpctGlobalInfo::setUsmLevel(USMLevel);
  DpctGlobalInfo::setIsIncMigration(!NoIncrementalMigration);
  DpctGlobalInfo::setCheckUnicodeSecurityFlag(CheckUnicodeSecurityFlag);
  DpctGlobalInfo::setEnablepProfilingFlag(EnablepProfilingFlag);
  DpctGlobalInfo::setFormatRange(FormatRng);
  DpctGlobalInfo::setFormatStyle(FormatST);
  DpctGlobalInfo::setCtadEnabled(EnableCTAD);
  DpctGlobalInfo::setGenBuildScriptEnabled(GenBuildScript);
  DpctGlobalInfo::setCommentsEnabled(EnableComments);
  DpctGlobalInfo::setHelperFuncPreferenceFlag(Preferences.getBits());
  DpctGlobalInfo::setUsingDRYPattern(!NoDRYPatternFlag);
  DpctGlobalInfo::setExperimentalFlag(Experimentals.getBits());
  DpctGlobalInfo::setExtensionDEFlag(~(NoDPCPPExtensions.getBits()));
  DpctGlobalInfo::setExtensionDDFlag(UseDPCPPExtensions.getBits());
  DpctGlobalInfo::setAssumedNDRangeDim(
      (NDRangeDim == AssumedNDRangeDimEnum::ARE_Dim1) ? 1 : 3);
  DpctGlobalInfo::setOptimizeMigrationFlag(OptimizeMigration.getValue());
  StopOnParseErrTooling = StopOnParseErr;
  InRootTooling = InRoot;

  if (ExcludePathList.getNumOccurrences()) {
    DpctGlobalInfo::setExcludePath(ExcludePathList);
  }

  std::vector<ExplicitNamespace> DefaultExplicitNamespaces = {
      ExplicitNamespace::EN_SYCL, ExplicitNamespace::EN_DPCT};
  if (NoClNamespaceInline.getNumOccurrences()) {
    if (UseExplicitNamespace.getNumOccurrences()) {
      DpctGlobalInfo::setExplicitNamespace(UseExplicitNamespace);
      clang::dpct::PrintMsg(
          "Note: Option --no-cl-namespace-inline is deprecated and will be "
          "ignored. Option --use-explicit-namespace is used instead.\n");
    } else {
      if (ExplicitClNamespace) {
        DpctGlobalInfo::setExplicitNamespace(std::vector<ExplicitNamespace>{
            ExplicitNamespace::EN_CL, ExplicitNamespace::EN_DPCT});
      } else {
        DpctGlobalInfo::setExplicitNamespace(DefaultExplicitNamespaces);
      }
      clang::dpct::PrintMsg(
          "Note: Option --no-cl-namespace-inline is deprecated. Use "
          "--use-explicit-namespace instead.\n");
    }
  } else {
    if (UseExplicitNamespace.getNumOccurrences()) {
      DpctGlobalInfo::setExplicitNamespace(UseExplicitNamespace);
    } else {
      DpctGlobalInfo::setExplicitNamespace(DefaultExplicitNamespaces);
    }
  }

  MapNames::setExplicitNamespaceMap();
  CallExprRewriterFactoryBase::initRewriterMap();
  TypeLocRewriterFactoryBase::initTypeLocRewriterMap();
  MemberExprRewriterFactoryBase::initMemberExprRewriterMap();
  clang::dpct::initHeaderSpellings();
  if (!RuleFile.empty()) {
    importRules(RuleFile);
  }

  {
    setValueToOptMap(clang::dpct::OPTION_AsyncHandler, AsyncHandlerFlag,
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
    setValueToOptMap(clang::dpct::OPTION_ExplicitClNamespace,
                     ExplicitClNamespace,
                     NoClNamespaceInline.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_ExtensionDEFlag,
                     DpctGlobalInfo::getExtensionDEFlag(),
                     NoDPCPPExtensions.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_ExtensionDDFlag,
                     DpctGlobalInfo::getExtensionDDFlag(),
                     UseDPCPPExtensions.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_NoDRYPattern, NoDRYPatternFlag,
                     NoDRYPattern.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_CompilationsDir, CompilationsDir,
                     OptParser->isPSpecified());
#ifdef _WIN32
    if (!VcxprojFilePath.empty()) {
      // To convert the relative path to absolute path.
      llvm::SmallString<128> AbsPath(VcxprojFilePath);
      llvm::sys::fs::make_absolute(AbsPath);
      llvm::sys::path::remove_dots(AbsPath, /*remove_dot_dot=*/true);
      setValueToOptMap(clang::dpct::OPTION_VcxprojFile, AbsPath.str().str(),
                       OptParser->isVcxprojfileSpecified());
    } else {
      setValueToOptMap(clang::dpct::OPTION_VcxprojFile, VcxprojFilePath,
                       OptParser->isVcxprojfileSpecified());
    }
#endif
    setValueToOptMap(clang::dpct::OPTION_ProcessAll, ProcessAllFlag,
                     ProcessAll.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_SyclNamedLambda, SyclNamedLambdaFlag,
                     SyclNamedLambda.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_ExperimentalFlag,
                     DpctGlobalInfo::getExperimentalFlag(),
                     Experimentals.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_HelperFuncPreferenceFlag,
                     DpctGlobalInfo::getHelperFuncPreferenceFlag(),
                     Preferences.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_ExplicitNamespace,
                     DpctGlobalInfo::getExplicitNamespaceSet(),
                     UseExplicitNamespace.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_UsmLevel,
                     static_cast<unsigned int>(DpctGlobalInfo::getUsmLevel()),
                     USMLevel.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_OptimizeMigration,
                     OptimizeMigration.getValue(),
                     OptimizeMigration.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_EnablepProfiling,
                     EnablepProfilingFlag, EnablepProfilingFlag);
    setValueToOptMap(clang::dpct::OPTION_RuleFile, MetaRuleObject::RuleFiles,
                     RuleFile.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_AnalysisScopePath,
                     DpctGlobalInfo::getAnalysisScope(),
                     AnalysisScope.getNumOccurrences());

    if (clang::dpct::DpctGlobalInfo::isIncMigration()) {
      std::string Msg;
      if (!canContinueMigration(Msg)) {
        ShowStatus(MigrationErrorDifferentOptSet, Msg);
        return MigrationErrorDifferentOptSet;
      }
    }
  }

  if (ReportType.getValue() == ReportTypeEnum::RTE_All ||
      ReportType.getValue() == ReportTypeEnum::RTE_Stats) {
    // When option "--report-type=stats" or option " --report-type=all" is
    // specified to get the migration status report, dpct namespace should be
    // enabled temporarily to get LOC migrated to helper functions in function
    // getLOCStaticFromCodeRepls() if it is not enabled.
    auto NamespaceSet = DpctGlobalInfo::getExplicitNamespaceSet();
    if (!NamespaceSet.count(ExplicitNamespace::EN_DPCT)) {
      std::vector<ExplicitNamespace> ENVec;
      ENVec.push_back(ExplicitNamespace::EN_DPCT);
      DpctGlobalInfo::setExplicitNamespace(ENVec);
      DpctGlobalInfo::setDPCTNamespaceTempEnabled();
    }
  }

  if (DpctGlobalInfo::getFormatRange() != clang::format::FormatRange::none) {
    parseFormatStyle();
  }

  volatile int RunCount = 0;
  do {
    if (RunCount == 1) {
      // Currently, we just need maximum two parse
      DpctGlobalInfo::setNeedRunAgain(false);
      DpctGlobalInfo::getInstance().resetInfo();
      DeviceFunctionDecl::reset();
    }
    DpctGlobalInfo::setRunRound(RunCount++);
    DpctToolAction Action(OutputFile.empty() &&
                                  !DpctGlobalInfo::isQueryAPIMapping()
                              ? llvm::errs()
                              : DpctTerm(),
                          Tool.getReplacements(), Passes,
                          {PassKind::PK_Analysis, PassKind::PK_Migration},
                          Tool.getFiles().getVirtualFileSystemPtr());

    if (ProcessAllFlag) {
      clang::tooling::SetFileProcessHandle(InRoot, OutRoot, processAllFiles);
    }

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

    if (RunResult && StopOnParseErr) {
      DumpOutputFile();
      if (RunResult == 1) {
        if (DpctGlobalInfo::isQueryAPIMapping()) {
          std::string Err = getDpctTermStr();
          StringRef ErrStr = Err;
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

    Action.runPasses();
  } while (DpctGlobalInfo::isNeedRunAgain());

  if (DpctGlobalInfo::isQueryAPIMapping()) {
    llvm::outs() << "CUDA API:" << llvm::raw_ostream::GREEN
                 << QueryAPIMappingSrc << llvm::raw_ostream::RESET
                 << "Is migrated to" << QueryAPIMappingOpt << ":";
    DiagnosticsEngine Diagnostics(
        IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
        IntrusiveRefCntPtr<DiagnosticOptions>(new DiagnosticOptions()));
    SourceManager Sources(Diagnostics, Tool.getFiles());
    LangOptions DefaultLangOptions;
    Rewriter Rewrite(Sources, DefaultLangOptions);
    // Must be only 1 file.
    tooling::applyAllReplacements(Tool.getReplacements().begin()->second,
                                  Rewrite);
    const auto &RewriteBuffer = Rewrite.buffer_begin()->second;
    static const std::string StartStr{"// Start"};
    static const std::string EndStr{"// End"};
    bool Flag = false;
    llvm::outs() << llvm::raw_ostream::BLUE;
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
        llvm::outs() << I.piece().substr(StartPos, EndPos - StartPos);
      }
    }
    llvm::outs() << llvm::raw_ostream::RESET;
    return MigrationSucceeded;
  }

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
    if (ReportOnlyFlag) {
      DumpOutputFile();
      return MigrationSucceeded;
    }
  }

  // if run was successful
  int Status = saveNewFiles(Tool, InRoot, OutRoot);
  ShowStatus(Status);

  DumpOutputFile();
  return Status;
}

int run(int argc, const char **argv) {
  int Status = runDPCT(argc, argv);
  if (IsUsingDefaultOutRoot) {
    removeDefaultOutRootFolder(OutRoot);
  }
  return Status;
}
