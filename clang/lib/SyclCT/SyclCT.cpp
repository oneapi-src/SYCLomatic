//===--- SyclCT.cpp -------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "clang/SyclCT/SyclCT.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Config.h"
#include "Debug.h"
#include "SaveNewFiles.h"
#include "Utility.h"
#include "ValidateArguments.h"
#include <string>

#include "ToolChains/Cuda.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "SignalProcess.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::syclct;
using namespace clang::tooling;

using namespace llvm::cl;

const char *const CtHelpMessage =
    "\n"
    "<source0> ... Paths of input source files. These paths are\n"
    "\tlooked up in the compilation database. If the path of a source file is\n"
    "\tabsolute, it must exist in the CMake source tree. If the path is\n"
    "\trelative, the current working directory must exist in the CMake\n"
    "\tsource tree and the path must be a subdirectory of the current\n"
    "\tworking directory. \"./\" prefixes in a relative path will be\n"
    "\tautomatically removed.  The remainder of a relative path must be a\n"
    "\tsuffix of a path in the compilation database.\n"
    "\n";

static OptionCategory SyclCTCat("DPC++ Compatibility Tool");
static extrahelp CommonHelp(CtHelpMessage);
static opt<std::string> Passes(
    "passes",
    desc("Comma separated list of migration passes, which will be applied. "
         "Only the specified passes are applied."),
    value_desc("FunctionAttrsRule,..."), cat(SyclCTCat));
static opt<std::string> InRoot(
    "in-root", desc("Directory path for root of source tree to be migrated. "
                    "Only files under this root will be migrated."),
    value_desc("/path/to/input/root/"), cat(SyclCTCat), llvm::cl::Optional);
static opt<std::string>
    OutRoot("out-root", desc("Directory path for root of generated files. "
                             "Directory will be created if it doesn't exist."),
            value_desc("/path/to/output/root/"), cat(SyclCTCat),
            llvm::cl::Optional);

static opt<std::string> ReportType(
    "report-type",
    desc("Comma separated list of report types. "
         "\"apis\": Information "
         "about API signatures that need migration "
         "and the number of times they were encountered. "
         "The report file name will have \".apis\" suffix added. "
         "\"stats\": High level migration statistics;  Lines "
         "Of Code (LOC) migrated to DPC++, LOC migrated to Compatibility API, "
         "LOC not needing migration, LOC needing migration, but not migrated. "
         "The report file name will have \".stats\" suffix added. "
         "\"all\": Generates all of the above reports. "
         "Default is \"stats\"."),
    value_desc("[all|apis|stats]"), cat(SyclCTCat), llvm::cl::Optional);

static opt<std::string>
    ReportFormat("report-format",
                 desc("Format of reports. \"csv\": Output is lines of comma "
                      "separated values. "
                      "Report file name extension will be \".csv\". "
                      "\"formatted\": Output is formatted to be easier to read "
                      "by human eyes. "
                      "Report file name extension will be \".log\". "
                      "Default is \"csv\"."),
                 value_desc("[csv|formatted]"), cat(SyclCTCat),
                 llvm::cl::Optional);

static opt<std::string> ReportFilePrefix(
    "report-file-prefix",
    desc("Prefix for the report file names. The full file name will have a "
         "suffix "
         "derived from the report-type and an extension derived from the "
         "report-format. "
         "For example: <prefix>.apis.csv or <prefix>.stats.log. "
         "If this option is not specified, the report will go "
         "to stdout. The report files are created in the "
         "directory, specified by -out-root.  Default is stdout."),
    value_desc("prefix"), cat(SyclCTCat), llvm::cl::Optional);
bool ReportOnlyFlag = false;
static opt<bool, true> ReportOnly(
    "report-only",
    llvm::cl::desc("Only reports are generated.  No DPC++ code is generated. "
                   "Default is to generate both reports and DPC++ code."),
    cat(SyclCTCat), llvm::cl::location(ReportOnlyFlag));

bool KeepOriginalCodeFlag = false;

static opt<bool, true>
    ShowOrigCode("keep-original-code",
                 llvm::cl::desc("Keep original code in comments of generated "
                                "DPC++ files. Default: off"),
                 cat(SyclCTCat), llvm::cl::location(KeepOriginalCodeFlag));

static opt<std::string>
    DiagsContent("report-diags-content",
                 desc("Diagnostics verbosity level. \"pass\": Basic migration "
                      "pass information. "
                      "\"transformation\": Detailed migration pass "
                      "transformation information."),
                 value_desc("[pass|transformation]"), cat(SyclCTCat),
                 llvm::cl::Optional, llvm::cl::Hidden);

static std::string WarningDesc("Comma separated list of warnings to "
                               " suppress.  Valid warning ids range from " +
                               std::to_string((size_t)Warnings::BEGIN) +
                               " to " +
                               std::to_string((size_t)Warnings::END - 1));
opt<std::string> SuppressWarnings("suppress-warnings", desc(WarningDesc),
                                  value_desc("WarningID,..."), cat(SyclCTCat));

bool SuppressWarningsAllFlag = false;
static std::string WarningAllDesc("Suppress all warnings");
opt<bool, true> SuppressWarningsAll("suppress-warnings-all",
                                    desc(WarningAllDesc), cat(SyclCTCat),
                                    location(SuppressWarningsAllFlag));

bool NoStopOnErrFlag = false;

static opt<bool, true>
    NoStopOnErr("no-stop-on-err",
                llvm::cl::desc("Continue migration and report generation after "
                               "possible errors. Default: off"),
                cat(SyclCTCat), llvm::cl::location(NoStopOnErrFlag));

opt<OutputVerbosityLev> OutputVerbosity(
    "output-verbosity", llvm::cl::desc("Set the output verbosity level:"),
    llvm::cl::values(
        clEnumVal(silent, "Only messages from clang"),
        clEnumVal(normal,
                  "Only warnings, errors, notes from both clang and syclct"),
        clEnumVal(detailed,
                  "Normal + messages about start and end of file parsing"),
        clEnumVal(diagnostics, "Everything, as now - which includes "
                               "information about conflicts, seg faults, "
                               "etc.... This one is default.")),
    llvm::cl::init(diagnostics), cat(SyclCTCat), llvm::cl::Optional);

opt<std::string> OutputFile(
    "output-file", desc("redirects stdout/stderr output to <file> in the "
                        "output diretory specified by '-out-root' option."),
    value_desc("output file name"), cat(SyclCTCat), llvm::cl::Optional);

// Currently, set IsPrintOnNormal false only at the place where messages about
// start and end of file parsing are produced,
//.i.e in the place "lib/Tooling:int ClangTool::run(ToolAction *Action)".
void PrintMsg(const std::string &Msg, bool IsPrintOnNormal = true) {
  if (!OutputFile.empty()) {
    //  Redirects stdout/stderr output to <file>
    SyclctTerm() << Msg;
  }

  switch (OutputVerbosity) {
  case detailed:
  case diagnostics:
    llvm::outs() << Msg;
    break;
  case normal:
    if (IsPrintOnNormal) {
      llvm::outs() << Msg;
    }
    break;
  case silent:
  default:
    break;
  }
}

std::string CudaPath;          // Global value for the CUDA install path.
std::string SyclctInstallPath; // Installation directory for this tool

class SyclCTConsumer : public ASTConsumer {
public:
  SyclCTConsumer(ReplTy &R, const CompilerInstance &CI, StringRef InFile)
      : ATM(CI, InRoot), Repl(R), PP(CI.getPreprocessor()) {
    int RequiredRType = 0;
    SourceProcessType FileType = GetSourceFileType(InFile);

    if (FileType & (TypeCudaSource | TypeCudaHeader)) {
      RequiredRType = ApplyToCudaFile;
    } else if (FileType & (TypeCppSource | TypeCppHeader)) {
      RequiredRType = ApplyToCppFile;
    }

    if (Passes != "") {
      // Separate string into list by comma
      auto Names = split(Passes, ',');

      std::vector<std::vector<std::string>> Rules;

      for (auto const &Name : Names) {
        auto *ID = ASTTraversalMetaInfo::getID(Name);
        auto MapEntry = ASTTraversalMetaInfo::getConstructorTable()[ID];
        auto RuleObj = (TranslationRule *)MapEntry();
        CommonRuleProperty RuleProperty = RuleObj->GetRuleProperty();
        auto RType = RuleProperty.RType;
        auto RulesDependon = RuleProperty.RulesDependon;

        // Add rules should be run on the source file
        if (RType & RequiredRType) {
          std::vector<std::string> Vec;
          Vec.push_back(Name);
          for (auto const &RuleName : RulesDependon) {
            Vec.push_back(RuleName);
          }
          Rules.push_back(Vec);
        }
      }

      std::vector<std::string> SortedRules = ruleTopoSort(Rules);
      for (std::vector<std::string>::reverse_iterator it = SortedRules.rbegin();
           it != SortedRules.rend(); it++) {
        auto *RuleID = ASTTraversalMetaInfo::getID(*it);
        if (!RuleID) {
          const std::string ErrMsg = "[ERROR] Rule\"" + *it + "\" not found\n";
          PrintMsg(ErrMsg);
          llvm_unreachable(ErrMsg.c_str());
        }
        ATM.emplaceTranslationRule(RuleID);
      }

    } else {
      ATM.emplaceAllRules(RequiredRType);
    }
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // The migration process is separated into two stages:
    // 1) Analysis of AST and identification of applicable migration rules
    // 2) Generation of actual textual Replacements
    // Such separation makes it possible to post-process the list of identified
    // migration rules before applying them.
    ATM.matchAST(Context, TransformSet, SSM);

    auto &Global = SyclctGlobalInfo::getInstance();
    for (const auto &I : TransformSet) {
      auto Repl = I->getReplacement(Context);
      Global.addReplacement(Repl);

      // TODO: Need to print debug info here
    }
  }

  void Initialize(ASTContext &Context) override {
    // Set Context for build information
    SyclctGlobalInfo::setContext(Context);

    PP.addPPCallbacks(llvm::make_unique<IncludesCallbacks>(
        TransformSet, Context.getSourceManager(), ATM));
  }

  ~SyclCTConsumer() {
    // Clean EmittedTransformations for input file migrated.
    ASTTraversalMetaInfo::getEmittedTransformations().clear();
  }

private:
  ASTTraversalManager ATM;
  TransformSetTy TransformSet;
  StmtStringMap SSM;
  ReplTy &Repl;
  Preprocessor &PP;
};

class SyclCTAction : public ASTFrontendAction {
  ReplTy &Repl;

public:
  SyclCTAction(ReplTy &R) : Repl(R) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return llvm::make_unique<SyclCTConsumer>(Repl, CI, InFile);
  }

  bool usesPreprocessorOnly() const override { return false; }
};

// Object of this class will be handed to RefactoringTool::run and will create
// the Action.
class SyclCTActionFactory : public FrontendActionFactory {
  ReplTy &Repl;

public:
  SyclCTActionFactory(ReplTy &R) : Repl(R) {}
  FrontendAction *create() override { return new SyclCTAction{Repl}; }
};

std::string getCudaInstallPath(int argc, const char **argv) {
  std::vector<const char *> Argv;
  Argv.reserve(argc);
  // do not copy "--" so the driver sees a possible --cuda-path option
  std::copy_if(argv, argv + argc, back_inserter(Argv),
               [](const char *s) { return std::strcmp(s, "--"); });

  // Output parameters to indicate errors in parsing. Not checked here,
  // OptParser will handle errors.
  unsigned MissingArgIndex, MissingArgCount;
  std::unique_ptr<llvm::opt::OptTable> Opts = driver::createDriverOptTable();
  llvm::opt::InputArgList ParsedArgs =
      Opts->ParseArgs(Argv, MissingArgIndex, MissingArgCount);

  // Create minimalist CudaInstallationDetector and return the InstallPath.
  DiagnosticsEngine E(nullptr, nullptr, nullptr, false);
  driver::Driver Driver("", llvm::sys::getDefaultTargetTriple(), E, nullptr);
  driver::CudaInstallationDetector CudaDetector(
      Driver, llvm::Triple(Driver.getTargetTriple()), ParsedArgs);

  std::string Path = CudaDetector.getInstallPath();
  makeCanonical(Path);
  return Path;
}

std::string getInstallPath(clang::tooling::ClangTool &Tool,
                           const char *invokeCommand) {
  SmallString<512> InstalledPath(invokeCommand);

  // Do a PATH lookup, if there are no directory components.
  if (llvm::sys::path::filename(InstalledPath) == InstalledPath) {
    if (llvm::ErrorOr<std::string> Tmp = llvm::sys::findProgramByName(
            llvm::sys::path::filename(InstalledPath.str()))) {
      InstalledPath = *Tmp;
    }
  }

  makeCanonical(InstalledPath);
  StringRef InstalledPathParent(llvm::sys::path::parent_path(InstalledPath));
  // Move up to parent directory of bin directory
  StringRef InstallPath = llvm::sys::path::parent_path(InstalledPathParent);
  return InstallPath.str();
}

// E.g. Path is "/usr/local/cuda/samples" and "cuda" is a symlink of cuda-8.0
// GetRealPath will return /usr/local/cuda-8.0/samples
std::string GetRealPath(clang::tooling::RefactoringTool &Tool, StringRef Path) {
  // Set up Rewriter and to get source manager.
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);
  SourceManager Sources(Diagnostics, Tool.getFiles());
  Rewriter Rewrite(Sources, DefaultLangOptions);
  const SourceManager &SM = Rewrite.getSourceMgr();

  llvm::SmallString<512> AbsolutePath(Path);
  // Try to get the real file path of the symlink.
  const DirectoryEntry *Dir = SM.getFileManager().getDirectory(
      llvm::sys::path::parent_path(AbsolutePath.str()));
  StringRef DirName = SM.getFileManager().getCanonicalName(Dir);
  SmallVector<char, 512> AbsoluteFilename;
  llvm::sys::path::append(AbsoluteFilename, DirName,
                          llvm::sys::path::filename(AbsolutePath.str()));
  return llvm::StringRef(AbsoluteFilename.data(), AbsoluteFilename.size())
      .str();
}

// To validate the root path of the project to be migrated.
void ValidateInputDirectory(clang::tooling::RefactoringTool &Tool,
                            std::string &InRoot) {
  std::string Path = GetRealPath(Tool, InRoot);

  if (isChildPath(CudaPath, Path)) {
    std::string ErrMsg =
        "[ERROR] Input root specified by \"-in-root\" option \"" + Path +
        "\" is in CUDA_PATH folder \"" + CudaPath + "\"\n";
    PrintMsg(ErrMsg);
    llvm_unreachable(ErrMsg.c_str());
  }

  if (isChildPath(Path, SyclctInstallPath) ||
      isSamePath(Path, SyclctInstallPath)) {
    std::string ErrMsg = "[ERROR] Input folder \"" + Path +
                         "\" is the parent or the same as the folder where "
                         "DPC++ Compatibility Tool is installed \"" +
                         SyclctInstallPath + "\"\n";
    PrintMsg(ErrMsg);
    llvm_unreachable(ErrMsg.c_str());
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

  const FileEntry *Entry = SM.getFileManager().getFile(Path);
  if (!Entry) {
    std::string ErrMsg = "FilePath Invalide...\n";
    PrintMsg(ErrMsg);
    llvm_unreachable(ErrMsg.c_str());
  }

  FileID FID = SM.getOrCreateFileID(Entry, SrcMgr::C_User);

  SourceLocation EndOfFile = SM.getLocForEndOfFile(FID);
  unsigned int LineNumber = SM.getSpellingLineNumber(EndOfFile, nullptr);
  return LineNumber;
}

static void printMetrics(clang::tooling::RefactoringTool &Tool) {

  for (const auto &Elem : LOCStaticsMap) {
    unsigned TotalLines = GetLinesNumber(Tool, Elem.first);
    unsigned TransToAPI = Elem.second[0];
    unsigned TransToSYCL = Elem.second[1];
    unsigned NotTrans = TotalLines - TransToSYCL - TransToAPI;
    unsigned NotSupport = Elem.second[2];

    SyclctStats() << "\n";
    SyclctStats()
        << "File name, LOC migrated to SYCL, LOC migrated to Compatibility "
           "API, LOC not needed to migrate, LOC not able to migrate";
    SyclctStats() << "\n";
    SyclctStats() << Elem.first + ", " + std::to_string(TransToSYCL) + ", " +
                         std::to_string(TransToAPI) + ", " +
                         std::to_string(NotTrans) + ", " +
                         std::to_string(NotSupport);
    SyclctStats() << "\n";
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
    std::string RFile = OutRoot + "/" + ReportFilePrefix +
                        (ReportFormat == "csv" ? ".apis.csv" : ".apis.log");
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(RFile, std::ios::binary);

    std::string Str;
    llvm::raw_string_ostream Title(Str);
    Title << (ReportFormat == "csv" ? "API name, Frequency"
                                    : "API name\t\t\t\tFrequency");

    File << Title.str() << std::endl;
    for (const auto &Elem : SrcAPIStaticsMap) {
      std::string APIName = Elem.first;
      unsigned int Count = Elem.second;
      if (ReportFormat == "csv") {
        File << APIName << "," << std::to_string(Count) << std::endl;
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
  SyclctStats() << "\nTotal migration time: " + std::to_string(Duration) +
                       " ms\n";
  if (ReportFilePrefix == "stdout") {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "----------Stats report---------------\n";
    OS << getSyclctStatsStr() << "\n";
    OS << "-------------------------------------\n";
    PrintMsg(OS.str());
  } else {
    std::string RFile = OutRoot + "/" + ReportFilePrefix +
                        (ReportFormat == "csv" ? ".stats.csv" : ".stats.log");
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(RFile, std::ios::binary);
    File << getSyclctStatsStr() << "\n";
  }
}

static void saveDiagsReport() {

  // SyclctDiags() << "\n";
  if (ReportFilePrefix == "stdout") {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "--------Diags message----------------\n";
    OS << getSyclctDiagsStr() << "\n";
    OS << "-------------------------------------\n";
    PrintMsg(OS.str());
  } else {
    std::string RFile = OutRoot + "/" + ReportFilePrefix + ".diags.log";
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(RFile, std::ios::binary);
    File << getSyclctDiagsStr() << "\n";
  }
}

std::string printCTVersion() {

  std::string buf;
  llvm::raw_string_ostream OS(buf);

  OS << "\nCompatibility Tool Version: " << SYCLCT_VERSION_MAJOR << "."
     << SYCLCT_VERSION_MINOR << "." << SYCLCT_VERSION_PATCH << " codebase:";

  std::string Path = getClangRepositoryPath();
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
    File << getSyclctTermStr() << "\n";
  }
}

void PrintReportOnFault(std::string &FaultMsg) {
  PrintMsg(FaultMsg);
  saveApisReport();
  saveDiagsReport();

  std::string FileApis = OutRoot + "/" + ReportFilePrefix +
                         (ReportFormat == "csv" ? ".apis.csv" : ".apis.log");
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

int run(int argc, const char **argv) {

#if defined(__linux__) || defined(_WIN64)
  InstallSignalHandle();
#endif
  // Set hangle for libclangTooling to proccess message for syclct
  clang::tooling::SetPrintHandler(PrintMsg);

  // CommonOptionsParser will adjust argc to the index of "--"
  int OriginalArgc = argc;
  llvm::cl::SetVersionPrinter(
      [](llvm::raw_ostream &OS) { OS << printCTVersion() << "\n"; });
  CommonOptionsParser OptParser(argc, argv, SyclCTCat);
  clock_t StartTime = clock();
  if (!makeCanonicalOrSetDefaults(InRoot, OutRoot,
                                  OptParser.getSourcePathList()))
    exit(-1);

  if (!validatePaths(InRoot, OptParser.getSourcePathList()))
    exit(-1);

  bool GenReport = false;
  if (checkReportArgs(ReportType, ReportFormat, ReportFilePrefix,
                      ReportOnlyFlag, GenReport, DiagsContent) == false)
    exit(-1);

  if (GenReport) {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "Generate report: "
       << "report-type:" << ReportType << ", report-format:" << ReportFormat
       << ", report-file-prefix:" << ReportFilePrefix << "\n";

    PrintMsg(OS.str());
  }

  CudaPath = getCudaInstallPath(OriginalArgc, argv);
  SYCLCT_DEBUG_WITH_TYPE(
      "CudaPath", SyclctLog() << "Cuda Path found: " << CudaPath << "\n");

  RefactoringTool Tool(OptParser.getCompilations(),
                       OptParser.getSourcePathList());
  SyclctInstallPath = getInstallPath(Tool, argv[0]);

  ValidateInputDirectory(Tool, InRoot);
  // Made "-- -x cuda --cuda-host-only" option set by default, .i.e
  // commandline "syclct -in-root ./ -out-root ./ ./topologyQuery.cu  --  -x
  // cuda
  // --cuda-host-only  -I../common/inc" became "syclct -in-root ./ -out-root
  // ./
  // ./topologyQuery.cu  -- -I../common/inc"
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "--cuda-host-only", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("cuda", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-x", ArgumentInsertPosition::BEGIN));

  SyclctGlobalInfo::setInRoot(InRoot);
  SyclctGlobalInfo::setKeepOriginCode(KeepOriginalCodeFlag);
  SyclCTActionFactory Factory(Tool.getReplacements());
  if (int RunResult = Tool.run(&Factory) && !NoStopOnErrFlag) {
    DebugInfo::ShowStatus(RunResult);
    DumpOutputFile();
    return RunResult;
  }

  auto &Global = SyclctGlobalInfo::getInstance();
  Global.buildReplacements();
  Global.emplaceReplacements(Tool.getReplacements());

  if (GenReport) {
    // report: apis, stats, diags
    if (ReportType.find("all") != std::string::npos ||
        ReportType.find("apis") != std::string::npos)
      saveApisReport();

    if (ReportType.find("all") != std::string::npos ||
        ReportType.find("stats") != std::string::npos) {
      clock_t EndTime = clock();
      double Duration = (double)(EndTime - StartTime) / (CLOCKS_PER_SEC / 1000);
      saveStatsReport(Tool, Duration);
    }
    // all doesn't include diags.
    if (ReportType.find("diags") != std::string::npos) {
      saveDiagsReport();
    }
    if (ReportOnlyFlag) {
      DumpOutputFile();
      return MigrationSucceeded;
    }
  }
  // if run was successful
  int Status = saveNewFiles(Tool, InRoot, OutRoot);
  DebugInfo::ShowStatus(Status);

  DumpOutputFile();
  return Status;
}
