//===--- DPCT.cpp -------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "clang/DPCT/DPCT.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Config.h"
#include "Debug.h"
#include "GAnalytics.h"
#include "SaveNewFiles.h"
#include "SignalProcess.h"
#include "Utility.h"
#include "ValidateArguments.h"
#include "VcxprojParser.h"
#include "Checkpoint.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Host.h"

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
} // namespace tooling
namespace dpct {
extern llvm::cl::OptionCategory DPCTCat;
void initWarningIDs();
#if defined(_WIN32)
#define MAX_PATH_LEN _MAX_PATH
#define MAX_NAME_LEN _MAX_FNAME
#else
#define MAX_PATH_LEN PATH_MAX
#define MAX_NAME_LEN NAME_MAX
#endif
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
    "  dpct -p=<path to location of compilation database file>\n\n"
    "Migrate one file in compilation database:\n\n"
    "  dpct -p=<path to location of compilation database file>  source.cpp\n\n"
#if defined(_WIN32)
    "Migrate all files available in vcxprojfile:\n\n"
    "  dpct --vcxprojfile=path/to/vcxprojfile.vcxproj\n"
#endif
    DiagRef
    ;

const char *const CtHelpHint =
    "  Warning: Please specify file(s) to be migrated.\n"
    "  Get help on Intel(R) DPC++ Compatibility Tool, run: dpct --help\n"
    "\n";

static extrahelp CommonHelp(CtHelpMessage);
static opt<std::string> Passes(
    "passes",
    desc("Comma separated list of migration passes, which will be applied.\n"
         "Only the specified passes are applied."),
    value_desc("IterationSpaceBuiltinRule,..."), cat(DPCTCat),
               llvm::cl::Hidden);
static opt<std::string>
    InRoot("in-root",
           desc("The directory path for the root of the source tree that needs "
                "to be migrated.\n"
                "Only files under this root are migrated. Default: Current"
                " directory, if input\nsource files are not provided. "
                "The directory of the first input source file, if\ninput"
                " source files are provided."),
           value_desc("dir"), cat(DPCTCat),
           llvm::cl::Optional);
static opt<std::string> OutRoot(
    "out-root",
    desc("The directory path for root of generated files. A directory is "
         "created if it\n"
         "does not exist. Default: dpct_output."),
    value_desc("dir"), cat(DPCTCat), llvm::cl::Optional);

static opt<std::string> SDKPath("cuda-path", desc("Directory path of SDK.\n"),
                                value_desc("dir"), cat(DPCTCat),
                                llvm::cl::Optional, llvm::cl::Hidden);

static opt<std::string>
    SDKIncludePath("cuda-include-path",
                   desc("The directory path of the CUDA header files."),
                   value_desc("dir"), cat(DPCTCat), llvm::cl::Optional);

static opt<ReportTypeEnum> ReportType(
    "report-type", desc("Comma separated list of report types."),
    llvm::cl::values(
        llvm::cl::OptionEnumValue{"apis", int(ReportTypeEnum::apis),
            "Information about API signatures that need migration and the "
            "number of times\n"
            "                                    they were encountered. The "
            "report file name will have .apis suffix added.", false},
        llvm::cl::OptionEnumValue{"stats", int(ReportTypeEnum::stats),
                  "High level migration statistics: Lines Of Code (LOC) that "
                  "are migrated to\n"
                  "                                    DPC++, LOC migrated to "
                  "DPC++ with helper functions, LOC not needing migration,\n"
                  "                                    LOC needing migration "
                  "suffix added. (default)", false},
        llvm::cl::OptionEnumValue{"all", int(ReportTypeEnum::all),
                  "All of the reports.", false}
        #ifdef DPCT_DEBUG_BUILD
        , llvm::cl::OptionEnumValue{"diags", int(ReportTypeEnum::diags),
                  "diags information", true}
        #endif
        ),
    llvm::cl::init(ReportTypeEnum::notsettype), value_desc("value"), cat(DPCTCat),
    llvm::cl::Optional);

static opt<ReportFormatEnum> ReportFormat(
    "report-format", desc("Format of the reports:\n"),
    llvm::cl::values(
        llvm::cl::OptionEnumValue{"csv", int(ReportFormatEnum::csv),
                  "Output is lines of comma separated values. The report file "
                  "name extension will\n"
                  "                                    be .csv. (default)", false},
        llvm::cl::OptionEnumValue{"formatted", int(ReportFormatEnum::formatted),
                  "Output is formatted to be easier to read for "
                  "human eyes. Report file name\n"
                  "                                    extension will be log.",
                  false}),
    llvm::cl::init(ReportFormatEnum::notsetformat), value_desc("value"), cat(DPCTCat),
    llvm::cl::Optional);

static opt<std::string> ReportFilePrefix(
    "report-file-prefix",
    desc(
        "Prefix for the report file names. The full file name will have a "
        "suffix derived\n"
        "from the report-type and an extension derived from the report-format. "
        "For\n"
        "example: <prefix>.apis.csv or <prefix>.stats.log. If this option is "
        "not\n"
        "specified, the report will go to stdout. The report files are created "
        "in the\n"
        "directory, specified by -out-root."),
    value_desc("prefix"), cat(DPCTCat), llvm::cl::Optional);
bool ReportOnlyFlag = false;
static opt<bool, true>
    ReportOnly("report-only",
               llvm::cl::desc("Only reports are generated. No DPC++ code is "
                              "generated. Default: off."),
               cat(DPCTCat), llvm::cl::location(ReportOnlyFlag));

bool KeepOriginalCodeFlag = false;

static opt<bool, true>
    ShowOrigCode("keep-original-code",
                 llvm::cl::desc("Keeps the original code in comments of "
                                "generated DPC++ files. Default: off.\n"),
                 cat(DPCTCat), llvm::cl::location(KeepOriginalCodeFlag));
#ifdef DPCT_DEBUG_BUILD
static opt<std::string>
    DiagsContent("report-diags-content",
                 desc("Diagnostics verbosity level. \"pass\": Basic migration "
                      "pass information. "
                      "\"transformation\": Detailed migration pass "
                      "transformation information."),
                 value_desc("[pass|transformation]"), cat(DPCTCat),
                 llvm::cl::Optional, llvm::cl::Hidden);
#endif
static std::string
    WarningDesc("Comma separated list of migration warnings to suppress. Valid "
                "warning IDs range\n"
                "from " + std::to_string((size_t)Warnings::BEGIN) + " to " +
                std::to_string((size_t)Warnings::END - 1) +
                ". Hyphen separated ranges are also allowed. For example:\n"
                "--suppress-warnings=1000-1010,1011.");
opt<std::string> SuppressWarnings("suppress-warnings", desc(WarningDesc),
                                  value_desc("value"), cat(DPCTCat));

bool SuppressWarningsAllFlag = false;
static std::string WarningAllDesc("Suppresses all migration warnings. Default: off.");
opt<bool, true> SuppressWarningsAll("suppress-warnings-all",
                                    desc(WarningAllDesc), cat(DPCTCat),
                                    location(SuppressWarningsAllFlag));
bool StopOnParseErr = false;
static opt<bool, true>
    StopOnParseErrOption("stop-on-parse-err",
                llvm::cl::desc("Stop migration and generation of reports if "
                               "parsing errors happened. Default: off. \n"),
                cat(DPCTCat), llvm::cl::location(StopOnParseErr));


bool SyclNamedLambdaFlag = false;
static opt<bool, true>
    SyclNamedLambda("sycl-named-lambda",
                llvm::cl::desc("Generates kernels with the kernel name. Default: off.\n"),
                cat(DPCTCat), llvm::cl::location(SyclNamedLambdaFlag));

opt<OutputVerbosityLev> OutputVerbosity(
    "output-verbosity", llvm::cl::desc("Sets the output verbosity level:"),
    llvm::cl::values(
        llvm::cl::OptionEnumValue{"silent", int(OutputVerbosityLev::silent),
                                  "Only messages from clang.", false},
        llvm::cl::OptionEnumValue{"normal", int(OutputVerbosityLev::normal),
                                  "\'silent\' and warnings, errors, and notes from dpct.",
                                  false},
        llvm::cl::OptionEnumValue{"detailed", int(OutputVerbosityLev::detailed),
                                  "\'normal\' and messages about which file is being processed.",
                                  false},
        llvm::cl::OptionEnumValue{"diagnostics", int(OutputVerbosityLev::diagnostics),
                                  "\'detailed\' and information about the detected "
                                  "conflicts and crashes. (default)", false}),
    llvm::cl::init(OutputVerbosityLev::diagnostics), value_desc("value"), cat(DPCTCat),
    llvm::cl::Optional);

opt<std::string>
    OutputFile("output-file",
               desc("Redirects the stdout/stderr output to <file> in the output"
                    " directory specified\n"
                    "by the --out-root option."),
               value_desc("file"), cat(DPCTCat),
               llvm::cl::Optional);

opt<UsmLevel> USMLevel(
    "usm-level", desc("Sets the USM level to use in source code generation.\n"),
    values(llvm::cl::OptionEnumValue{"restricted", int(UsmLevel::restricted),
                     "Uses API from DPC++ Explicit and Restricted Unified "
                     "Shared Memory extension\n"
                     "                                    for memory management"
                     " migration. (default)", false},
           llvm::cl::OptionEnumValue{"none", int(UsmLevel::none),
                     "Uses helper functions from DPCT header files for memory "
                     "management migration.", false}),
    init(UsmLevel::restricted), value_desc("value"), cat(DPCTCat), llvm::cl::Optional);

opt<format::FormatRange>
    FormatRng("format-range",
                llvm::cl::desc("Sets the range of formatting.\nThe values are:\n"),
                values(llvm::cl::OptionEnumValue{"migrated", int(format::FormatRange::migrated),
                     "Only formats the migrated code (default).", false},
                       llvm::cl::OptionEnumValue{"all", int(format::FormatRange::all),
                     "Formats all code.", false},
                       llvm::cl::OptionEnumValue{"none", int(format::FormatRange::none),
                     "Do not format any code.", false}),
    init(format::FormatRange::migrated), value_desc("value"), cat(DPCTCat), llvm::cl::Optional);

opt<DPCTFormatStyle>
    FormatST("format-style",
                llvm::cl::desc("Sets the formatting style.\nThe values are:\n"),
                values(llvm::cl::OptionEnumValue{"llvm", int(DPCTFormatStyle::llvm),
                     "Use the LLVM coding style.", false},
                       llvm::cl::OptionEnumValue{"google", int(DPCTFormatStyle::google),
                     "Use the Google coding style.", false},
                       llvm::cl::OptionEnumValue{"custom", int(DPCTFormatStyle::custom),
                     "Use the coding style defined in the .clang-format file (default).", false}),
    init(DPCTFormatStyle::custom), value_desc("value"), cat(DPCTCat), llvm::cl::Optional);

bool ExplicitClNamespace = false;
static opt<bool, true> NoClNamespaceInline(
  "no-cl-namespace-inline", llvm::cl::desc("Do not use cl namespace (cl::) inlining. Default: off.\n"),
  cat(DPCTCat), llvm::cl::location(ExplicitClNamespace));

bool NoDRYPatternFlag = false;
static opt<bool, true> NoDRYPattern(
  "no-dry-pattern", llvm::cl::desc("Do not use DRY (do not repeat yourself) pattern when functions from dpct\n"
                                   "namespace are inserted. Default: off.\n"),
  cat(DPCTCat), llvm::cl::location(NoDRYPatternFlag));

bool ProcessAllFlag = false;
static opt<bool, true>
    ProcessAll("process-all",
                 llvm::cl::desc("Migrates/copies all files from the --in-root directory"
                                " to the --out-root directory.\n"
                                "--in-root option should be explicitly specified. Default: off."),
                 cat(DPCTCat), llvm::cl::location(ProcessAllFlag));

static opt<bool> EnableCTAD(
    "enable-ctad",
    llvm::cl::desc("Use a C++17 class template argument deduction (CTAD) in "
                   "your generated code.\n"
                   "Default: off."),
    cat(DPCTCat), init(false));

static opt<bool> EnableComments(
    "comments", llvm::cl::desc("Insert comments explaining the generated code. Default: off."),
    cat(DPCTCat), init(false));

bool AsyncHandlerFlag = false;
static opt<bool, true>
    AsyncHandler("always-use-async-handler",
                 llvm::cl::desc("Always create the cl::sycl::queue with an async "
                                "exception handler. Default: off."),
                 cat(DPCTCat), llvm::cl::location(AsyncHandlerFlag));
// clang-format on

// TODO: implement one of this for each source language.
std::string CudaPath;
std::string DpctInstallPath;
std::unordered_map<std::string, bool> ChildOrSameCache;
std::unordered_map<std::string, bool> ChildPathCache;
std::unordered_map<std::string, llvm::SmallString<256>> RealPathCache;
int FatalErrorCnt=0;
extern bool StopOnParseErrTooling;
extern std::string InRootTooling;
JMP_BUF CPFileASTMaterEnter;
JMP_BUF CPRepPostprocessEnter;

class DPCTConsumer : public ASTConsumer {
public:
  DPCTConsumer(ReplTy &R, CompilerInstance &CI, StringRef InFile)
      : ATM(CI, InRoot), Repl(R), PP(CI.getPreprocessor()), CI(CI) {
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
        auto RuleObj = (MigrationRule *)MapEntry();
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
        }
        ATM.emplaceMigrationRule(RuleID);
      }

    } else {
      ATM.emplaceAllRules(RequiredRType);
    }
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    if(StopOnParseErr && Context.getDiagnostics().getClient()&&
        Context.getDiagnostics().getClient()->getNumErrors()>0){
        return;
    }
    // The migration process is separated into two stages:
    // 1) Analysis of AST and identification of applicable migration rules
    // 2) Generation of actual textual Replacements
    // Such separation makes it possible to post-process the list of identified
    // migration rules before applying them.
    ATM.matchAST(Context, TransformSet, SSM);

    auto &Global = DpctGlobalInfo::getInstance();
    std::unordered_set<std::string> DuplicateFilter;
    for (const auto &I : TransformSet) {
      auto Repl = I->getReplacement(Context);
      // For file path got in AST may be different with the one in preprocessing stage,
      // here only the file name is used to retrieve IncludeMapSet.
      const std::string FileName = llvm::sys::path::filename(Repl->getFilePath()).str();
      if(DuplicateFilter.find(FileName) == end(DuplicateFilter)) {
        DuplicateFilter.insert(FileName);
        auto Find = IncludeMapSet.find(FileName);
        if (Find != IncludeMapSet.end()) {
          for (const auto &Entry : Find->second) {
            Global.addReplacement(Entry->getReplacement(Context));
          }
        }
      }
      Global.addReplacement(Repl);
    }

    DebugInfo::printReplacements(TransformSet, Context);
  }

  void Initialize(ASTContext &Context) override {
    // Set Context for build information
    DpctGlobalInfo::setCompilerInstance(CI);

    PP.addPPCallbacks(std::make_unique<IncludesCallbacks>(
        TransformSet, IncludeMapSet, Context.getSourceManager(), ATM));
  }

  ~DPCTConsumer() {
    // Clean EmittedTransformations for input file migrated.
    ASTTraversalMetaInfo::getEmittedTransformations().clear();
  }

private:
  ASTTraversalManager ATM;
  TransformSetTy TransformSet;
  IncludeMapSetTy IncludeMapSet;
  StmtStringMap SSM;
  ReplTy &Repl;
  Preprocessor &PP;
  CompilerInstance &CI;
};

class DPCTAction : public ASTFrontendAction {
  ReplTy &Repl;

public:
  DPCTAction(ReplTy &R) : Repl(R) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<DPCTConsumer>(Repl, CI, InFile);
  }

  bool usesPreprocessorOnly() const override { return false; }
};

// Object of this class will be handed to RefactoringTool::run and will create
// the Action.
class DPCTActionFactory : public FrontendActionFactory {
  ReplTy &Repl;

public:
  DPCTActionFactory(ReplTy &R) : Repl(R) {}
  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<DPCTAction>(Repl);
  }
};

std::string getCudaInstallPath(int argc, const char **argv) {
  std::vector<const char *> Argv;
  Argv.reserve(argc);
  // do not copy "--" so the driver sees a possible sdk include path option
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
  driver::Driver Driver("", llvm::sys::getDefaultTargetTriple(), E, nullptr);
  driver::CudaInstallationDetector SDKDetector(
      Driver, llvm::Triple(Driver.getTargetTriple()), ParsedArgs);

  std::string Path = SDKDetector.getInstallPath().str();
  if (!SDKDetector.isValid()) {
    DebugInfo::ShowStatus(MigrationErrorInvalidSDKPath);
    exit(MigrationErrorInvalidSDKPath);
  }

  makeCanonical(Path);

  SmallString<512> CudaPathAbs;
  std::error_code EC = llvm::sys::fs::real_path(Path, CudaPathAbs);
  if ((bool)EC) {
    DebugInfo::ShowStatus(MigrationErrorInvalidSDKPath);
    exit(MigrationErrorInvalidSDKPath);
  }
  return CudaPathAbs.str().str();
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

  SmallString<512> InstallPathAbs;
  std::error_code EC = llvm::sys::fs::real_path(InstallPath, InstallPathAbs);
  if ((bool)EC) {
    DebugInfo::ShowStatus(MigrationErrorInvalidInstallPath);
    exit(MigrationErrorInvalidInstallPath);
  }
  return InstallPathAbs.str().str();
}

// To validate the root path of the project to be migrated.
void ValidateInputDirectory(clang::tooling::RefactoringTool &Tool,
                            std::string &InRoot) {

  if (isChildOrSamePath(CudaPath, InRoot)) {
    std::string ErrMsg =
        "[ERROR] Input root specified by \"-in-root\" option \"" + InRoot +
        "\" is in CUDA_PATH folder \"" + CudaPath + "\"\n";
    PrintMsg(ErrMsg);
    exit(MigrationErrorRunFromSDKFolder);
  }

  if (isChildOrSamePath(InRoot, DpctInstallPath)) {
    std::string ErrMsg = "[ERROR] Input folder \"" + InRoot +
                         "\" is the parent or the same as the folder where "
                         "Intel(R) DPC++ Compatibility Tool is installed \"" +
                         DpctInstallPath + "\"\n";
    PrintMsg(ErrMsg);
    exit(MigrationErrorInRootContainCTTool);
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
    exit(MigrationErrorInvalidFilePath);
  }

  FileID FID = SM.getOrCreateFileID(Entry, SrcMgr::C_User);

  SourceLocation EndOfFile = SM.getLocForEndOfFile(FID);
  unsigned int LineNumber = SM.getSpellingLineNumber(EndOfFile, nullptr);
  return LineNumber;
}

static void printMetrics(clang::tooling::RefactoringTool &Tool) {

  size_t Count = 0;
  for (const auto &Elem : LOCStaticsMap) {
    unsigned TotalLines = GetLinesNumber(Tool, Elem.first);
    unsigned TransToAPI = Elem.second[0];
    unsigned TransToSYCL = Elem.second[1];
    unsigned NotTrans = TotalLines - TransToSYCL - TransToAPI;
    unsigned NotSupport = Elem.second[2];
    if (Count == 0) {
      DpctStats() << "\n";
      DpctStats() << "File name, LOC migrated to DPC++, LOC migrated to helper "
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
        (ReportFormat.getValue() == ReportFormatEnum::csv ? ".apis.csv"
                                                          : ".apis.log");
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(RFile, std::ios::binary);

    std::string Str;
    llvm::raw_string_ostream Title(Str);
    Title << (ReportFormat.getValue() == ReportFormatEnum::csv
                  ? " API name, Frequency "
                  : "API name\t\t\t\tFrequency");

    File << Title.str() << std::endl;
    for (const auto &Elem : SrcAPIStaticsMap) {
      std::string APIName = Elem.first;
      unsigned int Count = Elem.second;
      if (ReportFormat.getValue() == ReportFormatEnum::csv) {
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
        (ReportFormat.getValue() == ReportFormatEnum::csv ? ".stats.csv"
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

  OS << "\nIntel(R) DPC++ Compatibility Tool Version: " << DPCT_VERSION_MAJOR
     << "." << DPCT_VERSION_MINOR << "-" << DPCT_VERSION_PATCH << " codebase:";
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

void PrintReportOnFault(std::string &FaultMsg) {
  PrintMsg(FaultMsg);
  saveApisReport();
  saveDiagsReport();

  std::string FileApis =
      OutRoot + "/" + ReportFilePrefix +
      (ReportFormat.getValue() == ReportFormatEnum::csv ? ".apis.csv"
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
  clang::format::FormattingAttemptStatus Status;
  StringRef StyleStr = "file"; // DPCTFormatStyle::custom
  if (clang::dpct::DpctGlobalInfo::getFormatStyle() ==
      DPCTFormatStyle::google) {
    StyleStr = "google";
  } else if (clang::dpct::DpctGlobalInfo::getFormatStyle() ==
             DPCTFormatStyle::llvm) {
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

int run(int argc, const char **argv) {

  if (argc < 2) {
    std::cout << CtHelpHint;
    return MigrationErrorShowHelp;
  }
  GAnalytics("");
#if defined(__linux__) || defined(_WIN32)
  InstallSignalHandle();
#endif

#if defined(_WIN32)
  // To support wildcard "*" in source file name in windows.
  llvm::InitLLVM X(argc, argv);
#endif

  // Set hangle for libclangTooling to proccess message for dpct
  clang::tooling::SetPrintHandler(PrintMsg);
  clang::tooling::SetFileSetInCompiationDB(dpct::DpctGlobalInfo::getFileSetInCompiationDB());

  // CommonOptionsParser will adjust argc to the index of "--"
  int OriginalArgc = argc;
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
              DebugInfo::ShowStatus(MigrationErrorCannotParseDatabase);
              exit(MigrationErrorCannotParseDatabase);
            } else if (DE.EC == -102) {
              DebugInfo::ShowStatus(MigrationErrorCannotFindDatabase);
              exit(MigrationErrorCannotFindDatabase);
            } else {
              DebugInfo::ShowStatus(MigrationError);
              exit(MigrationError);
            }
          });
    }
    // Filter and output error messages emitted by clang
    auto E =
        handleErrors(OptParser.takeError(), [](const llvm::StringError &E) {
          DpctLog() << E.getMessage();
        });
    dpct::DebugInfo::ShowStatus(MigrationOptionParsingError);
    exit(MigrationOptionParsingError);
  }

  if (!OutputFile.empty()) {
      //Set handle for libclangTooling to redirect warning message to DpctTerm
      clang::tooling::SetDiagnosticOutput(DpctTerm());
  }

  initWarningIDs();
  if (InRoot.size() >= MAX_PATH_LEN - 1) {
    DpctLog() << "Error: --in-root '" << InRoot << "' is too long\n";
    DebugInfo::ShowStatus(MigrationErrorPathTooLong);
    exit(MigrationErrorPathTooLong);
  }
  if (OutRoot.size() >= MAX_PATH_LEN - 1) {
    DpctLog() << "Error: --out-root '" << OutRoot << "' is too long\n";
    DebugInfo::ShowStatus(MigrationErrorPathTooLong);
    exit(MigrationErrorPathTooLong);
  }
  if (SDKIncludePath.size() >= MAX_PATH_LEN - 1) {
    DpctLog() << "Error: --cuda-include-path '" << SDKIncludePath
              << "' is too long\n";
    DebugInfo::ShowStatus(MigrationErrorPathTooLong);
    exit(MigrationErrorPathTooLong);
  }
  if (OutputFile.size() >= MAX_PATH_LEN - 1) {
    DpctLog() << "Error: --output-file '" << OutputFile << "' is too long\n";
    DebugInfo::ShowStatus(MigrationErrorPathTooLong);
    exit(MigrationErrorPathTooLong);
  }
  // Report file prefix is limited to 128, so that <report-type> and
  // <report-format> can be extended later
  if (ReportFilePrefix.size() >= 128) {
    DpctLog() << "Error: --report-file-prefix '" << ReportFilePrefix
              << "' is too long\n";
    DebugInfo::ShowStatus(MigrationErrorPrefixTooLong);
    exit(MigrationErrorPrefixTooLong);
  }
  auto P = std::find_if_not(
      ReportFilePrefix.begin(), ReportFilePrefix.end(),
      [](char C) { return ::isalpha(C) || ::isdigit(C) || C == '_'; });
  if (P != ReportFilePrefix.end()) {
    DpctLog() << "Error: --report-file-prefix contains special character '"
              << *P << "' \n";
    DebugInfo::ShowStatus(MigrationErrorSpecialCharacter);
    exit(MigrationErrorSpecialCharacter);
  }
  clock_t StartTime = clock();
  // just show -- --help information and then exit
  if (CommonOptionsParser::hasHelpOption(OriginalArgc, argv))
    exit(MigrationSucceeded);
  if (InRoot.empty() && ProcessAllFlag) {
    DebugInfo::ShowStatus(MigrationErrorNoExplicitInRoot);
    exit(MigrationErrorNoExplicitInRoot);
  }
  if (!makeCanonicalOrSetDefaults(InRoot, OutRoot,
                                  OptParser->getSourcePathList())) {
    DebugInfo::ShowStatus(MigrationErrorInvalidInRootOrOutRoot);
    exit(MigrationErrorInvalidInRootOrOutRoot);
  }

  int ValidPath=validatePaths(InRoot, OptParser->getSourcePathList());
  if (ValidPath == -1) {
    DebugInfo::ShowStatus(MigrationErrorInvalidInRootPath);
    exit(MigrationErrorInvalidInRootPath);
  } else if (ValidPath==-2) {
    DebugInfo::ShowStatus(MigrationErrorNoFileTypeAvail);
    exit(MigrationErrorNoFileTypeAvail);
  }

  int SDKIncPathRes =
      checkSDKPathOrIncludePath(SDKIncludePath, RealSDKIncludePath);
  if (SDKIncPathRes == -1) {
    DebugInfo::ShowStatus(MigrationErrorInvalidSDKPath);
    exit(MigrationErrorInvalidSDKPath);
  } else if (SDKIncPathRes == 0) {
    HasSDKIncludeOption = true;
  }

  int SDKPathRes = checkSDKPathOrIncludePath(SDKPath, RealSDKPath);
  if (SDKPathRes == -1) {
    DebugInfo::ShowStatus(MigrationErrorInvalidSDKPath);
    exit(MigrationErrorInvalidSDKPath);
  } else if (SDKPathRes == 0) {
    HasSDKPathOption = true;
  }

  bool GenReport = false;
  #ifdef DPCT_DEBUG_BUILD
  std::string &DVerbose = DiagsContent;
  #else
  std::string DVerbose ="";
  #endif
  if (checkReportArgs(ReportType.getValue(), ReportFormat.getValue(),
                      ReportFilePrefix, ReportOnlyFlag, GenReport,
                      DVerbose) == false) {
    DebugInfo::ShowStatus(MigrationErrorInvalidReportArgs);
    exit(MigrationErrorInvalidReportArgs);
  }

  if (GenReport) {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "Generate report: "
       << "report-type:"
       << (ReportType.getValue() == ReportTypeEnum::all
               ? "all"
               : (ReportType.getValue() == ReportTypeEnum::apis
                      ? "apis"
                      : (ReportType.getValue() == ReportTypeEnum::stats
                             ? "stats"
                             : "diags")))
       << ", report-format:"
       << (ReportFormat.getValue() == ReportFormatEnum::csv ? "csv"
                                                            : "formatted")
       << ", report-file-prefix:" << ReportFilePrefix << "\n";

    PrintMsg(OS.str());
  }

  // TODO: implement one of this for each source language.
  CudaPath = getCudaInstallPath(OriginalArgc, argv);
  DpctDiags() << "Cuda Include Path found: " << CudaPath << "\n";

  RefactoringTool Tool(OptParser->getCompilations(),
                       OptParser->getSourcePathList());
  DpctInstallPath = getInstallPath(Tool, argv[0]);

  ValidateInputDirectory(Tool, InRoot);

  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-nocudalib", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "--cuda-host-only", ArgumentInsertPosition::BEGIN));

  std::string CUDAVerMajor = "-D__CUDACC_VER_MAJOR__=" + std::to_string(SDKVersionMajor);
  Tool.appendArgumentsAdjuster(
    getInsertArgumentAdjuster(CUDAVerMajor.c_str(), ArgumentInsertPosition::BEGIN));

  std::string CUDAVerMinor = "-D__CUDACC_VER_MINOR__=" + std::to_string(SDKVersionMinor);
  Tool.appendArgumentsAdjuster(
    getInsertArgumentAdjuster(CUDAVerMinor.c_str(), ArgumentInsertPosition::BEGIN));
  Tool.appendArgumentsAdjuster(
    getInsertArgumentAdjuster("-D__NVCC__", ArgumentInsertPosition::BEGIN));

  SetSDKIncludePath(CudaPath);

#ifdef _WIN32
  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-fms-compatibility-version=19.00.24215.1",
                                ArgumentInsertPosition::BEGIN));
#endif
  DpctGlobalInfo::setInRoot(InRoot);
  DpctGlobalInfo::setCudaPath(CudaPath);
  DpctGlobalInfo::setKeepOriginCode(KeepOriginalCodeFlag);
  DpctGlobalInfo::setSyclNamedLambda(SyclNamedLambdaFlag);
  DpctGlobalInfo::setUsmLevel(USMLevel);
  DpctGlobalInfo::setFormatRange(FormatRng);
  DpctGlobalInfo::setFormatStyle(FormatST);
  DpctGlobalInfo::setCtadEnabled(EnableCTAD);
  DpctGlobalInfo::setCommentsEnabled(EnableComments);
  DpctGlobalInfo::setUsingDRYPattern(!NoDRYPatternFlag);
  StopOnParseErrTooling = StopOnParseErr;
  InRootTooling=InRoot;

  MapNames::setClNamespace(ExplicitClNamespace);
  if (DpctGlobalInfo::getFormatRange() != clang::format::FormatRange::none) {
    parseFormatStyle();
  }

  DPCTActionFactory Factory(Tool.getReplacements());
  if (int RunResult = Tool.run(&Factory) && StopOnParseErr) {
    DumpOutputFile();
    if (RunResult == 1) {
      DebugInfo::ShowStatus(MigrationErrorFileParseError);
      return MigrationErrorFileParseError;
    } else {
      // When RunResult equals to 2, it means no error but some files are
      // skipped due to missing compile commands.
      // And clang::tooling::ReFactoryTool will emit error message.
      return MigrationSKIPForMissingCompileCommand;
    }
  }

  int RetJmp=0;
  CHECKPOINT_ReplacementPostProcess_ENTRY(RetJmp);
  if(RetJmp==0) {
    auto &Global = DpctGlobalInfo::getInstance();
    Global.buildReplacements();
    Global.emplaceReplacements(Tool.getReplacements());
  }
  CHECKPOINT_ReplacementPostProcess_EXIT();

  if (GenReport) {
    // report: apis, stats, all, diags
    if (ReportType.getValue() == ReportTypeEnum::all ||
        ReportType.getValue() == ReportTypeEnum::apis)
      saveApisReport();

    if (ReportType.getValue() == ReportTypeEnum::all ||
        ReportType.getValue() == ReportTypeEnum::stats) {
      clock_t EndTime = clock();
      double Duration = (double)(EndTime - StartTime) / (CLOCKS_PER_SEC / 1000);
      saveStatsReport(Tool, Duration);
    }
    // all doesn't include diags.
    if (ReportType.getValue() == ReportTypeEnum::diags) {
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
