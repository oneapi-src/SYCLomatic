//===--- C2S.cpp -----------------------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "clang/C2S/C2S.h"
#include "MisleadingBidirectional.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "CallExprRewriter.h"
#include "Checkpoint.h"
#include "Config.h"
#include "CustomHelperFiles.h"
#include "ExternalReplacement.h"
#include "GenMakefile.h"
#include "IncrementalMigrationUtility.h"
#include "SaveNewFiles.h"
#include "SignalProcess.h"
#include "Statics.h"
#include "Utility.h"
#include "Rules.h"
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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

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
using namespace clang::c2s;
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
namespace c2s {
llvm::cl::OptionCategory &C2SCat = llvm::cl::getC2SCategory();
void initWarningIDs();
} // namespace c2s
} // namespace clang

// clang-format off
const char *const C2SHelpMessage =
    "\n"
    "<source0> ... Paths of input source files. These paths are looked up in "
    "the compilation database.\n\n"
    "EXAMPLES:\n\n"
    "Migrate single source file:\n\n"
    "  c2s source.cpp\n\n"
    "Migrate single source file with C++11 features:\n\n"
    "  c2s --extra-arg=\"-std=c++11\" source.cpp\n\n"
    "Migrate all files available in compilation database:\n\n"
    "  c2s -p=<path to location of compilation database file>\n\n"
    "Migrate one file in compilation database:\n\n"
    "  c2s -p=<path to location of compilation database file>  source.cpp\n\n"
#if defined(_WIN32)
    "Migrate all files available in vcxprojfile:\n\n"
    "  c2s --vcxprojfile=path/to/vcxprojfile.vcxproj\n"
#endif
    DiagRef
    ;

const char *const C2SHelpHint =
    "  Warning: Please specify file(s) to be migrated.\n"
    "  To get help on the tool usage, run: c2s --help\n"
    "\n";

static extrahelp CommonHelp(C2SHelpMessage);
static opt<std::string> Passes(
    "passes",
    desc("Comma separated list of migration passes, which will be applied.\n"
         "Only the specified passes are applied."),
    value_desc("IterationSpaceBuiltinRule,..."), cat(C2SCat),
               llvm::cl::Hidden);
static opt<std::string>
    InRoot("in-root",
           desc("The directory path for the root of the source tree that needs "
                "to be migrated.\n"
                "Only files under this root are migrated. Default: Current"
                " directory, if input\nsource files are not provided. "
                "If input source files are provided, the directory\n"
                "of the first input source file is used."),
           value_desc("dir"), cat(C2SCat),
           llvm::cl::Optional);
static opt<std::string> OutRoot(
    "out-root",
    desc("The directory path for root of generated files. A directory is "
         "created if it\n"
         "does not exist. Default: c2s_output."),
    value_desc("dir"), cat(C2SCat), llvm::cl::Optional);

static opt<std::string> SDKPath("cuda-path", desc("Directory path of SDK.\n"),
                                value_desc("dir"), cat(C2SCat),
                                llvm::cl::Optional, llvm::cl::Hidden);

static opt<std::string>
    CudaIncludePath("cuda-include-path",
                   desc("The directory path of the CUDA header files."),
                   value_desc("dir"), cat(C2SCat), llvm::cl::Optional);

static opt<ReportTypeEnum> ReportType(
    "report-type", desc("Specifies the type of report. Values are:\n"),
    llvm::cl::values(
        llvm::cl::OptionEnumValue{"apis", int(ReportTypeEnum::RTE_APIs),
            "Information about API signatures that need migration and the number of times\n"
            "they were encountered. The report file name will have .apis suffix added.", false},
        llvm::cl::OptionEnumValue{"stats", int(ReportTypeEnum::RTE_Stats),
                  "High level migration statistics: Lines Of Code (LOC) that are migrated to\n"
                  "DPC++, LOC migrated to DPC++ with helper functions, LOC not needing migration,\n"
                  "LOC needing migration but are not migrated. The report file name has the .stats\n"
                  "suffix added (default)", false},
        llvm::cl::OptionEnumValue{"all", int(ReportTypeEnum::RTE_All),
                  "All of the reports.", false}
        #ifdef C2S_DEBUG_BUILD
        , llvm::cl::OptionEnumValue{"diags", int(ReportTypeEnum::RTE_Diags),
                  "diags information", true}
        #endif
        ),
    llvm::cl::init(ReportTypeEnum::RTE_NotSetType), value_desc("value"), cat(C2SCat),
    llvm::cl::Optional);

static opt<ReportFormatEnum> ReportFormat(
    "report-format", desc("Format of the reports:\n"),
    llvm::cl::values(
        llvm::cl::OptionEnumValue{"csv", int(ReportFormatEnum::RFE_CSV),
                  "Output is lines of comma separated values. The report file "
                  "name extension will\n"
                  "be .csv. (default)", false},
        llvm::cl::OptionEnumValue{"formatted", int(ReportFormatEnum::RFE_Formatted),
                  "Output is formatted to be easier to read for "
                  "human eyes. Report file name\n"
                  "extension will be log.",
                  false}),
    llvm::cl::init(ReportFormatEnum::RFE_NotSetFormat), value_desc("value"), cat(C2SCat),
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
    value_desc("prefix"), cat(C2SCat), llvm::cl::Optional);
bool ReportOnlyFlag = false;
static opt<bool, true>
    ReportOnly("report-only",
               llvm::cl::desc("Only reports are generated. No DPC++ code is "
                              "generated. Default: off."),
               cat(C2SCat), llvm::cl::location(ReportOnlyFlag));

bool KeepOriginalCodeFlag = false;

static opt<bool, true>
    ShowOrigCode("keep-original-code",
                 llvm::cl::desc("Keeps the original code in comments of "
                                "generated DPC++ files. Default: off.\n"),
                 cat(C2SCat), llvm::cl::location(KeepOriginalCodeFlag));
#ifdef C2S_DEBUG_BUILD
static opt<std::string>
    DiagsContent("report-diags-content",
                 desc("Diagnostics verbosity level. \"pass\": Basic migration "
                      "pass information. "
                      "\"transformation\": Detailed migration pass "
                      "transformation information."),
                 value_desc("[pass|transformation]"), cat(C2SCat),
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
                                  value_desc("value"), cat(C2SCat));

bool SuppressWarningsAllFlag = false;
static std::string WarningAllDesc("Suppresses all migration warnings. Default: off.");
opt<bool, true> SuppressWarningsAll("suppress-warnings-all",
                                    desc(WarningAllDesc), cat(C2SCat),
                                    location(SuppressWarningsAllFlag));
bool StopOnParseErr = false;
static opt<bool, true>
    StopOnParseErrOption("stop-on-parse-err",
                llvm::cl::desc("Stop migration and generation of reports if "
                               "parsing errors happened. Default: off. \n"),
                cat(C2SCat), llvm::cl::location(StopOnParseErr));

bool CheckUnicodeSecurityFlag = false;
static opt<bool, true> CheckUnicodeSecurity(
    "check-unicode-security",
    llvm::cl::desc("Enable detection and warnings about Unicode constructs that can be exploited by using\n"
                   "bi-directional formatting codes and homoglyphs in identifiers. Default: off.\n"),
    cat(C2SCat), llvm::cl::location(CheckUnicodeSecurityFlag));

bool SyclNamedLambdaFlag = false;
static opt<bool, true>
    SyclNamedLambda("sycl-named-lambda",
                llvm::cl::desc("Generates kernels with the kernel name. Default: off.\n"),
                cat(C2SCat), llvm::cl::location(SyclNamedLambdaFlag));

opt<OutputVerbosityLevel> OutputVerbosity(
    "output-verbosity", llvm::cl::desc("Sets the output verbosity level:"),
    llvm::cl::values(
        llvm::cl::OptionEnumValue{"silent", int(OutputVerbosityLevel::OVL_Silent),
                                  "Only messages from clang.", false},
        llvm::cl::OptionEnumValue{"normal", int(OutputVerbosityLevel::OVL_Normal),
                                  "\'silent\' and warnings, errors, and notes from c2s.",
                                  false},
        llvm::cl::OptionEnumValue{"detailed", int(OutputVerbosityLevel::OVL_Detailed),
                                  "\'normal\' and messages about which file is being processed.",
                                  false},
        llvm::cl::OptionEnumValue{"diagnostics", int(OutputVerbosityLevel::OVL_Diagnostics),
                                  "\'detailed\' and information about the detected "
                                  "conflicts and crashes. (default)", false}),
    llvm::cl::init(OutputVerbosityLevel::OVL_Diagnostics), value_desc("value"), cat(C2SCat),
    llvm::cl::Optional);

opt<std::string>
    OutputFile("output-file",
               desc("Redirects the stdout/stderr output to <file> in the output"
                    " directory specified\n"
                    "by the --out-root option."),
               value_desc("file"), cat(C2SCat),
               llvm::cl::Optional);

list<std::string> RuleFile("rule-file", desc("Specifies the rule file path that contains rules used for migration.\n"),
                           value_desc("file"), cat(C2SCat), llvm::cl::ZeroOrMore);

opt<UsmLevel> USMLevel(
    "usm-level", desc("Sets the USM level to use in source code generation.\n"),
    values(llvm::cl::OptionEnumValue{"restricted", int(UsmLevel::UL_Restricted),
                     "Uses API from DPC++ Explicit and Restricted Unified "
                     "Shared Memory extension\n"
                     "for memory management migration. (default)", false},
           llvm::cl::OptionEnumValue{"none", int(UsmLevel::UL_None),
                     "Uses helper functions from C2S header files for memory "
                     "management migration.", false}),
    init(UsmLevel::UL_Restricted), value_desc("value"), cat(C2SCat), llvm::cl::Optional);

opt<format::FormatRange>
    FormatRng("format-range",
                llvm::cl::desc("Sets the range of formatting.\nThe values are:\n"),
                values(llvm::cl::OptionEnumValue{"migrated", int(format::FormatRange::migrated),
                     "Only formats the migrated code (default).", false},
                       llvm::cl::OptionEnumValue{"all", int(format::FormatRange::all),
                     "Formats all code.", false},
                       llvm::cl::OptionEnumValue{"none", int(format::FormatRange::none),
                     "Do not format any code.", false}),
    init(format::FormatRange::migrated), value_desc("value"), cat(C2SCat), llvm::cl::Optional);

opt<C2SFormatStyle>
    FormatST("format-style",
                llvm::cl::desc("Sets the formatting style.\nThe values are:\n"),
                values(llvm::cl::OptionEnumValue{"llvm", int(C2SFormatStyle::FS_LLVM),
                     "Use the LLVM coding style.", false},
                       llvm::cl::OptionEnumValue{"google", int(C2SFormatStyle::FS_Google),
                     "Use the Google coding style.", false},
                       llvm::cl::OptionEnumValue{"custom", int(C2SFormatStyle::FS_Custom),
                     "Use the coding style defined in the .clang-format file (default).", false}),
    init(C2SFormatStyle::FS_Custom), value_desc("value"), cat(C2SCat), llvm::cl::Optional);

bool ExplicitClNamespace = false;
static opt<bool, true> NoClNamespaceInline(
    "no-cl-namespace-inline",
    llvm::cl::desc("DEPRECATED: Do not use cl:: namespace inline. Default: off. This option will be\n"
                   "ignored if the replacement option --use-explicit-namespace is used.\n"),
    cat(C2SCat), llvm::cl::location(ExplicitClNamespace));

bool NoDRYPatternFlag = false;
static opt<bool, true> NoDRYPattern(
  "no-dry-pattern", llvm::cl::desc("Do not use DRY (do not repeat yourself) pattern when functions from c2s\n"
                                   "namespace are inserted. Default: off.\n"),
  cat(C2SCat), llvm::cl::location(NoDRYPatternFlag));

bool NoUseGenericSpaceFlag = false;
static opt<bool, true> NoUseGenericSpace(
  "no-use-generic-space", llvm::cl::desc("sycl::access::address_space::generic_space is not used during atomic\n"
                                         " function's migration. Default: off.\n"),
  cat(C2SCat), llvm::cl::location(NoUseGenericSpaceFlag), llvm::cl::ReallyHidden);

bool ProcessAllFlag = false;
static opt<bool, true>
    ProcessAll("process-all",
                 llvm::cl::desc("Migrates or copies all files, except hidden, from the --in-root directory\n"
                 "to the --out-root directory. The --in-root option should be explicitly specified.\n"
                 "Default: off."),
                 cat(C2SCat), llvm::cl::location(ProcessAllFlag));

static opt<bool> EnableCTAD(
    "enable-ctad",
    llvm::cl::desc("Use a C++17 class template argument deduction (CTAD) in "
                   "your generated code.\n"
                   "Default: off."),
    cat(C2SCat), init(false));

static opt<bool> EnableComments(
    "comments", llvm::cl::desc("Insert comments explaining the generated code. Default: off."),
    cat(C2SCat), init(false));

static opt<HelperFilesCustomizationLevel> UseCustomHelperFileLevel(
    "use-custom-helper", desc("Customize the helper header files for migrated code. The values are:\n"),
    values(
        llvm::cl::OptionEnumValue{
            "none", int(HelperFilesCustomizationLevel::HFCL_None),
            "No customization (default).", false},
        llvm::cl::OptionEnumValue{
            "file", int(HelperFilesCustomizationLevel::HFCL_File),
            "Limit helper header files to only the necessary files for the migrated code and\n"
            "place them in the --out-root directory.", false},
        llvm::cl::OptionEnumValue{
            "api", int(HelperFilesCustomizationLevel::HFCL_API),
            "Limit helper header files to only the necessary APIs for the migrated code and\n"
            "place them in the --out-root directory.", false},
        llvm::cl::OptionEnumValue{
            "all", int(HelperFilesCustomizationLevel::HFCL_All),
            "Generate a complete set of helper header files and place them in the --out-root\n"
            "directory.", false}),
    init(HelperFilesCustomizationLevel::HFCL_None), value_desc("value"),
    cat(C2SCat), llvm::cl::Optional);

opt<std::string> CustomHelperFileName(
    "custom-helper-name",
    desc(
        "Specifies the helper headers folder name and main helper header file name.\n"
        "Default: c2s."),
    init("c2s"), value_desc("name"), cat(C2SCat), llvm::cl::Optional);

bool AsyncHandlerFlag = false;
static opt<bool, true>
    AsyncHandler("always-use-async-handler",
                 llvm::cl::desc("Always create the cl::sycl::queue with an async "
                                "exception handler. Default: off."),
                 cat(C2SCat), llvm::cl::location(AsyncHandlerFlag));

static opt<AssumedNDRangeDimEnum> NDRangeDim(
    "assume-nd-range-dim",
    desc("Provides a hint to the tool on the dimensionality of nd_range to use in generated code.\n"
         "The values are:\n"),
    values(
        llvm::cl::OptionEnumValue{"1", 1,
                                  "Generate kernel code assuming 1D nd_range "
                                  "where possible, and 3D in other cases.",
                                  false},
        llvm::cl::OptionEnumValue{
            "3", 3,
            "Generate kernel code assuming 3D nd_range (default).",
            false}),
    init(AssumedNDRangeDimEnum::ARE_Dim3), value_desc("value"), cat(C2SCat),
    llvm::cl::Optional);

static list<ExplicitNamespace> UseExplicitNamespace(
    "use-explicit-namespace",
    llvm::cl::desc(
        "Defines the namespaces to use explicitly in generated code. The value is a comma\n"
        "separated list. Default: c2s, sycl.\n"
        "Possible values are:"),
    llvm::cl::CommaSeparated,
    values(llvm::cl::OptionEnumValue{"none", int(ExplicitNamespace::EN_None),
                                     "Generate code without namespaces. Cannot "
                                     "be used with other values.",
                                     false},
           llvm::cl::OptionEnumValue{
               "cl", int(ExplicitNamespace::EN_CL),
               "Generate code with cl::sycl:: namespace. Cannot be used with "
               "sycl or sycl-math values.",
               false},
           llvm::cl::OptionEnumValue{"dpct", int(ExplicitNamespace::EN_DPCT),
                                     "DEPRECATED: Generate code with c2s:: namespace. Please use c2s instead.",
                                     false},
           llvm::cl::OptionEnumValue{"c2s", int(ExplicitNamespace::EN_C2S),
                                     "Generate code with c2s:: namespace.",
                                     false},
           llvm::cl::OptionEnumValue{
               "sycl", int(ExplicitNamespace::EN_SYCL),
               "Generate code with sycl:: namespace. Cannot be used with cl or "
               "sycl-math values.",
               false},
           llvm::cl::OptionEnumValue{
               "sycl-math", int(ExplicitNamespace::EN_SYCL_Math),
               "Generate code with sycl:: namespace, applied only for SYCL math functions.\n"
               "Cannot be used with cl or sycl values.",
               false}),
    value_desc("value"), cat(C2SCat), llvm::cl::ZeroOrMore);

// When more dpcpp extensions are implemented, more extension names will be
// added into the value of option --no-dpcpp-extensions, currently only
// Enqueued barriers is supported.
static list<DPCPPExtensions> NoDPCPPExtensions(
    "no-dpcpp-extensions",
    llvm::cl::desc(
        "Comma separated list of DPC++ extensions not to be used in migrated "
        "code.\n"
        "By default, these extensions will be used in migrated code."),
    llvm::cl::CommaSeparated,
    values(llvm::cl::OptionEnumValue{"enqueued_barriers",
                                     int(DPCPPExtensions::Ext_EnqueueBarrier),
                                     "Enqueued barriers DPC++ extension.",
                                     false}),
    value_desc("value"), cat(C2SCat), llvm::cl::ZeroOrMore,
    llvm::cl::cb<void, DPCPPExtensions>(C2SGlobalInfo::setExtensionUnused));

static bits<ExperimentalFeatures> Experimentals(
  "use-experimental-features",
  llvm::cl::desc(
    "Comma separated list of experimental features to be used in migrated "
    "code.\n"
    "By default, experimental features will not be used in migrated code.\nThe values are:\n"),
  llvm::cl::CommaSeparated,
  values(
    llvm::cl::OptionEnumValue{
        "nd_range_barrier", int(ExperimentalFeatures::Exp_NdRangeBarrier),
        "Experimental helper function used to help cross group synchronization during migration.\n",
        false },
    llvm::cl::OptionEnumValue{
        "free-function-queries", int(ExperimentalFeatures::Exp_FreeQueries),
        "Experimental extension that allows getting `id`, `item`, `nd_item`, `group`, and\n"
        "`sub_group` instances globally.",
        false },
    llvm::cl::OptionEnumValue{
        "local-memory-kernel-scope-allocation", int(ExperimentalFeatures::Exp_GroupSharedMemory),
        "Experimental extension that allows allocation of local memory objects at the kernel\n"
        "functor scope",
        false }),
  value_desc("value"), cat(C2SCat), llvm::cl::ZeroOrMore);

opt<bool> GenBuildScript(
    "gen-build-script",
    llvm::cl::desc("Generates makefile for migrated file(s) in -out-root directory. Default: off."),
    cat(C2SCat), init(false));

opt<std::string>
    BuildScriptFile("build-script-file",
               desc("Specifies the name of generated makefile for migrated file(s).\n"
                    "Default name: Makefile.c2s."),
               value_desc("file"), cat(C2SCat),
               llvm::cl::Optional);

static list<std::string> ExcludePathList(
    "in-root-exclude",
    llvm::cl::desc(
        "Excludes the specified directory or file from processing."),
    value_desc("dir|file"), cat(C2SCat), llvm::cl::ZeroOrMore);

static opt<bool> OptimizeMigration(
    "optimize-migration",
    llvm::cl::desc("Generates DPC++ code applying more aggressive assumptions that potentially\n"
                   "may alter the semantics of your program. Default: off."),
    cat(C2SCat), init(false));

static opt<bool> NoIncrementalMigration(
    "no-incremental-migration",
    llvm::cl::desc("Tells the tool to not perform an incremental migration.\n"
                   "Default: off (incremental migration happens)."),
    cat(C2SCat), init(false));
// clang-format on

// TODO: implement one of this for each source language.
std::string CudaPath;
std::string C2SInstallPath;
std::unordered_map<std::string, bool> ChildOrSameCache;
std::unordered_map<std::string, bool> ChildPathCache;
std::unordered_map<std::string, llvm::SmallString<256>> RealPathCache;
std::unordered_map<std::string, bool> IsDirectoryCache;
int FatalErrorCnt = 0;
extern bool StopOnParseErrTooling;
extern std::string InRootTooling;
JMP_BUF CPFileASTMaterEnter;
JMP_BUF CPRepPostprocessEnter;
JMP_BUF CPFormatCodeEnter;

class C2SConsumer : public ASTConsumer {
public:
  C2SConsumer(ReplTy &R, CompilerInstance &CI, StringRef InFile)
      : ATM(CI, InRoot), Repl(R), PP(CI.getPreprocessor()), CI(CI) {
    if (Passes != "") {
      // Separate string into list by comma
      auto Names = split(Passes, ',');

      for (auto const &Name : Names) {
        auto *ID = ASTTraversalMetaInfo::getID(Name);
        ATM.emplaceMigrationRule(ID);
      }
    } else {
      ATM.emplaceAllRules();
    }
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    if (StopOnParseErr && Context.getDiagnostics().getClient() &&
        Context.getDiagnostics().getClient()->getNumErrors() > 0) {
      return;
    }
    // The migration process is separated into two stages:
    // 1) Analysis of AST and identification of applicable migration rules
    // 2) Generation of actual textual Replacements
    // Such separation makes it possible to post-process the list of identified
    // migration rules before applying them.
    ATM.matchAST(Context, TransformSet, SSM);

    auto &Global = C2SGlobalInfo::getInstance();
    std::unordered_set<std::string> DuplicateFilter;
    for (const auto &I : TransformSet) {
      auto Repl = I->getReplacement(Context);

      // When processing __constant__ between two executions, tool may set the
      // replacement from TextModification as nullptr to ignore this
      // replacement.
      if (Repl == nullptr)
        continue;

      // For file path got in AST may be different with the one in preprocessing
      // stage, here only the file name is used to retrieve IncludeMapSet.
      const std::string FileName =
          llvm::sys::path::filename(Repl->getFilePath()).str();
      if (DuplicateFilter.find(FileName) == end(DuplicateFilter)) {
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

    StaticsInfo::printReplacements(TransformSet, Context);
  }

  void Initialize(ASTContext &Context) override {
    // Set Context for build information
    C2SGlobalInfo::setCompilerInstance(CI);

    PP.addPPCallbacks(std::make_unique<IncludesCallbacks>(
        TransformSet, IncludeMapSet, Context.getSourceManager(), ATM));

    if (C2SGlobalInfo::getCheckUnicodeSecurityFlag()) {
      CommentHandler =
          std::make_shared<MisleadingBidirectionalHandler>(TransformSet);
      PP.addCommentHandler(CommentHandler.get());
    }
  }

  void HandleCXXExplicitFunctionInstantiation(
      const FunctionDecl *Specialization, const FunctionTypeLoc &FTL,
      const ParsedAttributes &Attrs,
      const TemplateArgumentListInfo &TAList) override {
    if (!FTL || !Specialization)
      return;
    ExplicitInstantiationDecl::processFunctionTypeLoc(FTL);
    if (Specialization->getTemplateSpecializationKind() !=
        TSK_ExplicitInstantiationDefinition)
      return;
    if (Specialization->hasAttr<CUDADeviceAttr>() ||
        Specialization->hasAttr<CUDAGlobalAttr>()) {
      DeviceFunctionDecl::LinkExplicitInstantiation(Specialization, FTL, Attrs,
                                                    TAList);
    }
  }

  ~C2SConsumer() {
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
  std::shared_ptr<MisleadingBidirectionalHandler> CommentHandler;
};

class C2SAction : public ASTFrontendAction {
  ReplTy &Repl;

public:
  C2SAction(ReplTy &R) : Repl(R) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<C2SConsumer>(Repl, CI, InFile);
  }

  bool usesPreprocessorOnly() const override { return false; }
};

// Object of this class will be handed to RefactoringTool::run and will create
// the Action.
class C2SActionFactory : public FrontendActionFactory {
  ReplTy &Repl;

public:
  C2SActionFactory(ReplTy &R) : Repl(R) {}
  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<C2SAction>(Repl);
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
  driver::Driver Driver("", llvm::sys::getDefaultTargetTriple(), E);
  driver::CudaInstallationDetector CudaIncludeDetector(
      Driver, llvm::Triple(Driver.getTargetTriple()), ParsedArgs);

  std::string Path = CudaIncludeDetector.getInstallPath().str();

  if (!CudaIncludePath.empty()) {
    if (!CudaIncludeDetector.isIncludePathValid()) {
      ShowStatus(MigrationErrorInvalidCudaIncludePath);
      c2sExit(MigrationErrorInvalidCudaIncludePath);
    }

    if (!CudaIncludeDetector.isVersionSupported()) {
      ShowStatus(MigrationErrorCudaVersionUnsupported);
      c2sExit(MigrationErrorCudaVersionUnsupported);
    }
  } else if (!CudaIncludeDetector.isSupportedVersionAvailable()) {
    ShowStatus(MigrationErrorSupportedCudaVersionNotAvailable);
    c2sExit(MigrationErrorSupportedCudaVersionNotAvailable);
  }

  makeCanonical(Path);

  SmallString<512> CudaPathAbs;
  std::error_code EC = llvm::sys::fs::real_path(Path, CudaPathAbs);
  if ((bool)EC) {
    ShowStatus(MigrationErrorInvalidCudaIncludePath);
    c2sExit(MigrationErrorInvalidCudaIncludePath);
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
    ShowStatus(MigrationErrorInvalidInstallPath);
    c2sExit(MigrationErrorInvalidInstallPath);
  }
  return InstallPathAbs.str().str();
}

// To validate the root path of the project to be migrated.
void ValidateInputDirectory(clang::tooling::RefactoringTool &Tool,
                            std::string &InRoot) {

  if (isChildOrSamePath(CudaPath, InRoot)) {
    ShowStatus(MigrationErrorRunFromSDKFolder);
    c2sExit(MigrationErrorRunFromSDKFolder);
  }

  if (isChildOrSamePath(InRoot, CudaPath)) {
    ShowStatus(MigrationErrorInRootContainSDKFolder);
    c2sExit(MigrationErrorInRootContainSDKFolder);
  }

  if (isChildOrSamePath(InRoot, C2SInstallPath)) {
    ShowStatus(MigrationErrorInRootContainCTTool);
    c2sExit(MigrationErrorInRootContainCTTool);
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
    c2sExit(MigrationErrorInvalidFilePath);
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
      C2SStats() << "\n";
      C2SStats() << "File name, LOC migrated to DPC++, LOC migrated to helper "
                     "functions, "
                     "LOC not needed to migrate, LOC not able to migrate";
      C2SStats() << "\n";
    }
    C2SStats() << Elem.first + ", " + std::to_string(TransToSYCL) + ", " +
                       std::to_string(TransToAPI) + ", " +
                       std::to_string(NotTrans) + ", " +
                       std::to_string(NotSupport);
    C2SStats() << "\n";
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
  C2SStats() << "\nTotal migration time: " + std::to_string(Duration) +
                     " ms\n";
  if (ReportFilePrefix == "stdout") {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "----------Stats report---------------\n";
    OS << getC2SStatsStr() << "\n";
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
    File << getC2SStatsStr() << "\n";
  }
}

static void saveDiagsReport() {

  // C2SDiags() << "\n";
  if (ReportFilePrefix == "stdout") {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "--------Diags message----------------\n";
    OS << getC2SDiagsStr() << "\n";
    OS << "-------------------------------------\n";
    PrintMsg(OS.str());
  } else {
    std::string RFile = OutRoot + "/" + ReportFilePrefix + ".diags.log";
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(RFile, std::ios::binary);
    File << getC2SDiagsStr() << "\n";
  }
}

std::string printCTVersion() {

  std::string buf;
  llvm::raw_string_ostream OS(buf);

  OS << "\n" << C2S_TOOL_NAME << " version " << C2S_VERSION_MAJOR
     << "." << C2S_VERSION_MINOR << "." << C2S_VERSION_PATCH << "."
     << " Codebase:";
  std::string Revision = getClangRevision();
  if (!Revision.empty()) {
    OS << '(';
    if (!Revision.empty()) {
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
    File << getC2STermStr() << "\n";
  }
}

void PrintReportOnFault(std::string &FaultMsg) {
  PrintMsg(FaultMsg);
  saveApisReport();
  saveDiagsReport();

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
  clang::format::FormattingAttemptStatus Status;
  StringRef StyleStr = "file"; // C2SFormatStyle::Custom
  if (clang::c2s::C2SGlobalInfo::getFormatStyle() ==
      C2SFormatStyle::FS_Google) {
    StyleStr = "google";
  } else if (clang::c2s::C2SGlobalInfo::getFormatStyle() ==
             C2SFormatStyle::FS_LLVM) {
    StyleStr = "llvm";
  }
  std::string StyleSearchPath = clang::tooling::getFormatSearchPath().empty()
                                    ? clang::c2s::C2SGlobalInfo::getInRoot()
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

  C2SGlobalInfo::setCodeFormatStyle(Style);
}

int runC2S(int argc, const char **argv) {

  if (argc < 2) {
    std::cout << C2SHelpHint;
    return MigrationErrorShowHelp;
  }
#if defined(__linux__) || defined(_WIN32)
  InstallSignalHandle();
#endif

#if defined(_WIN32)
  // To support wildcard "*" in source file name in windows.
  llvm::InitLLVM X(argc, argv);
#endif

  // Set hangle for libclangTooling to proccess message for c2s
  clang::tooling::SetPrintHandle(PrintMsg);
  clang::tooling::SetFileSetInCompiationDB(
      c2s::C2SGlobalInfo::getFileSetInCompiationDB());

  // CommonOptionsParser will adjust argc to the index of "--"
  int OriginalArgc = argc;
  clang::tooling::SetModuleFiles(c2s::C2SGlobalInfo::getModuleFiles());
#ifdef _WIN32
  // Set function handle for libclangTooling to parse vcxproj file.
  clang::tooling::SetParserHandle(vcxprojParser);
#endif
  llvm::cl::SetVersionPrinter(
      [](llvm::raw_ostream &OS) { OS << printCTVersion() << "\n"; });
  auto OptParser =
      CommonOptionsParser::create(argc, argv, C2SCat, llvm::cl::OneOrMore);
  if (!OptParser) {
    if (OptParser.errorIsA<C2SError>()) {
      llvm::Error NewE =
          handleErrors(OptParser.takeError(), [](const C2SError &DE) {
            if (DE.EC == -101) {
              ShowStatus(MigrationErrorCannotParseDatabase);
              c2sExit(MigrationErrorCannotParseDatabase);
            } else if (DE.EC == -102) {
              ShowStatus(MigrationErrorCannotFindDatabase);
              c2sExit(MigrationErrorCannotFindDatabase);
            } else {
              ShowStatus(MigrationError);
              c2sExit(MigrationError);
            }
          });
    }
    // Filter and output error messages emitted by clang
    auto E =
        handleErrors(OptParser.takeError(), [](const llvm::StringError &E) {
          C2SLog() << E.getMessage();
        });
    c2s::ShowStatus(MigrationOptionParsingError);
    c2sExit(MigrationOptionParsingError);
  }

  if (!OutputFile.empty()) {
    // Set handle for libclangTooling to redirect warning message to C2STerm
    clang::tooling::SetDiagnosticOutput(C2STerm());
  }

  initWarningIDs();
  if (InRoot.size() >= MAX_PATH_LEN - 1) {
    C2SLog() << "Error: --in-root '" << InRoot << "' is too long\n";
    ShowStatus(MigrationErrorPathTooLong);
    c2sExit(MigrationErrorPathTooLong);
  }
  if (OutRoot.size() >= MAX_PATH_LEN - 1) {
    C2SLog() << "Error: --out-root '" << OutRoot << "' is too long\n";
    ShowStatus(MigrationErrorPathTooLong);
    c2sExit(MigrationErrorPathTooLong);
  }
  if (CudaIncludePath.size() >= MAX_PATH_LEN - 1) {
    C2SLog() << "Error: --cuda-include-path '" << CudaIncludePath
              << "' is too long\n";
    ShowStatus(MigrationErrorPathTooLong);
    c2sExit(MigrationErrorPathTooLong);
  }
  if (OutputFile.size() >= MAX_PATH_LEN - 1) {
    C2SLog() << "Error: --output-file '" << OutputFile << "' is too long\n";
    ShowStatus(MigrationErrorPathTooLong);
    c2sExit(MigrationErrorPathTooLong);
  }
  // Report file prefix is limited to 128, so that <report-type> and
  // <report-format> can be extended later
  if (ReportFilePrefix.size() >= 128) {
    C2SLog() << "Error: --report-file-prefix '" << ReportFilePrefix
              << "' is too long\n";
    ShowStatus(MigrationErrorPrefixTooLong);
    c2sExit(MigrationErrorPrefixTooLong);
  }
  auto P = std::find_if_not(
      ReportFilePrefix.begin(), ReportFilePrefix.end(),
      [](char C) { return ::isalpha(C) || ::isdigit(C) || C == '_'; });
  if (P != ReportFilePrefix.end()) {
    C2SLog() << "Error: --report-file-prefix contains special character '"
              << *P << "' \n";
    ShowStatus(MigrationErrorSpecialCharacter);
    c2sExit(MigrationErrorSpecialCharacter);
  }
  clock_t StartTime = clock();
  // just show -- --help information and then exit
  if (CommonOptionsParser::hasHelpOption(OriginalArgc, argv))
    c2sExit(MigrationSucceeded);
  if (InRoot.empty() && ProcessAllFlag) {
    ShowStatus(MigrationErrorNoExplicitInRoot);
    c2sExit(MigrationErrorNoExplicitInRoot);
  }

  if (!makeInRootCanonicalOrSetDefaults(InRoot,
                                        OptParser->getSourcePathList())) {
    ShowStatus(MigrationErrorInvalidInRootOrOutRoot);
    c2sExit(MigrationErrorInvalidInRootOrOutRoot);
  }

  int ValidPath = validatePaths(InRoot, OptParser->getSourcePathList());
  if (ValidPath == -1) {
    ShowStatus(MigrationErrorInvalidInRootPath);
    c2sExit(MigrationErrorInvalidInRootPath);
  } else if (ValidPath == -2) {
    ShowStatus(MigrationErrorNoFileTypeAvail);
    c2sExit(MigrationErrorNoFileTypeAvail);
  }

  int SDKIncPathRes =
      checkSDKPathOrIncludePath(CudaIncludePath, RealSDKIncludePath);
  if (SDKIncPathRes == -1) {
    ShowStatus(MigrationErrorInvalidCudaIncludePath);
    c2sExit(MigrationErrorInvalidCudaIncludePath);
  } else if (SDKIncPathRes == 0) {
    HasSDKIncludeOption = true;
  }

  int SDKPathRes = checkSDKPathOrIncludePath(SDKPath, RealSDKPath);
  if (SDKPathRes == -1) {
    ShowStatus(MigrationErrorInvalidCudaIncludePath);
    c2sExit(MigrationErrorInvalidCudaIncludePath);
  } else if (SDKPathRes == 0) {
    HasSDKPathOption = true;
  }

  bool GenReport = false;
#ifdef C2S_DEBUG_BUILD
  std::string &DVerbose = DiagsContent;
#else
  std::string DVerbose = "";
#endif
  if (checkReportArgs(ReportType.getValue(), ReportFormat.getValue(),
                      ReportFilePrefix, ReportOnlyFlag, GenReport,
                      DVerbose) == false) {
    ShowStatus(MigrationErrorInvalidReportArgs);
    c2sExit(MigrationErrorInvalidReportArgs);
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

  // TODO: implement one of this for each source language.
  CudaPath = getCudaInstallPath(OriginalArgc, argv);
  C2SDiags() << "Cuda Include Path found: " << CudaPath << "\n";

  RefactoringTool Tool(OptParser->getCompilations(),
                       OptParser->getSourcePathList());

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
  C2SInstallPath = getInstallPath(Tool, argv[0]);

  ValidateInputDirectory(Tool, InRoot);

  IsUsingDefaultOutRoot = OutRoot.empty();
  if (!makeOutRootCanonicalOrSetDefaults(OutRoot)) {
    ShowStatus(MigrationErrorInvalidInRootOrOutRoot);
    c2sExit(MigrationErrorInvalidInRootOrOutRoot, false);
  }
  c2s::C2SGlobalInfo::setOutRoot(OutRoot);

  validateCustomHelperFileNameArg(UseCustomHelperFileLevel,
                                  CustomHelperFileName,
                                  c2s::C2SGlobalInfo::getOutRoot());

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
      (SDKVersionMajor == 11 && SDKVersionMinor == 6)) {
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

  C2SGlobalInfo::setInRoot(InRoot);
  C2SGlobalInfo::setOutRoot(OutRoot);
  C2SGlobalInfo::setCudaPath(CudaPath);
  C2SGlobalInfo::setKeepOriginCode(KeepOriginalCodeFlag);
  C2SGlobalInfo::setSyclNamedLambda(SyclNamedLambdaFlag);
  C2SGlobalInfo::setUsmLevel(USMLevel);
  C2SGlobalInfo::setIsIncMigration(!NoIncrementalMigration);
  C2SGlobalInfo::setHelperFilesCustomizationLevel(UseCustomHelperFileLevel);
  C2SGlobalInfo::setCheckUnicodeSecurityFlag(
    CheckUnicodeSecurityFlag);
  C2SGlobalInfo::setCustomHelperFileName(CustomHelperFileName);
  HelperFileNameMap[HelperFileEnum::C2S] =
      C2SGlobalInfo::getCustomHelperFileName() + ".hpp";
  C2SGlobalInfo::setFormatRange(FormatRng);
  C2SGlobalInfo::setFormatStyle(FormatST);
  C2SGlobalInfo::setCtadEnabled(EnableCTAD);
  C2SGlobalInfo::setGenBuildScriptEnabled(GenBuildScript);
  C2SGlobalInfo::setCommentsEnabled(EnableComments);
  C2SGlobalInfo::setUsingDRYPattern(!NoDRYPatternFlag);
  C2SGlobalInfo::setUsingGenericSpace(!NoUseGenericSpaceFlag);
  C2SGlobalInfo::setExperimentalFlag(Experimentals.getBits());
  C2SGlobalInfo::setAssumedNDRangeDim(
      (NDRangeDim == AssumedNDRangeDimEnum::ARE_Dim1) ? 1 : 3);
  C2SGlobalInfo::setOptimizeMigrationFlag(OptimizeMigration.getValue());
  StopOnParseErrTooling = StopOnParseErr;
  InRootTooling = InRoot;

  if (ExcludePathList.getNumOccurrences()) {
    C2SGlobalInfo::setExcludePath(ExcludePathList);
  }

  std::vector<ExplicitNamespace> DefaultExplicitNamespaces = {
      ExplicitNamespace::EN_SYCL, ExplicitNamespace::EN_C2S};
  if (NoClNamespaceInline.getNumOccurrences()) {
    if (UseExplicitNamespace.getNumOccurrences()) {
      C2SGlobalInfo::setExplicitNamespace(UseExplicitNamespace);
      clang::c2s::PrintMsg(
          "Note: Option --no-cl-namespace-inline is deprecated and will be "
          "ignored. Option --use-explicit-namespace is used instead.\n");
    } else {
      if (ExplicitClNamespace) {
        C2SGlobalInfo::setExplicitNamespace(std::vector<ExplicitNamespace>{
            ExplicitNamespace::EN_CL, ExplicitNamespace::EN_C2S});
      } else {
        C2SGlobalInfo::setExplicitNamespace(DefaultExplicitNamespaces);
      }
      clang::c2s::PrintMsg(
          "Note: Option --no-cl-namespace-inline is deprecated. Use "
          "--use-explicit-namespace instead.\n");
    }
  } else {
    if (UseExplicitNamespace.getNumOccurrences()) {
      C2SGlobalInfo::setExplicitNamespace(UseExplicitNamespace);
    } else {
      C2SGlobalInfo::setExplicitNamespace(DefaultExplicitNamespaces);
    }
  }

  MapNames::setExplicitNamespaceMap();
  CallExprRewriterFactoryBase::initRewriterMap();
  if (!RuleFile.empty()) {
    importRules(RuleFile);
  }

  {
    setValueToOptMap(clang::c2s::OPTION_AsyncHandler, AsyncHandlerFlag,
                     AsyncHandler.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_NDRangeDim,
                     static_cast<unsigned int>(NDRangeDim.getValue()),
                     NDRangeDim.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_CommentsEnabled,
                     C2SGlobalInfo::isCommentsEnabled(),
                     EnableComments.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_CustomHelperFileName,
                     C2SGlobalInfo::getCustomHelperFileName(),
                     CustomHelperFileName.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_CtadEnabled,
                     C2SGlobalInfo::isCtadEnabled(),
                     EnableCTAD.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_ExplicitClNamespace,
                     ExplicitClNamespace,
                     NoClNamespaceInline.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_ExtensionFlag,
                     C2SGlobalInfo::getExtensionFlag(),
                     NoDPCPPExtensions.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_NoDRYPattern, NoDRYPatternFlag,
                     NoDRYPattern.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_NoUseGenericSpace, NoUseGenericSpaceFlag,
                     NoUseGenericSpace.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_CompilationsDir, CompilationsDir,
                     OptParser->isPSpecified());
#ifdef _WIN32
    if (!VcxprojFilePath.empty()) {
      // To convert the relative path to absolute path.
      llvm::SmallString<128> AbsPath(VcxprojFilePath);
      llvm::sys::fs::make_absolute(AbsPath);
      llvm::sys::path::remove_dots(AbsPath, /*remove_dot_dot=*/true);
      setValueToOptMap(clang::c2s::OPTION_VcxprojFile, AbsPath.str().str(),
                       OptParser->isVcxprojfileSpecified());
    } else {
      setValueToOptMap(clang::c2s::OPTION_VcxprojFile, VcxprojFilePath,
                       OptParser->isVcxprojfileSpecified());
    }
#endif
    setValueToOptMap(clang::c2s::OPTION_ProcessAll, ProcessAllFlag,
                     ProcessAll.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_SyclNamedLambda, SyclNamedLambdaFlag,
                     SyclNamedLambda.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_ExperimentalFlag,
                     C2SGlobalInfo::getExperimentalFlag(),
                     Experimentals.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_ExplicitNamespace,
                     C2SGlobalInfo::getExplicitNamespaceSet(),
                     UseExplicitNamespace.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_UsmLevel,
                     static_cast<unsigned int>(C2SGlobalInfo::getUsmLevel()),
                     USMLevel.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_OptimizeMigration,
                     OptimizeMigration.getValue(),
                     OptimizeMigration.getNumOccurrences());
    setValueToOptMap(clang::c2s::OPTION_RuleFile, MetaRuleObject::RuleFiles,
                     RuleFile.getNumOccurrences());

    if (clang::c2s::C2SGlobalInfo::isIncMigration()) {
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
    // specified to get the migration status report, c2s namespace should be
    // enabled temporarily to get LOC migrated to helper functions in function
    // getLOCStaticFromCodeRepls() if it is not enabled.
    auto NamespaceSet = C2SGlobalInfo::getExplicitNamespaceSet();
    if (!NamespaceSet.count(ExplicitNamespace::EN_C2S) &&
        !NamespaceSet.count(ExplicitNamespace::EN_DPCT)) {
      std::vector<ExplicitNamespace> ENVec;
      ENVec.push_back(ExplicitNamespace::EN_C2S);
      C2SGlobalInfo::setExplicitNamespace(ENVec);
      C2SGlobalInfo::setC2SNamespaceTempEnabled();
    }
  }

  if (C2SGlobalInfo::getFormatRange() != clang::format::FormatRange::none) {
    parseFormatStyle();
  }

  auto &Global = C2SGlobalInfo::getInstance();
  int RunCount = 0;
  do {
    if (RunCount == 1) {
      // Currently, we just need maximum two parse
      C2SGlobalInfo::setNeedRunAgain(false);
      C2SGlobalInfo::getInstance().resetInfo();
      DeviceFunctionDecl::reset();
    }
    C2SGlobalInfo::setRunRound(RunCount++);
    C2SActionFactory Factory(Tool.getReplacements());

    if (ProcessAllFlag) {
      clang::tooling::SetFileProcessHandle(InRoot, OutRoot, processAllFiles);
    }

    int RunResult = Tool.run(&Factory);
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
        ShowStatus(MigrationErrorFileParseError);
        return MigrationErrorFileParseError;
      } else {
        // When RunResult equals to 2, it means no error but some files are
        // skipped due to missing compile commands.
        // And clang::tooling::ReFactoryTool will emit error message.
        return MigrationSKIPForMissingCompileCommand;
      }
    }

    int RetJmp = 0;
    CHECKPOINT_ReplacementPostProcess_ENTRY(RetJmp);
    if (RetJmp == 0) {
      try {
        Global.buildReplacements();
        Global.postProcess();
        Global.emplaceReplacements(Tool.getReplacements());
      } catch (std::exception &e) {
        std::string FaultMsg =
            "Error: c2s internal error. c2s tries to recover and write the migration result.\n";
        llvm::errs() << FaultMsg;
      }
    }

    CHECKPOINT_ReplacementPostProcess_EXIT();
  } while (C2SGlobalInfo::isNeedRunAgain());

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
  int Status = runC2S(argc, argv);
  if (IsUsingDefaultOutRoot) {
    removeDefaultOutRootFolder(OutRoot);
  }
  return Status;
}
