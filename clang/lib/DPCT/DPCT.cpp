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
#include "CallExprRewriter.h"
#include "TypeLocRewriters.h"
#include "Checkpoint.h"
#include "Config.h"
#include "CustomHelperFiles.h"
#include "ExternalReplacement.h"
#include "GAnalytics.h"
#include "GenMakefile.h"
#include "IncrementalMigrationUtility.h"
#include "MisleadingBidirectional.h"
#include "Rules.h"
#include "QueryApiMapping.h"
#include "SaveNewFiles.h"
#include "SignalProcess.h"
#include "Statics.h"
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
    "  To get help on the tool usage, run: dpct --help\n"
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
                "If input source files are provided, the directory\n"
                "of the first input source file is used."),
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
    CudaIncludePath("cuda-include-path",
                   desc("The directory path of the CUDA header files."),
                   value_desc("dir"), cat(DPCTCat), llvm::cl::Optional);

static opt<ReportTypeEnum> ReportType(
    "report-type", desc("Specifies the type of report. Values are:\n"),
    llvm::cl::values(
        llvm::cl::OptionEnumValue{"apis", int(ReportTypeEnum::RTE_APIs),
            "Information about API signatures that need migration and the number of times\n"
            "they were encountered. The report file name will have .apis suffix added.", false},
        llvm::cl::OptionEnumValue{"stats", int(ReportTypeEnum::RTE_Stats),
                  "High level migration statistics: Lines Of Code (LOC) that are migrated to\n"
                  "SYCL, LOC migrated to SYCL with helper functions, LOC not needing migration,\n"
                  "LOC needing migration but are not migrated. The report file name has the .stats\n"
                  "suffix added (default)", false},
        llvm::cl::OptionEnumValue{"all", int(ReportTypeEnum::RTE_All),
                  "All of the reports.", false}
        #ifdef DPCT_DEBUG_BUILD
        , llvm::cl::OptionEnumValue{"diags", int(ReportTypeEnum::RTE_Diags),
                  "diags information", true}
        #endif
        ),
    llvm::cl::init(ReportTypeEnum::RTE_NotSetType), value_desc("value"), cat(DPCTCat),
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
    llvm::cl::init(ReportFormatEnum::RFE_NotSetFormat), value_desc("value"), cat(DPCTCat),
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
               llvm::cl::desc("Only reports are generated. No SYCL code is "
                              "generated. Default: off."),
               cat(DPCTCat), llvm::cl::location(ReportOnlyFlag));

bool KeepOriginalCodeFlag = false;

static opt<bool, true>
    ShowOrigCode("keep-original-code",
                 llvm::cl::desc("Keeps the original code in comments of "
                                "generated SYCL files. Default: off.\n"),
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

bool CheckUnicodeSecurityFlag = false;
static opt<bool, true> CheckUnicodeSecurity(
    "check-unicode-security",
    llvm::cl::desc("Enable detection and warnings about Unicode constructs that can be exploited by using\n"
                   "bi-directional formatting codes and homoglyphs in identifiers. Default: off.\n"),
    cat(DPCTCat), llvm::cl::location(CheckUnicodeSecurityFlag));

bool SyclNamedLambdaFlag = false;
static opt<bool, true>
    SyclNamedLambda("sycl-named-lambda",
                llvm::cl::desc("Generates kernels with the kernel name. Default: off.\n"),
                cat(DPCTCat), llvm::cl::location(SyclNamedLambdaFlag));

opt<OutputVerbosityLevel> OutputVerbosity(
    "output-verbosity", llvm::cl::desc("Sets the output verbosity level:"),
    llvm::cl::values(
        llvm::cl::OptionEnumValue{"silent", int(OutputVerbosityLevel::OVL_Silent),
                                  "Only messages from clang.", false},
        llvm::cl::OptionEnumValue{"normal", int(OutputVerbosityLevel::OVL_Normal),
                                  "\'silent\' and warnings, errors, and notes from dpct.",
                                  false},
        llvm::cl::OptionEnumValue{"detailed", int(OutputVerbosityLevel::OVL_Detailed),
                                  "\'normal\' and messages about which file is being processed.",
                                  false},
        llvm::cl::OptionEnumValue{"diagnostics", int(OutputVerbosityLevel::OVL_Diagnostics),
                                  "\'detailed\' and information about the detected "
                                  "conflicts and crashes. (default)", false}),
    llvm::cl::init(OutputVerbosityLevel::OVL_Diagnostics), value_desc("value"), cat(DPCTCat),
    llvm::cl::Optional);

opt<std::string>
    OutputFile("output-file",
               desc("Redirects the stdout/stderr output to <file> in the output"
                    " directory specified\n"
                    "by the --out-root option."),
               value_desc("file"), cat(DPCTCat),
               llvm::cl::Optional);

list<std::string> RuleFile("rule-file", desc("Specifies the rule file path that contains rules used for migration.\n"),
                           value_desc("file"), cat(DPCTCat), llvm::cl::ZeroOrMore);

opt<UsmLevel> USMLevel(
    "usm-level", desc("Sets the Unified Shared Memory (USM) level to use in source code generation.\n"),
    values(llvm::cl::OptionEnumValue{"restricted", int(UsmLevel::UL_Restricted),
                     "Uses USM API for memory management migration. (default)", false},
           llvm::cl::OptionEnumValue{"none", int(UsmLevel::UL_None),
                     "Uses helper functions from DPCT header files for memory "
                     "management migration.", false}),
    init(UsmLevel::UL_Restricted), value_desc("value"), cat(DPCTCat), llvm::cl::Optional);

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
                values(llvm::cl::OptionEnumValue{"llvm", int(DPCTFormatStyle::FS_LLVM),
                     "Use the LLVM coding style.", false},
                       llvm::cl::OptionEnumValue{"google", int(DPCTFormatStyle::FS_Google),
                     "Use the Google coding style.", false},
                       llvm::cl::OptionEnumValue{"custom", int(DPCTFormatStyle::FS_Custom),
                     "Use the coding style defined in the .clang-format file (default).", false}),
    init(DPCTFormatStyle::FS_Custom), value_desc("value"), cat(DPCTCat), llvm::cl::Optional);

bool ExplicitClNamespace = false;
static opt<bool, true> NoClNamespaceInline(
    "no-cl-namespace-inline",
    llvm::cl::desc("DEPRECATED: Do not use cl:: namespace inline. Default: off. This option will be\n"
                   "ignored if the replacement option --use-explicit-namespace is used.\n"),
    cat(DPCTCat), llvm::cl::location(ExplicitClNamespace));

bool NoDRYPatternFlag = false;
static opt<bool, true> NoDRYPattern(
  "no-dry-pattern", llvm::cl::desc("Do not use DRY (do not repeat yourself) pattern when functions from dpct\n"
                                   "namespace are inserted. Default: off.\n"),
  cat(DPCTCat), llvm::cl::location(NoDRYPatternFlag));

bool NoUseGenericSpaceFlag = false;
static opt<bool, true> NoUseGenericSpace(
  "no-use-generic-space", llvm::cl::desc("sycl::access::address_space::generic_space is not used during atomic\n"
                                         " function's migration. Default: off.\n"),
  cat(DPCTCat), llvm::cl::location(NoUseGenericSpaceFlag), llvm::cl::ReallyHidden);

bool ProcessAllFlag = false;
static opt<bool, true>
    ProcessAll("process-all",
                 llvm::cl::desc("Migrates or copies all files, except hidden, from the --in-root directory\n"
                 "to the --out-root directory. The --in-root option should be explicitly specified.\n"
                 "Default: off."),
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
    cat(DPCTCat), llvm::cl::Optional);

opt<std::string> CustomHelperFileName(
    "custom-helper-name",
    desc(
        "Specifies the helper headers folder name and main helper header file name.\n"
        "Default: dpct."),
    init("dpct"), value_desc("name"), cat(DPCTCat), llvm::cl::Optional);

bool AsyncHandlerFlag = false;
static opt<bool, true> AsyncHandler(
    "always-use-async-handler",
    llvm::cl::desc("Use async exception handler when creating new sycl::queue "
                   "with dpct::create_queue\nin addition to default "
                   "dpct::get_default_queue. Default: off."),
    cat(DPCTCat), llvm::cl::location(AsyncHandlerFlag));

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
    init(AssumedNDRangeDimEnum::ARE_Dim3), value_desc("value"), cat(DPCTCat),
    llvm::cl::Optional);

static list<ExplicitNamespace> UseExplicitNamespace(
    "use-explicit-namespace",
    llvm::cl::desc(
        "Defines the namespaces to use explicitly in generated code. The value is a comma\n"
        "separated list. Default: dpct, sycl.\n"
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
                                     "Generate code with dpct:: namespace.",
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
    value_desc("value"), cat(DPCTCat), llvm::cl::ZeroOrMore);

// When more dpcpp extensions are implemented, more extension names will be
// added into the value of option --no-dpcpp-extensions, currently only
// Enqueued barriers is supported.
static list<DPCPPExtensions> NoDPCPPExtensions(
    "no-dpcpp-extensions",
    llvm::cl::desc(
        "Comma separated list of extensions not to be used in migrated "
        "code.\n"
        "By default, these extensions will be used in migrated code."),
    llvm::cl::CommaSeparated,
    values(llvm::cl::OptionEnumValue{"enqueued_barriers",
                                     int(DPCPPExtensions::Ext_EnqueueBarrier),
                                     "Enqueued barriers extension.",
                                     false}),
    value_desc("value"), cat(DPCTCat), llvm::cl::ZeroOrMore,
    llvm::cl::cb<void, DPCPPExtensions>(DpctGlobalInfo::setExtensionUnused));

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
        false },
    llvm::cl::OptionEnumValue{
        "logical-group", int(ExperimentalFeatures::Exp_LogicalGroup),
        "Experimental helper function used to group some work-items logically.",
        false }),
  value_desc("value"), cat(DPCTCat), llvm::cl::ZeroOrMore);

opt<bool> GenBuildScript(
    "gen-build-script",
    llvm::cl::desc("Generates makefile for migrated file(s) in -out-root directory. Default: off."),
    cat(DPCTCat), init(false));

opt<std::string>
    BuildScriptFile("build-script-file",
               desc("Specifies the name of generated makefile for migrated file(s).\n"
                    "Default name: Makefile.dpct."),
               value_desc("file"), cat(DPCTCat),
               llvm::cl::Optional);

static list<std::string> ExcludePathList(
    "in-root-exclude",
    llvm::cl::desc(
        "Excludes the specified directory or file from processing."),
    value_desc("dir|file"), cat(DPCTCat), llvm::cl::ZeroOrMore);

static opt<bool> OptimizeMigration(
    "optimize-migration",
    llvm::cl::desc("Generates SYCL code applying more aggressive assumptions that potentially\n"
                   "may alter the semantics of your program. Default: off."),
    cat(DPCTCat), init(false));

static opt<bool> NoIncrementalMigration(
    "no-incremental-migration",
    llvm::cl::desc("Tells the tool to not perform an incremental migration.\n"
                   "Default: off (incremental migration happens)."),
    cat(DPCTCat), init(false));

static opt<std::string> QueryApiMapping("query-api-mapping",
    llvm::cl::desc("Query mapped SYCL API from CUDA API."),
    value_desc("api"), cat(DPCTCat), llvm::cl::Optional);
// clang-format on

// TODO: implement one of this for each source language.
std::string CudaPath;
std::string DpctInstallPath;
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

class DPCTConsumer : public ASTConsumer {
public:
  DPCTConsumer(CompilerInstance &CI, StringRef InFile)
      : ATM(CI, InRoot), PP(CI.getPreprocessor()), CI(CI) {
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

    auto &Global = DpctGlobalInfo::getInstance();
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
    DpctGlobalInfo::setCompilerInstance(CI);

    PP.addPPCallbacks(std::make_unique<IncludesCallbacks>(
        TransformSet, IncludeMapSet, Context.getSourceManager(), ATM));

    if (DpctGlobalInfo::getCheckUnicodeSecurityFlag()) {
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

  ~DPCTConsumer() {
    // Clean EmittedTransformations for input file migrated.
    ASTTraversalMetaInfo::getEmittedTransformations().clear();
  }

private:
  ASTTraversalManager ATM;
  TransformSetTy TransformSet;
  IncludeMapSetTy IncludeMapSet;
  StmtStringMap SSM;
  Preprocessor &PP;
  CompilerInstance &CI;
  std::shared_ptr<MisleadingBidirectionalHandler> CommentHandler;
};

class DPCTAction : public ASTFrontendAction {
public:
  DPCTAction() = default;

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<DPCTConsumer>(CI, InFile);
  }

  bool usesPreprocessorOnly() const override { return false; }
};

// Object of this class will be handed to RefactoringTool::run and will create
// the Action.
class DPCTActionFactory : public FrontendActionFactory {
public:
  DPCTActionFactory() = default;
  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<DPCTAction>();
  }
};

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

  if (!CudaIncludePath.empty()) {
    if (!CudaIncludeDetector.isIncludePathValid()) {
      ShowStatus(MigrationErrorInvalidCudaIncludePath);
      dpctExit(MigrationErrorInvalidCudaIncludePath);
    }

    if (!CudaIncludeDetector.isVersionSupported()) {
      ShowStatus(MigrationErrorCudaVersionUnsupported);
      dpctExit(MigrationErrorCudaVersionUnsupported);
    }
  } else if (!CudaIncludeDetector.isSupportedVersionAvailable()) {
    ShowStatus(MigrationErrorSupportedCudaVersionNotAvailable);
    dpctExit(MigrationErrorSupportedCudaVersionNotAvailable);
  }

  makeCanonical(Path);

  SmallString<512> CudaPathAbs;
  std::error_code EC = llvm::sys::fs::real_path(Path, CudaPathAbs);
  if ((bool)EC) {
    ShowStatus(MigrationErrorInvalidCudaIncludePath);
    dpctExit(MigrationErrorInvalidCudaIncludePath);
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
    dpctExit(MigrationErrorInvalidInstallPath);
  }
  return InstallPathAbs.str().str();
}

// To validate the root path of the project to be migrated.
void ValidateInputDirectory(clang::tooling::RefactoringTool &Tool,
                            std::string &InRoot) {

  if (isChildOrSamePath(CudaPath, InRoot)) {
    ShowStatus(MigrationErrorRunFromSDKFolder);
    dpctExit(MigrationErrorRunFromSDKFolder);
  }

  if (isChildOrSamePath(InRoot, CudaPath)) {
    ShowStatus(MigrationErrorInRootContainSDKFolder);
    dpctExit(MigrationErrorInRootContainSDKFolder);
  }

  if (isChildOrSamePath(InRoot, DpctInstallPath)) {
    ShowStatus(MigrationErrorInRootContainCTTool);
    dpctExit(MigrationErrorInRootContainCTTool);
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
  GAnalytics("");
#if defined(__linux__) || defined(_WIN32)
  InstallSignalHandle();
#endif

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

  if (QueryApiMapping.getNumOccurrences()) {
    ApiMappingEntry::initEntryMap();
    ApiMappingEntry::printMappingDesc(llvm::outs(), QueryApiMapping);
    dpctExit(MigrationSucceeded);
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

  // TODO: implement one of this for each source language.
  CudaPath = getCudaInstallPath(OriginalArgc, argv);
  DpctDiags() << "Cuda Include Path found: " << CudaPath << "\n";

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
  DpctInstallPath = getInstallPath(Tool, argv[0]);

  ValidateInputDirectory(Tool, InRoot);

  IsUsingDefaultOutRoot = OutRoot.empty();
  if (!makeOutRootCanonicalOrSetDefaults(OutRoot)) {
    ShowStatus(MigrationErrorInvalidInRootOrOutRoot);
    dpctExit(MigrationErrorInvalidInRootOrOutRoot, false);
  }
  dpct::DpctGlobalInfo::setOutRoot(OutRoot);

  validateCustomHelperFileNameArg(UseCustomHelperFileLevel,
                                  CustomHelperFileName,
                                  dpct::DpctGlobalInfo::getOutRoot());

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
      (SDKVersionMajor == 11 && SDKVersionMinor == 7)) {
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
  DpctGlobalInfo::setCudaPath(CudaPath);
  DpctGlobalInfo::setKeepOriginCode(KeepOriginalCodeFlag);
  DpctGlobalInfo::setSyclNamedLambda(SyclNamedLambdaFlag);
  DpctGlobalInfo::setUsmLevel(USMLevel);
  DpctGlobalInfo::setIsIncMigration(!NoIncrementalMigration);
  DpctGlobalInfo::setHelperFilesCustomizationLevel(UseCustomHelperFileLevel);
  DpctGlobalInfo::setCheckUnicodeSecurityFlag(CheckUnicodeSecurityFlag);
  DpctGlobalInfo::setCustomHelperFileName(CustomHelperFileName);
  HelperFileNameMap[HelperFileEnum::Dpct] =
      DpctGlobalInfo::getCustomHelperFileName() + ".hpp";
  DpctGlobalInfo::setFormatRange(FormatRng);
  DpctGlobalInfo::setFormatStyle(FormatST);
  DpctGlobalInfo::setCtadEnabled(EnableCTAD);
  DpctGlobalInfo::setGenBuildScriptEnabled(GenBuildScript);
  DpctGlobalInfo::setCommentsEnabled(EnableComments);
  DpctGlobalInfo::setUsingDRYPattern(!NoDRYPatternFlag);
  DpctGlobalInfo::setUsingGenericSpace(!NoUseGenericSpaceFlag);
  DpctGlobalInfo::setExperimentalFlag(Experimentals.getBits());
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
  CallExprRewriterFactoryBase::initMethodRewriterMap();
  TypeLocRewriterFactoryBase::initTypeLocRewriterMap();
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
    setValueToOptMap(clang::dpct::OPTION_CustomHelperFileName,
                     DpctGlobalInfo::getCustomHelperFileName(),
                     CustomHelperFileName.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_CtadEnabled,
                     DpctGlobalInfo::isCtadEnabled(),
                     EnableCTAD.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_ExplicitClNamespace,
                     ExplicitClNamespace,
                     NoClNamespaceInline.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_ExtensionFlag,
                     DpctGlobalInfo::getExtensionFlag(),
                     NoDPCPPExtensions.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_NoDRYPattern, NoDRYPatternFlag,
                     NoDRYPattern.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_NoUseGenericSpace,
                     NoUseGenericSpaceFlag,
                     NoUseGenericSpace.getNumOccurrences());
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
    setValueToOptMap(clang::dpct::OPTION_ExplicitNamespace,
                     DpctGlobalInfo::getExplicitNamespaceSet(),
                     UseExplicitNamespace.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_UsmLevel,
                     static_cast<unsigned int>(DpctGlobalInfo::getUsmLevel()),
                     USMLevel.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_OptimizeMigration,
                     OptimizeMigration.getValue(),
                     OptimizeMigration.getNumOccurrences());
    setValueToOptMap(clang::dpct::OPTION_RuleFile, MetaRuleObject::RuleFiles,
                     RuleFile.getNumOccurrences());

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

  auto &Global = DpctGlobalInfo::getInstance();
  volatile int RunCount = 0;
  do {
    if (RunCount == 1) {
      // Currently, we just need maximum two parse
      DpctGlobalInfo::setNeedRunAgain(false);
      DpctGlobalInfo::getInstance().resetInfo();
      DeviceFunctionDecl::reset();
    }
    DpctGlobalInfo::setRunRound(RunCount++);
    DPCTActionFactory Factory;

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
        // When RunResult equals to 2, it means no error, but some files are
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
        std::string FaultMsg = "Error: dpct internal error. dpct tries to "
                               "recover and write the migration result.\n";
        llvm::errs() << FaultMsg;
      }
    }

    CHECKPOINT_ReplacementPostProcess_EXIT();
  } while (DpctGlobalInfo::isNeedRunAgain());

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
