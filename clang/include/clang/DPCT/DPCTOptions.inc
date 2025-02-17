//===--------------- DpctOptions.inc--------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/****************************************************************************
*
* DPCT_OPTION(TEMPLATE, TYPE, NAME, OPTION_CLASS, OPTION_ACTIONS, OPTION_ARG_NAME, ...)
*  TEMPLATE: clang::dpct::opt | clang::dpct::list | clang::dpct::bits
*  TYPE: Option data type
*  NAME: Option enum name and variable name
*  OPTION_CLASS: clang::dpct::DpctOptionClass::OC_Attribute 
*    | clang::dpct::DpctOptionClass::OC_Action
*  OPTION_ACTIONS: DPCT_OPTION_ACTIONS([DpctActionKind, ...])
*  OPTION_ARG_NAME: Option argument name
*
*  ...: Other option attributes, e.g. llvm::cl::desc(...), llvm::cl::value_desc(...)
*
****************************************************************************/

#ifdef DPCT_OPTIONS_VAR

#if DPCT_OPTIONS_IN_CLANG_DPCT

#define DPCT_OPTION(TEMPLATE, TYPE, NAME, ...)                                 \
  clang::dpct::DpctOption<TEMPLATE, TYPE, llvm::cl::parser<TYPE>> NAME(        \
      clang::dpct::DpctOptionNameKind::OPT_##NAME, __VA_ARGS__,                \
      llvm::cl::cat(CtHelpCatAll));
#define DPCT_OPTION_ACTIONS(...)                                               \
  { __VA_ARGS__ }

#elif DPCT_OPTIONS_IN_CLANG_TOOLING

#define DPCT_OPTION(TEMPLATE, TYPE, NAME, OPTION_CLASS, OPTION_ACTIONS, ...)   \
  static TEMPLATE<TYPE, llvm::cl::parser<TYPE>> NAME(                          \
      __VA_ARGS__, llvm::cl::sub(llvm::cl::SubCommand::getAll()),              \
      llvm::cl::cat(Category));
#define DPCT_OPTION_ACTIONS(...) 0

#else

#define DPCT_OPTION(...)

#endif

#define DPCT_ALIASE(ALIASE_FOR, ...)                                           \
  static llvm::cl::alias ALIASE_FOR##Aliase(__VA_ARGS__,                       \
                                            llvm::cl::aliasopt(ALIASE_FOR));

#define DPCT_OPTION_ENUM_VALUE(...)                                            \
  llvm::cl::OptionEnumValue { __VA_ARGS__ }
#define DPCT_OPTION_VALUES(...) llvm::cl::values(__VA_ARGS__)

#else

#ifndef DPCT_OPTIONS_IN_CLANG_TOOLING
#define DPCT_OPTIONS_IN_CLANG_TOOLING 1
#endif

#ifndef DPCT_OPTIONS_IN_CLANG_DPCT
#define DPCT_OPTIONS_IN_CLANG_DPCT 1
#endif

#ifndef DPCT_OPTION
#define DPCT_OPTION(...)
#endif

#ifndef DPCT_ALIASE
#define DPCT_ALIASE(...)
#endif

#endif


#ifndef DPCT_SOURCEPATH_OPTION
#define DPCT_SOURCEPATH_OPTION(...)                                            \
  DPCT_OPTION(clang::dpct::list, std::string, __VA_ARGS__)
#endif

#ifndef DPCT_PATH_OPTION
#define DPCT_PATH_OPTION(...)                                                  \
  DPCT_OPTION(clang::dpct::opt, std::string, __VA_ARGS__)
#endif

#ifndef DPCT_ENUM_OPTION
#define DPCT_ENUM_OPTION(TEMPLATE, ENUM_TYPE, ...)                             \
  DPCT_OPTION(TEMPLATE, clang::dpct::ENUM_TYPE, __VA_ARGS__)
#endif

#ifndef DPCT_FLAG_OPTION
#define DPCT_FLAG_OPTION(...)                                                  \
  DPCT_OPTION(clang::dpct::opt, bool, __VA_ARGS__, llvm::cl::init(false))
#endif

#ifndef DPCT_HIDDEN_OPTION
#define DPCT_HIDDEN_OPTION(...) DPCT_OPTION(__VA_ARGS__, llvm::cl::Hidden)
#endif

#ifndef DPCT_OPTION_ACTIONS
#define DPCT_OPTION_ACTIONS(...)
#endif

#ifndef DPCT_OPTION_ENUM_VALUE
#define DPCT_OPTION_ENUM_VALUE(...)
#endif

#ifndef DPCT_OPTION_VALUES
#define DPCT_OPTION_VALUES(...)
#endif

#if DPCT_OPTIONS_IN_CLANG_TOOLING
DPCT_PATH_OPTION(
    BuildPath, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                        clang::dpct::DpctActionKind::DAK_Migration),
    "compilation-database",
    llvm::cl::desc("The directory path for the compilation database "
                   "(compile_commands.json) for the files to\n"
                   "be migrated. Paths and build options described in the "
                   "project file are used to guide the\n"
                   "migration.\n"
                   "When no path is specified, a search for "
                   "compile_commands.json is attempted through all\n"
                   "parent directories of the first input source file.\n"
                   "Same as -p."),
    llvm::cl::value_desc("dir"), llvm::cl::Optional)

DPCT_ALIASE(BuildPath, "p", llvm::cl::desc("Alias for --compilation-database."),
            llvm::cl::NotHidden)

DPCT_OPTION(clang::dpct::list, std::string, ArgsAfter,
            clang::dpct::DpctOptionClass::OC_Attribute,
            DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                                clang::dpct::DpctActionKind::DAK_Migration,
                                clang::dpct::DpctActionKind::DAK_Query),
            "extra-arg",
            llvm::cl::desc("Additional argument to append to the "
                           "migration command line, example:\n"
                           "--extra-arg=\"-I /path/to/header\". The "
                           "options that can be passed this way can\n"
                           "be found with the dpct -- -help command."),
            llvm::cl::value_desc("string"), llvm::cl::CommaSeparated)

#ifdef _WIN32
DPCT_PATH_OPTION(
    VcxprojFile, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                        clang::dpct::DpctActionKind::DAK_Migration),
    "vcxprojfile",
    llvm::cl::desc("The directory path for the C++ project file (.vcxproj) of "
                   "the Visual Studio project to\n"
                   "migrate. Paths and build options described in the project "
                   "file are used to guide the\n"
                   "migration."),
    llvm::cl::value_desc("file"), llvm::cl::Optional)
#endif

#define DPCT_OPTION_SOURCE_PATH_DESC llvm::cl::desc("[<source0> ... <sourceN>]")
DPCT_SOURCEPATH_OPTION(
    SourcePaths, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_BuildScript,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    llvm::cl::Positional, DPCT_OPTION_SOURCE_PATH_DESC, llvm::cl::ZeroOrMore)
#undef DPCT_SOURCE_PATH_DESC

DPCT_HIDDEN_OPTION(
    clang::dpct::list, std::string, ArgsBefore,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Query,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "extra-arg-before",
    cl::desc("Additional argument to prepend to the compiler command line.\n"
             "Refer to extra-arg option.\n"))

#endif // !DPCT_OPTIONS_IN_CLANG_TOOLING

#if DPCT_OPTIONS_IN_CLANG_DPCT
DPCT_PATH_OPTION(
    InRoot, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                        clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_BuildScript),
    "in-root",
    llvm::cl::desc(
        "The directory path for the root of the source tree that needs "
        "to be migrated.\n"
        "Only files under this root are migrated. Default: Current"
        " directory, if input\nsource files are not provided. "
        "If input source files are provided, the directory\n"
        "of the first input source file is used."),
    llvm::cl::value_desc("dir"), llvm::cl::cat(CtHelpCatBasic),
    llvm::cl::Optional)

DPCT_PATH_OPTION(
    OutRoot, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_BuildScript),
    "out-root",
    llvm::cl::desc(
        "The directory path for root of generated files. A directory is "
        "created if it\n"
        "does not exist. Default: dpct_output."),
    llvm::cl::value_desc("dir"), llvm::cl::cat(CtHelpCatBasic),
    llvm::cl::Optional)

DPCT_PATH_OPTION(
    CudaInclude, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                        clang::dpct::DpctActionKind::DAK_Query,
                        clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_BuildScript),
    "cuda-include-path",
    llvm::cl::desc("The directory path of the CUDA header files."),
    llvm::cl::value_desc("dir"), cat(CtHelpCatAll), cat(CtHelpCatBasic),
    llvm::cl::Optional)

DPCT_PATH_OPTION(
    ReportFilePrefix, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "report-file-prefix",
    llvm::cl::desc(
        "Specify the prefix for the migration report file names. The full file "
        "name will "
        "have a suffix derived\n"
        "from the report-type, and an extension derived from the "
        "report-format. "
        "For\n"
        "example: <prefix>.apis.csv or <prefix>.stats.log. If this option is "
        "not\n"
        "specified, the report will go to stdout. The report files are created "
        "in the\n"
        "directory, specified by -out-root."),
    llvm::cl::value_desc("prefix"), llvm::cl::cat(CtHelpCatReportGen),
    llvm::cl::Optional)

DPCT_FLAG_OPTION(
    ReportOnly, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "report-only",
    llvm::cl::desc("Generate migration reports only. No SYCL "
                   "code will be generated. Default: off."),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatReportGen))

DPCT_FLAG_OPTION(
    KeepOriginalCode, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "keep-original-code",
    llvm::cl::desc("Keep the original code in comments of "
                   "generated SYCL files. Default: off.\n"),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen))

#ifdef DPCT_DEBUG_BUILD
DPCT_ENUM_OPTION(
    clang::dpct::opt, ReportTypeEnum, ReportType,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "report-type",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE("all", int(ReportTypeEnum::RTE_All),
                               "All of the migration reports.", false),
        DPCT_OPTION_ENUM_VALUE(
            "apis", int(ReportTypeEnum::RTE_APIs),
            "Information about API signatures that need migration "
            "and the number of times\n"
            "they were encountered. The report file name will have "
            ".apis suffix added.",
            false),
        DPCT_OPTION_ENUM_VALUE("diags", int(ReportTypeEnum::RTE_Diags),
                               "diags information", true),
        DPCT_OPTION_ENUM_VALUE(
            "stats", int(ReportTypeEnum::RTE_Stats),
            "High level migration statistics: Lines Of Code (LOC) "
            "that are migrated to\n"
            "SYCL, LOC migrated to SYCL with helper functions, LOC "
            "not needing migration,\n"
            "LOC needing migration but are not migrated. The report "
            "file name has the .stats\n"
            "suffix added (default)",
            false)),
    llvm::cl::desc("Specify the type of migration report. Values are:\n"),
    llvm::cl::init(ReportTypeEnum::RTE_NotSetType),
    llvm::cl::value_desc("value"), llvm::cl::cat(CtHelpCatReportGen),
    llvm::cl::Optional)

#else
DPCT_ENUM_OPTION(
    clang::dpct::opt, ReportTypeEnum, ReportType,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "report-type",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE("all", int(ReportTypeEnum::RTE_All),
                               "All of the migration reports.", false),
        DPCT_OPTION_ENUM_VALUE(
            "apis", int(ReportTypeEnum::RTE_APIs),
            "Information about API signatures that need migration "
            "and the number of times\n"
            "they were encountered. The report file name will have "
            ".apis suffix added.",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "stats", int(ReportTypeEnum::RTE_Stats),
            "High level migration statistics: Lines Of Code (LOC) "
            "that are migrated to\n"
            "SYCL, LOC migrated to SYCL with helper functions, LOC "
            "not needing migration,\n"
            "LOC needing migration but are not migrated. The report "
            "file name has the .stats\n"
            "suffix added (default)",
            false)),
    llvm::cl::desc("Specify the type of migration report. Values are:\n"),
    llvm::cl::init(ReportTypeEnum::RTE_NotSetType),
    llvm::cl::value_desc("value"), llvm::cl::cat(CtHelpCatReportGen),
    llvm::cl::Optional)
#endif

DPCT_ENUM_OPTION(
    clang::dpct::opt, ReportFormatEnum, ReportFormat,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "report-format",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE(
            "csv", int(ReportFormatEnum::RFE_CSV),
            "Output will be lines of comma-separated values. The report file "
            "name extension will\n"
            "be .csv. (default)",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "formatted", int(ReportFormatEnum::RFE_Formatted),
            "Output will be formatted for easier readability. Report "
            "file name\n"
            "extension will be log.",
            false)),
    llvm::cl::desc("Specify the format of the migration reports:\n"),
    llvm::cl::init(ReportFormatEnum::RFE_NotSetFormat),
    llvm::cl::value_desc("value"), llvm::cl::cat(CtHelpCatReportGen),
    llvm::cl::Optional)

DPCT_OPTION(clang::dpct::list, std::string, SuppressWarnings,
            clang::dpct::DpctOptionClass::OC_Attribute,
            DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
            "suppress-warnings", llvm::cl::desc(SuppressWarningsMessage),
            llvm::cl::value_desc("value"), llvm::cl::CommaSeparated,
            llvm::cl::cat(CtHelpCatWarnings))

DPCT_FLAG_OPTION(
    SuppressWarningsAll, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "suppress-warnings-all",
    llvm::cl::desc("Suppress all migration warnings. Default: off."),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatWarnings))

DPCT_FLAG_OPTION(StopOnParseErr, clang::dpct::DpctOptionClass::OC_Attribute,
                 DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                                     clang::dpct::DpctActionKind::DAK_Analysis),
                 "stop-on-parse-err",
                 llvm::cl::desc("Stop migration and generation of reports if "
                                "parsing errors happened. Default: off. \n"),
                 llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatBasic))

DPCT_FLAG_OPTION(
    CheckUnicodeSecurity, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "check-unicode-security",
    llvm::cl::desc("Enable detection and warnings about Unicode constructs "
                   "that can be exploited by using\n"
                   "bi-directional formatting codes and homoglyphs in "
                   "identifiers. Default: off.\n"),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen))

DPCT_FLAG_OPTION(
    EnablepProfiling, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "enable-profiling",
    llvm::cl::desc(
        "Enable SYCL queue profiling in helper functions. default: auto (when "
        "set to\nauto, the enable-profiling option will only be used if the "
        "tool deduces that profiling is required during migration).\n"),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen))

DPCT_FLAG_OPTION(
    SyclNamedLambda, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "sycl-named-lambda",
    llvm::cl::desc("Generate kernels with the kernel name. Default: off.\n"),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen))

DPCT_ENUM_OPTION(
    clang::dpct::opt, OutputVerbosityLevel, OutputVerbosity,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "output-verbosity",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE(
            "detailed", int(OutputVerbosityLevel::OVL_Detailed),
            "'normal' and messages about which file is being processed.",
            false),
        DPCT_OPTION_ENUM_VALUE("diagnostics",
                               int(OutputVerbosityLevel::OVL_Diagnostics),
                               "'detailed' and information about the detected "
                               "conflicts and crashes. (default)",
                               false),
        DPCT_OPTION_ENUM_VALUE(
            "normal", int(OutputVerbosityLevel::OVL_Normal),
            "'silent' and warnings, errors, and notes from dpct.", false),
        DPCT_OPTION_ENUM_VALUE("silent", int(OutputVerbosityLevel::OVL_Silent),
                               "Only messages from clang.", false)),
    llvm::cl::desc("Set the output verbosity level:"),
    llvm::cl::init(OutputVerbosityLevel::OVL_Diagnostics),
    llvm::cl::value_desc("value"), llvm::cl::cat(CtHelpCatWarnings),
    llvm::cl::Optional)

DPCT_PATH_OPTION(
    OutputFile, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "output-file",
    llvm::cl::desc("Redirect the stdout/stderr output to <file> in the output"
                   " directory specified\n"
                   "by the --out-root option."),
    llvm::cl::value_desc("file"), llvm::cl::cat(CtHelpCatWarnings),
    llvm::cl::Optional)

DPCT_OPTION(clang::dpct::list, std::string, RuleFile,
            clang::dpct::DpctOptionClass::OC_Attribute,
            DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                                clang::dpct::DpctActionKind::DAK_BuildScript,
                                clang::dpct::DpctActionKind::DAK_Analysis),
            "rule-file",
            llvm::cl::desc(
                "Specify the rule file "
                "for migration. Also, reference to the predefined rules in"
                " \"extensions\" directory in the root folder of the tool.\n"),
            llvm::cl::value_desc("file"), llvm::cl::cat(CtHelpCatAdvanced),
            llvm::cl::ZeroOrMore)

DPCT_ENUM_OPTION(
    clang::dpct::opt, UsmLevel, USMLevel,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "usm-level",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE(
            "none", int(UsmLevel::UL_None),
            "Uses helper functions from DPCT header files for memory "
            "management migration.",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "restricted", int(UsmLevel::UL_Restricted),
            "Uses USM API for memory management migration. (default)", false)),
    llvm::cl::desc("Set the Unified Shared Memory (USM) level to use in "
                   "source code generation.\n"),
    llvm::cl::init(UsmLevel::UL_Restricted), llvm::cl::value_desc("value"),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen),
    llvm::cl::Optional)

DPCT_ENUM_OPTION(
    clang::dpct::bits, BuildScriptKind, BuildScript,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_BuildScript,
                        clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "migrate-build-script",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE("CMake", int(BuildScriptKind::BS_CMake),
                               "Migrate the CMake file(s).", false),
        DPCT_OPTION_ENUM_VALUE("Python", int(BuildScriptKind::BS_Python),
                               "Migrate the python build script file(s) of "
                               "PyTorch based project.",
                               false)),
    llvm::cl::desc(
        "EXPERIMENTAL: A comma-separated list of build script type(s) to be "
        "migrated.\n"
        "By default, no build script is migrated.\n"
        "The values are:\n"),
    llvm::cl::CommaSeparated, llvm::cl::value_desc("value"),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatAdvanced),
    llvm::cl::Optional, llvm::cl::ZeroOrMore)

DPCT_ENUM_OPTION(
    clang::dpct::opt, FormatRange, FormatRng,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "format-range",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE("all", int(format::FormatRange::all),
                               "Format all code.", false),
        DPCT_OPTION_ENUM_VALUE("migrated", int(format::FormatRange::migrated),
                               "Only format the migrated code (default).",
                               false),
        DPCT_OPTION_ENUM_VALUE("none", int(format::FormatRange::none),
                               "Do not format any code.", false)),
    llvm::cl::desc("Set the range of formatting.\nThe values are:\n"),
    llvm::cl::init(format::FormatRange::migrated),
    llvm::cl::value_desc("value"), llvm::cl::cat(CtHelpCatAdvanced),
    llvm::cl::Optional)

DPCT_ENUM_OPTION(
    clang::dpct::opt, DPCTFormatStyle, FormatST,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "format-style",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE(
            "custom", int(DPCTFormatStyle::FS_Custom),
            "Use the coding style defined in the .clang-format file (default).",
            false),
        DPCT_OPTION_ENUM_VALUE("llvm", int(DPCTFormatStyle::FS_LLVM),
                               "Use the LLVM coding style.", false),
        DPCT_OPTION_ENUM_VALUE("google", int(DPCTFormatStyle::FS_Google),
                               "Use the Google coding style.", false)),
    llvm::cl::desc("Set the formatting style.\nThe values are:\n"),
    llvm::cl::init(DPCTFormatStyle::FS_Custom), llvm::cl::value_desc("value"),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatAdvanced),
    llvm::cl::Optional)

DPCT_FLAG_OPTION(NoDRYPattern, clang::dpct::DpctOptionClass::OC_Attribute,
                 DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                                     clang::dpct::DpctActionKind::DAK_Analysis),
                 "no-dry-pattern",
                 llvm::cl::desc("Do not use DRY (do not repeat yourself) "
                                "pattern when functions from dpct\n"
                                "namespace are inserted. Default: off.\n"),
                 llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen))

DPCT_FLAG_OPTION(
    ProcessAll, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "process-all",
    llvm::cl::desc("Migrate or copies all files, except hidden, from the "
                   "--in-root directory\n"
                   "to the --out-root directory. The --in-root option should "
                   "be explicitly specified.\n"
                   "Default: off."),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatBasic))

DPCT_FLAG_OPTION(
    EnableCodePin, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "enable-codepin",
    llvm::cl::desc(
        "EXPERIMENTAL: Generate instrumented CUDA and SYCL code for debug and "
        "verification purposes in the directory <dir>_codepin_cuda and "
        "<dir>_codepin_sycl, where <dir> is specified by --out-root option."),
    llvm::cl::cat(CtHelpCatAll))

DPCT_FLAG_OPTION(
    EnableCTAD, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "enable-ctad",
    llvm::cl::desc("Use a C++17 class template argument deduction (CTAD) in "
                   "your generated code.\n"
                   "Default: off."),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen))

DPCT_FLAG_OPTION(
    EnableComments, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration), "comments",
    llvm::cl::desc(
        "Insert comments explaining the generated code. Default: off."),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen))

DPCT_FLAG_OPTION(
    AsyncHandler, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "always-use-async-handler",
    llvm::cl::desc("Use async exception handler when creating new sycl::queue "
                   "with dpct::create_queue\nin addition to default "
                   "dpct::get_default_queue. Default: off."),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen))

DPCT_ENUM_OPTION(
    clang::dpct::opt, AssumedNDRangeDimEnum, NDRangeDim,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "assume-nd-range-dim",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE("1", 1,
                               "Generate kernel code assuming 1D nd_range "
                               "where possible, and 3D in other cases.",
                               false),
        DPCT_OPTION_ENUM_VALUE(
            "3", 3, "Generate kernel code assuming 3D nd_range (default).",
            false)),
    llvm::cl::desc("Provide a hint to the tool on the dimensionality of "
                   "nd_range to use in generated code.\n"
                   "The values are:\n"),
    llvm::cl::init(AssumedNDRangeDimEnum::ARE_Dim3),
    llvm::cl::value_desc("value"), llvm::cl::cat(CtHelpCatCodeGen),
    llvm::cl::Optional)

DPCT_ENUM_OPTION(
    clang::dpct::list, ExplicitNamespace, UseExplicitNamespace,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "use-explicit-namespace",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE("dpct",
                               int(clang::dpct::ExplicitNamespace::EN_DPCT),
                               "Generate code with dpct:: namespace.", false),
        DPCT_OPTION_ENUM_VALUE("none",
                               int(clang::dpct::ExplicitNamespace::EN_None),
                               "Generate code without any namespaces. Cannot "
                               "be used with other values.",
                               false),
        DPCT_OPTION_ENUM_VALUE(
            "sycl", int(clang::dpct::ExplicitNamespace::EN_SYCL),
            "Generate code with sycl:: namespace. Cannot be used with cl or "
            "sycl-math values.",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "sycl-math", int(clang::dpct::ExplicitNamespace::EN_SYCL_Math),
            "Generate code with sycl:: namespace, applied only for "
            "SYCL math functions.\n"
            "Cannot be used with cl or sycl values.",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "syclcompat", int(clang::dpct::ExplicitNamespace::EN_SYCLCompat),
            "Generate code with syclcompat:: namespace.", false)),
    llvm::cl::desc("Define the namespaces to use explicitly in generated "
                   "code. The <value> is a comma\n"
                   "separated list. Default: dpct/syclcompat, sycl.\n"
                   "Possible values are:"),
    llvm::cl::CommaSeparated, llvm::cl::value_desc("value"),
    llvm::cl::cat(CtHelpCatCodeGen), llvm::cl::ZeroOrMore)

DPCT_ENUM_OPTION(
    clang::dpct::bits, DPCPPExtensionsDefaultEnabled, NoDPCPPExtensions,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "no-dpcpp-extensions",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE(
            "bfloat16", int(DPCPPExtensionsDefaultEnabled::ExtDE_BFloat16),
            "Disable the SYCL extensions for bfloat16.", false),
        DPCT_OPTION_ENUM_VALUE(
            "device_info", int(DPCPPExtensionsDefaultEnabled::ExtDE_DeviceInfo),
            "Disable the Intel extensions for device information, if "
            "supported by the compiler and the backend.",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "enqueued_barriers",
            int(DPCPPExtensionsDefaultEnabled::ExtDE_EnqueueBarrier),
            "Disable the enqueued barriers extension.", false),
        DPCT_OPTION_ENUM_VALUE(
            "peer_access", int(DPCPPExtensionsDefaultEnabled::ExtDE_PeerAccess),
            "Disable the peer access extension.", false),
        DPCT_OPTION_ENUM_VALUE("assert",
                               int(DPCPPExtensionsDefaultEnabled::ExtDE_Assert),
                               "Disable the assert extension.", false),
        DPCT_OPTION_ENUM_VALUE(
            "queue_empty", int(DPCPPExtensionsDefaultEnabled::ExtDE_QueueEmpty),
            "Disable the queue empty extension.", false),
        DPCT_OPTION_ENUM_VALUE("all",
                               int(DPCPPExtensionsDefaultEnabled::ExtDE_All),
                               "Disable all extensions listed in this option.",
                               false)),
    llvm::cl::desc("A comma-separated list of extensions not to be used in "
                   "migrated code.\n"
                   "By default, these extensions are used in migrated code.\n"
                   "The values are:"),
    llvm::cl::CommaSeparated, llvm::cl::value_desc("value"),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatAdvanced),
    llvm::cl::ZeroOrMore)

DPCT_ENUM_OPTION(
    clang::dpct::bits, DPCPPExtensionsDefaultDisabled, UseDPCPPExtensions,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "use-dpcpp-extensions",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE(
            "c_cxx_standard_library",
            int(DPCPPExtensionsDefaultDisabled::ExtDD_CCXXStandardLibrary),
            "Use std functions from the libdevice library (provided by "
            "Intel(R) oneAPI DPC++/C++ Compiler)\n"
            "and C/C++ Standard Library to migrate functions which have no "
            "mapping in the SYCL standard.\n"
            "If this value is used together with intel_device_math, the "
            "intel_device_math functions take\n"
            "precedence.",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "intel_device_math",
            int(DPCPPExtensionsDefaultDisabled::ExtDD_IntelDeviceMath),
            "Use sycl::ext::intel::math functions from the libdevice library "
            "(provided by Intel(R) oneAPI\n"
            "DPC++/C++ Compiler) to migrate functions which have no mapping in "
            "the SYCL standard.",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "all", int(DPCPPExtensionsDefaultDisabled::ExtDD_All),
            "Enable all DPC++ extensions listed in this option.", false)),
    llvm::cl::desc(
        "A comma-separated list of extensions to be used in migrated code.\n"
        "By default, these extensions are not used in migrated code."),
    llvm::cl::CommaSeparated, llvm::cl::value_desc("value"),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatAdvanced),
    llvm::cl::ZeroOrMore)

DPCT_ENUM_OPTION(
    clang::dpct::bits, ExperimentalFeatures, Experimentals,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "use-experimental-features",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE(
            "free-function-queries", int(ExperimentalFeatures::Exp_FreeQueries),
            "Experimental extension that allows getting 'id', "
            "'item', 'nd_item', 'group', and\n"
            "'sub_group' instances globally.",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "local-memory-kernel-scope-allocation",
            int(ExperimentalFeatures::Exp_GroupSharedMemory),
            "Experimental extension that allows allocation of local "
            "memory objects at the kernel\n"
            "functor scope",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "logical-group", int(ExperimentalFeatures::Exp_LogicalGroup),
            "Experimental helper function used to logically group work-items.",
            false),
        DPCT_OPTION_ENUM_VALUE("root-group",
                               int(ExperimentalFeatures::Exp_RootGroup),
                               "Experimental extension that allows use of root "
                               "group class and relative API.\n",
                               false),
        DPCT_OPTION_ENUM_VALUE(
            "nd_range_barrier", int(ExperimentalFeatures::Exp_NdRangeBarrier),
            "DEPRECATED : Experimental helper function used to help "
            "cross-group "
            "synchronization during migration. Please use the following option "
            "instead: --use-experimental-features=root-group.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "user-defined-reductions",
            int(ExperimentalFeatures::Exp_UserDefineReductions),
            "Experimental extension that allows user-defined reductions.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "masked-sub-group-operation",
            int(ExperimentalFeatures::Exp_MaskedSubGroupFunction),
            "Experimental helper function used to execute sub-group "
            "operation with mask.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "dpl-experimental-api",
            int(ExperimentalFeatures::Exp_DPLExperimentalAPI),
            "Experimental extension that allows use of experimental "
            "oneDPL APIs.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "occupancy-calculation",
            int(ExperimentalFeatures::Exp_OccupancyCalculation),
            "Experimental helper function used to calculate occupancy.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "matrix", int(ExperimentalFeatures::Exp_Matrix),
            "Experimental extension that allows use of matrix "
            "extension like class joint_matrix.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "bfloat16_math_functions",
            int(ExperimentalFeatures::Exp_BFloat16Math),
            "Experimental extension that allows use of bfloat16 math "
            "functions.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "bindless_images", int(ExperimentalFeatures::Exp_BindlessImages),
            "Experimental extension that allows use of bindless images APIs.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "graph", int(ExperimentalFeatures::Exp_Graph),
            "Experimental extension that allows use of SYCL Graph APIs.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "non-uniform-groups",
            int(ExperimentalFeatures::Exp_NonUniformGroups),
            "Experimental extension that allows use of non-uniform groups.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "device_global", int(ExperimentalFeatures::Exp_DeviceGlobal),
            "Experimental extension that allows device scoped memory "
            "allocations into SYCL that can\n"
            "be accessed within a kernel using syntax similar to C++ global "
            "variables.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "virtual_mem", int(ExperimentalFeatures::Exp_VirtualMemory),
            "Experimental extension that allows map an address range onto "
            "multiple allocations of physical memory.",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "in_order_queue_events",
            int(ExperimentalFeatures::Exp_InOrderQueueEvents),
            "Experimental extension that allows get the event from the "
            "last command submission into the queue and set an external "
            "event as an implicit dependence on the next command submitted to "
            "the queue.",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "non-stdandard-sycl-builtins",
            int(ExperimentalFeatures::Exp_NonStandardSYCLBuiltins),
            "Experimental extension that allows to use non standard SYCL "
            "builtin functions.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "prefetch", int(ExperimentalFeatures::Exp_Prefetch),
            "Experimental extension that allows use of SYCL prefetch APIs.\n",
            false),
        DPCT_OPTION_ENUM_VALUE(
            "all", int(ExperimentalFeatures::Exp_All),
            "Enable all experimental extensions listed in this option.\n",
            false)),
    llvm::cl::desc(
        "A comma-separated list of experimental features to be used in "
        "migrated "
        "code.\n"
        "By default, experimental features will not be used in migrated "
        "code.\nThe values are:\n"),
    llvm::cl::CommaSeparated, llvm::cl::value_desc("value"),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatAdvanced),
    llvm::cl::ZeroOrMore)

DPCT_FLAG_OPTION(GenBuildScript, clang::dpct::DpctOptionClass::OC_Attribute,
                 DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration,
                                     clang::dpct::DpctActionKind::DAK_Analysis),
                 "gen-build-script",
                 llvm::cl::desc("Generate makefile for migrated file(s) "
                                "in -out-root directory. Default: off."),
                 llvm::cl::cat(CtHelpCatAll),
                 llvm::cl::cat(CtHelpCatBuildScript))

DPCT_PATH_OPTION(
    BuildScriptFile, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "build-script-file",
    llvm::cl::desc(
        "Specify the name of generated makefile for migrated file(s).\n"
        "Default name: Makefile.dpct."),
    llvm::cl::value_desc("file"), llvm::cl::cat(CtHelpCatBuildScript),
    llvm::cl::Optional)

DPCT_FLAG_OPTION(
    MigrateBuildScriptOnly, clang::dpct::DpctOptionClass::OC_Action,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_BuildScript),
    "migrate-build-script-only",
    llvm::cl::desc(
        "EXPERIMENTAL: Only migrate the build script(s). Default: off."),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatBasic),
    llvm::cl::Optional)

DPCT_OPTION(
    clang::dpct::list, std::string, ExcludePathList,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_BuildScript,
                        clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Analysis),
    "in-root-exclude",
    llvm::cl::desc("Exclude the specified directory or file from processing."),
    llvm::cl::value_desc("dir|file"), llvm::cl::cat(CtHelpCatBasic),
    llvm::cl::ZeroOrMore)

DPCT_FLAG_OPTION(
    OptimizeMigration, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                        clang::dpct::DpctActionKind::DAK_Migration),
    "optimize-migration",
    llvm::cl::desc("Generate SYCL code applying more aggressive assumptions "
                   "that potentially\n"
                   "may alter the semantics of your program. Default: off."),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen))

DPCT_FLAG_OPTION(
    NoIncrementalMigration, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                        clang::dpct::DpctActionKind::DAK_Migration),
    "no-incremental-migration",
    llvm::cl::desc("Tell the tool to not perform an incremental migration.\n"
                   "Default: off (incremental migration happens)."))

DPCT_OPTION(
    clang::dpct::list, std::string, AnalysisScopeOpt,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                        clang::dpct::DpctActionKind::DAK_Migration),
    "analysis-scope-path",
    llvm::cl::desc(
        "The directory path for the analysis scope of the source tree that "
        "needs to be migrated.\n"
        "Default: the value of --in-root."),
    llvm::cl::value_desc("dir"), llvm::cl::cat(CtHelpCatBasic),
    llvm::cl::ZeroOrMore)

DPCT_FLAG_OPTION(
    LimitChangeExtension, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "change-cuda-files-extension-only",
    llvm::cl::desc(
        "Limit extension change to .cu and .cuh files only. Default: off."),
    llvm::cl::value_desc("file"), llvm::cl::cat(CtHelpCatCodeGen))

DPCT_ENUM_OPTION(
    clang::dpct::opt, SYCLFileExtensionEnum, SYCLFileExtension,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "sycl-file-extension",
    DPCT_OPTION_VALUES(
        DPCT_OPTION_ENUM_VALUE(
            "dp-cpp", int(SYCLFileExtensionEnum::DP_CPP),
            "Use extension '.dp.cpp' and '.dp.hpp' (default).", false),
        DPCT_OPTION_ENUM_VALUE("sycl-cpp", int(SYCLFileExtensionEnum::SYCL_CPP),
                               "Use extension '.sycl.cpp' and '.sycl.hpp'.",
                               false),
        DPCT_OPTION_ENUM_VALUE("cpp", int(SYCLFileExtensionEnum::CPP),
                               "Use extension '.cpp' and '.hpp'.", false)),
    llvm::cl::desc("Specify the extension of migrated source file(s).\nThe "
                   "values are:\n"),
    llvm::cl::init(SYCLFileExtensionEnum::DP_CPP),
    llvm::cl::value_desc("value"), llvm::cl::Optional)

DPCT_FLAG_OPTION(
    GenHelperFunction, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "gen-helper-function",
    llvm::cl::desc("Generate helper function files in the "
                   "--out-root directory. Default: off."),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen))

DPCT_FLAG_OPTION(
    PathToHelperFunction, clang::dpct::DpctOptionClass::OC_Action,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Help),
    "helper-function-dir",
    llvm::cl::desc(
        "Print the installation directory for helper function header files."),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatHelpInfo))

#ifndef _WIN32
DPCT_FLAG_OPTION(
    InterceptBuildCommand, clang::dpct::DpctOptionClass::OC_Action,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Independent),
    "intercept-build",
    llvm::cl::desc("Intercept build tool to generate a compilation database."),
    llvm::cl::value_desc("build command"),
    llvm::cl::cat(CtHelpCatInterceptBuild))
#endif

DPCT_FLAG_OPTION(
    CodePinReport, clang::dpct::DpctOptionClass::OC_Action,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Independent),
    "codepin-report",
    llvm::cl::desc(
        "Call codepin-report.py to generate CodePin report by parsing "
        "execution log files generated by instrumented CUDA and SYCL code."),
    llvm::cl::value_desc("build command"))

DPCT_OPTION(clang::dpct::opt, std::string, QueryAPIMapping,
            clang::dpct::DpctOptionClass::OC_Action,
            DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Query),
            "query-api-mapping",
            llvm::cl::desc(
                "Query functionally compatible SYCL API to migrate CUDA API."),
            llvm::cl::value_desc("api"), llvm::cl::cat(CtHelpCatQueryAPI),
            llvm::cl::Optional)

DPCT_ENUM_OPTION(
    clang::dpct::bits, HelperFuncPreference, Preferences,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                        clang::dpct::DpctActionKind::DAK_Migration),
    "helper-function-preference",
    DPCT_OPTION_VALUES(DPCT_OPTION_ENUM_VALUE(
        "no-queue-device", int(HelperFuncPreference::NoQueueDevice),
        "Call SYCL API to get queue and device "
        "instead of calling helper function.\n",
        false)),
    llvm::cl::desc("The preference of helper function usage in migration.\n"),
    llvm::cl::CommaSeparated, llvm::cl::value_desc("value"),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatCodeGen),
    llvm::cl::ZeroOrMore)

DPCT_FLAG_OPTION(
    UseSYCLCompat, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                        clang::dpct::DpctActionKind::DAK_Migration,
                        clang::dpct::DpctActionKind::DAK_Help,
                        clang::dpct::DpctActionKind::DAK_Query),
    "use-syclcompat",
    llvm::cl::desc(
        "Use SYCLcompat header-only library (syclcompat:: namespace) to assist "
        "the migration of input source code.\nDefault: off.\n"),
    llvm::cl::cat(CtHelpCatCodeGen), llvm::cl::cat(CtHelpCatAll))

DPCT_FLAG_OPTION(
    AnalysisMode, clang::dpct::DpctOptionClass::OC_Action,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis),
    "analysis-mode",
    llvm::cl::desc("Only generate a report for porting effort. Default: off."),
    llvm::cl::cat(CtHelpCatAll), llvm::cl::cat(CtHelpCatBasic))

DPCT_PATH_OPTION(
    AnalysisModeOutputFile, clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis),
    "analysis-mode-output-file",
    llvm::cl::desc(
        "Specify the file where the analysis mode report is saved. Default: "
        "Output to stdout."),
    llvm::cl::value_desc("file"), llvm::cl::cat(CtHelpCatBasic))

DPCT_HIDDEN_OPTION(
    clang::dpct::opt, std::string, SDKPathOpt,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                        clang::dpct::DpctActionKind::DAK_Migration),
    "cuda-path", desc("Directory path of SDK.\n"), llvm::cl::value_desc("dir"),
    llvm::cl::Optional)

#define DPCT_OPTION_PASS_VALUE_DESC                                            \
  llvm::cl::value_desc("IterationSpaceBuiltinRule,...")
DPCT_HIDDEN_OPTION(
    clang::dpct::opt, std::string, Passes,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Analysis,
                        clang::dpct::DpctActionKind::DAK_Migration),
    "passes",
    llvm::cl::desc(
        "Comma separated list of migration passes, which will be applied.\n"
        "Only the specified passes are applied."),
    DPCT_OPTION_PASS_VALUE_DESC)
#undef DPCT_OPTION_PASS_VALUE_DESC

#ifdef DPCT_DEBUG_BUILD
DPCT_HIDDEN_OPTION(
    clang::dpct::opt, std::string, DiagsContent,
    clang::dpct::DpctOptionClass::OC_Attribute,
    DPCT_OPTION_ACTIONS(clang::dpct::DpctActionKind::DAK_Migration),
    "report-diags-content",
    llvm::cl::desc("Diagnostics verbosity level. \"pass\": Basic migration "
                   "pass information. "
                   "\"transformation\": Detailed migration pass "
                   "transformation information."),
    llvm::cl::value_desc("[pass|transformation]"), llvm::cl::cat(CtHelpCatAll),
    llvm::cl::Optional)
#endif

#endif // !DPCT_OPTIONS_IN_CLANG_DPCT

#undef DPCT_OPTIONS_NAME_ENUM
#undef DPCT_OPTIONS_VAR
#undef DPCT_OPTIONS_IN_CLANG_TOOLING
#undef DPCT_OPTIONS_IN_CLANG_DPCT
#undef DPCT_OPTION
#undef DPCT_ALIASE
#undef DPCT_FLAG_OPTION
#undef DPCT_ENUM_OPTION
#undef DPCT_PATH_OPTION
#undef DPCT_SOURCEPATH_OPTION
#undef DPCT_OPTION_VALUES
#undef DPCT_OPTION_ENUM_VALUE
#undef DPCT_OPTION_ACTIONS
