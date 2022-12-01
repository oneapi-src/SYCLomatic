#ifndef DPCT_OPT_TYPE
#define DPCT_OPT_TYPE(...)
#endif
#ifndef DPCT_OPT_ENUM
#define DPCT_OPT_ENUM(NAME, ...)
#endif
#ifndef DPCT_OPTION_VALUES
#define DPCT_OPTION_VALUES(...)
#endif
#ifndef DPCT_NON_ENUM_OPTION
#define DPCT_NON_ENUM_OPTION(OPT_TYPE, OPT_VAR, OPTION_NAME, ...)
#endif
#ifndef DPCT_ENUM_OPTION
#define DPCT_ENUM_OPTION(OPT_TYPE, OPT_VAR, OPTION_NAME, ...)
#endif

#ifdef DPCT_OPTIONS_IN_CLANG_TOOLING
DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<std::string>), BuildPath, "p",
           llvm::cl::desc("The directory path for the compilation database (compile_commands.json). When no\n"
               "path is specified, a search for compile_commands.json is attempted through all\n"
               "parent directories of the first input source file."),
           llvm::cl::value_desc("dir"), llvm::cl::cat(Category),
           llvm::cl::Optional, llvm::cl::sub(*llvm::cl::AllSubCommands))
DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::list<std::string>), ArgsAfter, "extra-arg",
           llvm::cl::desc("Additional argument to append to the migration command line, example:\n"
              "--extra-arg=\"-I /path/to/header\". The options that can be passed this way can\n"
              "be found with the dpct -- -help command."),
           llvm::cl::value_desc("string"), llvm::cl::cat(Category),
           llvm::cl::sub(*llvm::cl::AllSubCommands))
#ifdef _WIN32
DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<std::string>), VcxprojFile, "vcxprojfile",
           llvm::cl::desc("The file path of vcxproj."),
           llvm::cl::value_desc("file"), llvm::cl::cat(Category),
           llvm::cl::Optional, llvm::cl::sub(*llvm::cl::AllSubCommands))
#endif
#endif // !DPCT_OPTIONS_IN_CLANG_TOOLING

#ifdef DPCT_OPTIONS_IN_CLANG_DPCT
DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<std::string>), InRoot, "in-root",
           llvm::cl::desc("The directory path for the root of the source tree that needs "
                "to be migrated.\n"
                "Only files under this root are migrated. Default: Current"
                " directory, if input\nsource files are not provided. "
                "If input source files are provided, the directory\n"
                "of the first input source file is used."),
           llvm::cl::value_desc("dir"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<std::string>), OutRoot, "out-root",
           llvm::cl::desc("The directory path for root of generated files. A directory is "
                "created if it\n"
                "does not exist. Default: dpct_output."),
           llvm::cl::value_desc("dir"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)
DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<std::string>), CudaIncludePath, "cuda-include-path",
           llvm::cl::desc("The directory path of the CUDA header files."),
           llvm::cl::value_desc("dir"), cat(DPCTCat),
           llvm::cl::Optional)
DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<std::string>), ReportFilePrefix, "report-file-prefix",
    llvm::cl::desc(
        "Prefix for the report file names. The full file name will have a "
        "suffix derived\n"
        "from the report-type and an extension derived from the report-format. "
        "For\n"
        "example: <prefix>.apis.csv or <prefix>.stats.log. If this option is "
        "not\n"
        "specified, the report will go to stdout. The report files are created "
        "in the\n"
        "directory, specified by -out-root."),
           llvm::cl::value_desc("prefix"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)
DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool, true>), ReportOnly, "report-only",
           llvm::cl::desc("Only reports are generated. No SYCL code is "
                              "generated. Default: off."),
           llvm::cl::cat(DPCTCat), llvm::cl::location(ReportOnlyFlag))
DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool, true>), ShowOrigCode, "keep-original-code",
           llvm::cl::desc("Keeps the original code in comments of "
                                "generated SYCL files. Default: off.\n"),
           llvm::cl::cat(DPCTCat), llvm::cl::location(KeepOriginalCodeFlag))

#ifdef DPCT_DEBUG_BUILD
DPCT_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<ReportTypeEnum>), ReportType, "report-type",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("apis", int(ReportTypeEnum::RTE_APIs),
            "Information about API signatures that need migration and the number of times\n"
            "they were encountered. The report file name will have .apis suffix added.", false),
             DPCT_OPT_ENUM("stats", int(ReportTypeEnum::RTE_Stats),
                  "High level migration statistics: Lines Of Code (LOC) that are migrated to\n"
                  "SYCL, LOC migrated to SYCL with helper functions, LOC not needing migration,\n"
                  "LOC needing migration but are not migrated. The report file name has the .stats\n"
                  "suffix added (default)", false),
             DPCT_OPT_ENUM("all", int(ReportTypeEnum::RTE_All),
                  "All of the reports.", false),
             DPCT_OPT_ENUM("diags", int(ReportTypeEnum::RTE_Diags),
                  "diags information", true)
           ),
           llvm::cl::desc("Specifies the type of report. Values are:\n"),
           llvm::cl::init(ReportTypeEnum::RTE_NotSetType), llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)
#else
DPCT_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<ReportTypeEnum>), ReportType, "report-type",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("apis", int(ReportTypeEnum::RTE_APIs),
            "Information about API signatures that need migration and the number of times\n"
            "they were encountered. The report file name will have .apis suffix added.", false),
             DPCT_OPT_ENUM("stats", int(ReportTypeEnum::RTE_Stats),
                  "High level migration statistics: Lines Of Code (LOC) that are migrated to\n"
                  "SYCL, LOC migrated to SYCL with helper functions, LOC not needing migration,\n"
                  "LOC needing migration but are not migrated. The report file name has the .stats\n"
                  "suffix added (default)", false),
             DPCT_OPT_ENUM("all", int(ReportTypeEnum::RTE_All),
                  "All of the reports.", false)
           ),
           llvm::cl::desc("Specifies the type of report. Values are:\n"),
           llvm::cl::init(ReportTypeEnum::RTE_NotSetType), llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)
#endif

DPCT_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<ReportFormatEnum>), ReportFormat, "report-format",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("csv", int(ReportFormatEnum::RFE_CSV),
                  "Output is lines of comma separated values. The report file "
                  "name extension will\n"
                  "be .csv. (default)", false),
             DPCT_OPT_ENUM("formatted", int(ReportFormatEnum::RFE_Formatted),
                  "Output is formatted to be easier to read for "
                  "human eyes. Report file name\n"
                  "extension will be log.",
                  false)
           ),
           llvm::cl::desc("Format of the reports:\n"),
           llvm::cl::init(ReportFormatEnum::RFE_NotSetFormat), llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(llvm::cl::opt<std::string>), SuppressWarnings, "suppress-warnings",
           llvm::cl::desc(SuppressWarningsMessage),
                llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat))

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool, true>), SuppressWarningsAll, "suppress-warnings-all",
           llvm::cl::desc("Suppresses all migration warnings. Default: off."),
           llvm::cl::cat(DPCTCat), llvm::cl::location(SuppressWarningsAllFlag))

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool, true>), StopOnParseErrOption, "stop-on-parse-err",
           llvm::cl::desc("Stop migration and generation of reports if "
                               "parsing errors happened. Default: off. \n"),
           llvm::cl::cat(DPCTCat), llvm::cl::location(StopOnParseErr))

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool, true>), CheckUnicodeSecurity, "check-unicode-security",
           llvm::cl::desc("Enable detection and warnings about Unicode constructs that can be exploited by using\n"
                   "bi-directional formatting codes and homoglyphs in identifiers. Default: off.\n"),
           llvm::cl::cat(DPCTCat), llvm::cl::location(CheckUnicodeSecurityFlag))
DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool, true>), SyclNamedLambda, "sycl-named-lambda",
           llvm::cl::desc("Generates kernels with the kernel name. Default: off.\n"),
           llvm::cl::cat(DPCTCat), llvm::cl::location(SyclNamedLambdaFlag))

DPCT_ENUM_OPTION(DPCT_OPT_TYPE(llvm::cl::opt<OutputVerbosityLevel>), OutputVerbosity, "output-verbosity",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("silent", int(OutputVerbosityLevel::OVL_Silent),
                                  "Only messages from clang.", false),
             DPCT_OPT_ENUM("normal", int(OutputVerbosityLevel::OVL_Normal),
                                  "\'silent\' and warnings, errors, and notes from dpct.",
                                  false),
             DPCT_OPT_ENUM("detailed", int(OutputVerbosityLevel::OVL_Detailed),
                                  "\'normal\' and messages about which file is being processed.",
                                  false),
             DPCT_OPT_ENUM("diagnostics", int(OutputVerbosityLevel::OVL_Diagnostics),
                                  "\'detailed\' and information about the detected "
                                  "conflicts and crashes. (default)", false)
           ),
           llvm::cl::desc("Sets the output verbosity level:"),
           llvm::cl::init(OutputVerbosityLevel::OVL_Diagnostics), llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(llvm::cl::opt<std::string>), OutputFile, "output-file",
           llvm::cl::desc("Redirects the stdout/stderr output to <file> in the output"
                    " directory specified\n"
                    "by the --out-root option."),
           llvm::cl::value_desc("file"), llvm::cl::cat(DPCTCat), llvm::cl::Optional)

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::list<std::string>), RuleFile, "rule-file",
           llvm::cl::desc("Specifies the rule file path that contains rules used for migration.\n"),
           llvm::cl::value_desc("file"), llvm::cl::cat(DPCTCat), llvm::cl::ZeroOrMore)

DPCT_ENUM_OPTION(DPCT_OPT_TYPE(llvm::cl::opt<UsmLevel>), USMLevel, "usm-level",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("restricted", int(UsmLevel::UL_Restricted),
                     "Uses USM API for memory management migration. (default)", false),
             DPCT_OPT_ENUM("none", int(UsmLevel::UL_None),
                     "Uses helper functions from DPCT header files for memory "
                     "management migration.", false)
           ),
           llvm::cl::desc("Sets the Unified Shared Memory (USM) level to use in source code generation.\n"),
           llvm::cl::init(UsmLevel::UL_Restricted), llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)

DPCT_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<format::FormatRange>), FormatRng, "format-range",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("migrated", int(format::FormatRange::migrated),
                     "Only formats the migrated code (default).", false),
             DPCT_OPT_ENUM("all", int(format::FormatRange::all),
                     "Formats all code.", false),
             DPCT_OPT_ENUM("none", int(format::FormatRange::none),
                     "Do not format any code.", false)
           ),
           llvm::cl::desc("Sets the range of formatting.\nThe values are:\n"),
           llvm::cl::init(format::FormatRange::migrated), llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)
DPCT_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<DPCTFormatStyle>), FormatST, "format-style",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("llvm", int(DPCTFormatStyle::FS_LLVM),
                     "Use the LLVM coding style.", false),
             DPCT_OPT_ENUM("google", int(DPCTFormatStyle::FS_Google),
                     "Use the Google coding style.", false),
             DPCT_OPT_ENUM("custom", int(DPCTFormatStyle::FS_Custom),
                     "Use the coding style defined in the .clang-format file (default).", false)
           ),
           llvm::cl::desc("Sets the formatting style.\nThe values are:\n"),
           llvm::cl::init(DPCTFormatStyle::FS_Custom), llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool, true>), NoClNamespaceInline, "no-cl-namespace-inline",
           llvm::cl::desc("DEPRECATED: Do not use cl:: namespace inline. Default: off. This option will be\n"
                   "ignored if the replacement option --use-explicit-namespace is used.\n"),
           llvm::cl::cat(DPCTCat), llvm::cl::location(ExplicitClNamespace))

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool, true>), NoDRYPattern, "no-dry-pattern",
           llvm::cl::desc("Do not use DRY (do not repeat yourself) pattern when functions from dpct\n"
                                   "namespace are inserted. Default: off.\n"),
           llvm::cl::cat(DPCTCat), llvm::cl::location(NoDRYPatternFlag))

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool, true>), ProcessAll, "process-all",
           llvm::cl::desc("Migrates or copies all files, except hidden, from the --in-root directory\n"
                 "to the --out-root directory. The --in-root option should be explicitly specified.\n"
                 "Default: off."),
           llvm::cl::cat(DPCTCat), llvm::cl::location(ProcessAllFlag))

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool>), EnableCTAD, "enable-ctad",
           llvm::cl::desc("Use a C++17 class template argument deduction (CTAD) in "
                   "your generated code.\n"
                   "Default: off."),
           llvm::cl::cat(DPCTCat), llvm::cl::init(false))

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool>), EnableComments, "comments",
           llvm::cl::desc("Insert comments explaining the generated code. Default: off."),
           llvm::cl::cat(DPCTCat), llvm::cl::init(false))

DPCT_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<HelperFilesCustomizationLevel>), UseCustomHelperFileLevel, "use-custom-helper",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("none", int(HelperFilesCustomizationLevel::HFCL_None),
            "No customization (default).", false),
             DPCT_OPT_ENUM("file", int(HelperFilesCustomizationLevel::HFCL_File),
            "Limit helper header files to only the necessary files for the migrated code and\n"
            "place them in the --out-root directory.", false),
             DPCT_OPT_ENUM("api", int(HelperFilesCustomizationLevel::HFCL_API),
            "Limit helper header files to only the necessary APIs for the migrated code and\n"
            "place them in the --out-root directory.", false),
             DPCT_OPT_ENUM("all", int(HelperFilesCustomizationLevel::HFCL_All),
            "Generate a complete set of helper header files and place them in the --out-root\n"
            "directory.", false)
           ),
           llvm::cl::desc("DEPRECATED: Customize the helper header files for migrated code. The values are:\n"),
           llvm::cl::init(HelperFilesCustomizationLevel::HFCL_None), llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<std::string>), CustomHelperFileName, "custom-helper-name",
        llvm::cl::desc("DEPRECATED: Specifies the helper headers folder name and main helper header file name.\n"
        "Default: dpct."),
        llvm::cl::init("dpct"), llvm::cl::value_desc("name"), llvm::cl::cat(DPCTCat), llvm::cl::Optional)

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool, true>), AsyncHandler, "always-use-async-handler",
        llvm::cl::desc("Use async exception handler when creating new sycl::queue "
                   "with dpct::create_queue\nin addition to default "
                   "dpct::get_default_queue. Default: off."),
        llvm::cl::cat(DPCTCat), llvm::cl::location(AsyncHandlerFlag))

DPCT_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<AssumedNDRangeDimEnum>), NDRangeDim, "assume-nd-range-dim",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("1", 1,
                                  "Generate kernel code assuming 1D nd_range "
                                  "where possible, and 3D in other cases.",
                                  false),
             DPCT_OPT_ENUM("3", 3,
            "Generate kernel code assuming 3D nd_range (default).",
            false)
           ),
           llvm::cl::desc("Provides a hint to the tool on the dimensionality of nd_range to use in generated code.\n"
           "The values are:\n"),
           llvm::cl::init(AssumedNDRangeDimEnum::ARE_Dim3), llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat),
           llvm::cl::Optional)

DPCT_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::list<ExplicitNamespace>), UseExplicitNamespace, "use-explicit-namespace",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("none", int(ExplicitNamespace::EN_None),
                                     "Generate code without namespaces. Cannot "
                                     "be used with other values.",
                                     false),
             DPCT_OPT_ENUM("cl", int(ExplicitNamespace::EN_CL),
               "DEPRECATED: Generate code with cl::sycl:: namespace. Cannot be used with "
               "sycl or sycl-math values.",
               false),
             DPCT_OPT_ENUM("dpct", int(ExplicitNamespace::EN_DPCT),
                                     "Generate code with dpct:: namespace.",
                                     false),
             DPCT_OPT_ENUM("sycl", int(ExplicitNamespace::EN_SYCL),
               "Generate code with sycl:: namespace. Cannot be used with cl or "
               "sycl-math values.",
               false),
             DPCT_OPT_ENUM("sycl-math", int(ExplicitNamespace::EN_SYCL_Math),
               "Generate code with sycl:: namespace, applied only for SYCL math functions.\n"
               "Cannot be used with cl or sycl values.",
               false)
           ),
           llvm::cl::desc("Defines the namespaces to use explicitly in generated code. The value is a comma\n"
           "separated list. Default: dpct, sycl.\n"
           "Possible values are:"),
           llvm::cl::CommaSeparated, llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat), llvm::cl::ZeroOrMore)

DPCT_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::list<DPCPPExtensionsDefaultEnabled>), NoDPCPPExtensions, "no-dpcpp-extensions",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("enqueued_barriers",
                                     int(DPCPPExtensionsDefaultEnabled::ExtDE_EnqueueBarrier),
                                     "Enqueued barriers extension.",
                                     false),
             DPCT_OPT_ENUM("device_info",
                                     int(DPCPPExtensionsDefaultEnabled::ExtDE_DeviceInfo),
                                     "Intel's Extensions for Device Information.",
                                     false)                            
           ),
           llvm::cl::desc("Comma separated list of extensions not to be used in migrated "
           "code.\n"
           "By default, these extensions will be used in migrated code."),
           llvm::cl::CommaSeparated, llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat), llvm::cl::ZeroOrMore,
           llvm::cl::cb<void, DPCPPExtensionsDefaultEnabled>(DpctGlobalInfo::setExtensionDEUnused))

DPCT_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::list<DPCPPExtensionsDefaultDisabled>), UseDPCPPExtensions, "use-dpcpp-extensions",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("c_cxx_standard_library",
                                     int(DPCPPExtensionsDefaultDisabled::ExtDD_CCXXStandardLibrary),
                                     "A set of functions in std namespace from the C and C++ standard libraries that are\n"
                                     "allowed to be used in SYCL device code.",
                                     false),
             DPCT_OPT_ENUM("intel_device_math",
                                     int(DPCPPExtensionsDefaultDisabled::ExtDD_IntelDeviceMath),
                                     "A set of functions in sycl::ext::intel::math namespace from the libdevice that are\n"
                                     "allowed to be used in SYCL device code.",
                                     false)
           ),
           llvm::cl::desc("A comma separated list of extensions to be used in migrated code.\n"
           "By default, these extensions are not used in migrated code."),
           llvm::cl::CommaSeparated, llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat), llvm::cl::ZeroOrMore,
           llvm::cl::cb<void, DPCPPExtensionsDefaultDisabled>(DpctGlobalInfo::setExtensionDDUsed))

DPCT_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::bits<ExperimentalFeatures>), Experimentals, "use-experimental-features",
           DPCT_OPTION_VALUES(
             DPCT_OPT_ENUM("free-function-queries", int(ExperimentalFeatures::Exp_FreeQueries),
        "Experimental extension that allows getting `id`, `item`, `nd_item`, `group`, and\n"
        "`sub_group` instances globally.",
        false),
             DPCT_OPT_ENUM("local-memory-kernel-scope-allocation", int(ExperimentalFeatures::Exp_GroupSharedMemory),
        "Experimental extension that allows allocation of local memory objects at the kernel\n"
        "functor scope",
        false),
             DPCT_OPT_ENUM("logical-group", int(ExperimentalFeatures::Exp_LogicalGroup),
        "Experimental helper function used to logically group work-items.",
        false),
             DPCT_OPT_ENUM("nd_range_barrier", int(ExperimentalFeatures::Exp_NdRangeBarrier),
        "Experimental helper function used to help cross group synchronization during migration.\n",
        false)
           ),
           llvm::cl::desc("Comma separated list of experimental features to be used in migrated "
           "code.\n"
           "By default, experimental features will not be used in migrated code.\nThe values are:\n"),
           llvm::cl::CommaSeparated, llvm::cl::value_desc("value"), llvm::cl::cat(DPCTCat), llvm::cl::ZeroOrMore)

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(llvm::cl::opt<bool>), GenBuildScript, "gen-build-script",
        llvm::cl::desc("Generates makefile for migrated file(s) in -out-root directory. Default: off."),
        llvm::cl::cat(DPCTCat), llvm::cl::init(false))
DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(llvm::cl::opt<std::string>), BuildScriptFile, "build-script-file",
        llvm::cl::desc("Specifies the name of generated makefile for migrated file(s).\n"
                    "Default name: Makefile.dpct."),
        llvm::cl::value_desc("file"), llvm::cl::cat(DPCTCat), llvm::cl::Optional)

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::list<std::string>), ExcludePathList, "in-root-exclude",
        llvm::cl::desc("Excludes the specified directory or file from processing."),
        llvm::cl::value_desc("dir|file"), llvm::cl::cat(DPCTCat), llvm::cl::ZeroOrMore)

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool>), OptimizeMigration, "optimize-migration",
        llvm::cl::desc("Generates SYCL code applying more aggressive assumptions that potentially\n"
                   "may alter the semantics of your program. Default: off."),
        llvm::cl::cat(DPCTCat), llvm::cl::init(false))

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<bool>), NoIncrementalMigration, "no-incremental-migration",
        llvm::cl::desc("Tells the tool to not perform an incremental migration.\n"
                   "Default: off (incremental migration happens)."),
        llvm::cl::cat(DPCTCat), llvm::cl::init(false))

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<std::string>), QueryApiMapping, "query-api-mapping",
        llvm::cl::desc("Outputs to stdout functionally compatible SYCL API mapping for CUDA API."),
        llvm::cl::value_desc("api"), llvm::cl::cat(DPCTCat), llvm::cl::Optional)

DPCT_NON_ENUM_OPTION(DPCT_OPT_TYPE(static llvm::cl::opt<std::string>), AnalysisScope, "analysis-scope-path",
        llvm::cl::desc("The directory path for the analysis scope of the source tree that "
             "needs to be migrated.\n"
             "Default: the value of --in-root."),
        llvm::cl::value_desc("dir"), llvm::cl::cat(DPCTCat))

#endif // !DPCT_OPTIONS_IN_CLANG_DPCT
