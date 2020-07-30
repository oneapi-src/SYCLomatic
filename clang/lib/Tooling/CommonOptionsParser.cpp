//===--- CommonOptionsParser.cpp - common options for clang tools ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the CommonOptionsParser class used to parse common
//  command-line options for clang tools, so that they can be run as separate
//  command-line applications with a consistent common interface for handling
//  compilation database and input files.
//
//  It provides a common subset of command-line options, common algorithm
//  for locating a compilation database and source files, and help messages
//  for the basic command-line interface.
//
//  It creates a CompilationDatabase and reads common command-line options.
//
//  This class uses the Clang Tooling infrastructure, see
//    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
//  for details on setting it up with LLVM source tree.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

using namespace clang::tooling;
using namespace llvm;

const char *const CommonOptionsParser::HelpMessage =
    "\n"
    "-p <build-path> is used to read a compile command database.\n"
    "\n"
    "\tFor example, it can be a CMake build directory in which a file named\n"
    "\tcompile_commands.json exists (use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\n"
    "\tCMake option to get this output). When no build path is specified,\n"
    "\ta search for compile_commands.json will be attempted through all\n"
    "\tparent paths of the first input file . See:\n"
    "\thttps://clang.llvm.org/docs/HowToSetupToolingForLLVM.html for an\n"
    "\texample of setting up Clang Tooling on a source tree.\n"
    "\n"
    "<source0> ... specify the paths of source files. These paths are\n"
    "\tlooked up in the compile command database. If the path of a file is\n"
    "\tabsolute, it needs to point into CMake's source tree. If the path is\n"
    "\trelative, the current working directory needs to be in the CMake\n"
    "\tsource tree and the file must be in a subdirectory of the current\n"
    "\tworking directory. \"./\" prefixes in the relative files will be\n"
    "\tautomatically removed, but the rest of a relative path must be a\n"
    "\tsuffix of a path in the compile command database.\n"
    "\n";

#ifdef INTEL_CUSTOMIZATION
#ifdef _WIN32
std::string VcxprojFilePath;
#endif
namespace clang {
namespace tooling {
static std::string FormatSearchPath = "";
std::string getFormatSearchPath() { return FormatSearchPath; }
#ifdef _WIN32
static FunPtrParserType FPtrParser = nullptr;

void SetParserHandle(FunPtrParserType FPParser) {
  FPtrParser = FPParser;
}

void DoParserHandle(std::string &BuildDir, std::string &FilePath) {
  if (FPtrParser != nullptr) {
    FPtrParser(BuildDir, FilePath);
  }
}
#endif
} // namespace tooling
} // namespace clang
#endif


void ArgumentsAdjustingCompilations::appendArgumentsAdjuster(
    ArgumentsAdjuster Adjuster) {
  Adjusters.push_back(std::move(Adjuster));
}

std::vector<CompileCommand>
ArgumentsAdjustingCompilations::getCompileCommands(StringRef FilePath) const {
  return adjustCommands(Compilations->getCompileCommands(FilePath));
}

std::vector<std::string> ArgumentsAdjustingCompilations::getAllFiles() const {
  return Compilations->getAllFiles();
}

std::vector<CompileCommand>
ArgumentsAdjustingCompilations::getAllCompileCommands() const {
  return adjustCommands(Compilations->getAllCompileCommands());
}

std::vector<CompileCommand> ArgumentsAdjustingCompilations::adjustCommands(
    std::vector<CompileCommand> Commands) const {
  for (CompileCommand &Command : Commands)
    for (const auto &Adjuster : Adjusters)
      Command.CommandLine = Adjuster(Command.CommandLine, Command.Filename);
  return Commands;
}

llvm::Error CommonOptionsParser::init(
    int &argc, const char **argv, cl::OptionCategory &Category,
    llvm::cl::NumOccurrencesFlag OccurrencesFlag, const char *Overview) {
#ifdef INTEL_CUSTOMIZATION
  bool IsCudaFile = false;
  int OriArgc = argc;
  static cl::opt<std::string> BuildPath(
      "p",
      cl::desc("The directory path for the compilation database (compile_commands.json). When no\n"
               "path is specified, a search for compile_commands.json is attempted through all\n"
               "parent directories of the first input source file."),
      cl::Optional, cl::cat(Category), cl::value_desc("dir"),
      cl::sub(*cl::AllSubCommands));

  static cl::list<std::string> SourcePaths(
      cl::Positional, cl::desc("[<source0> ... <sourceN>]"), llvm::cl::ZeroOrMore,
      cl::cat(Category), cl::sub(*cl::AllSubCommands));
#ifdef _WIN32
  static cl::opt<std::string>
    VcxprojFile("vcxprojfile",
                cl::desc("The file path of vcxproj."),
                cl::value_desc("file"),
                cl::Optional, cl::cat(Category),
                cl::sub(*cl::AllSubCommands));
#endif
#else
  static cl::opt<std::string> BuildPath("p", cl::desc("Build path"),
                                        cl::Optional, cl::cat(Category),
                                        cl::sub(*cl::AllSubCommands));

  static cl::list<std::string> SourcePaths(
      cl::Positional, cl::desc("<source0> [... <sourceN>]"), OccurrencesFlag,
      cl::cat(Category), cl::sub(*cl::AllSubCommands));
#endif

#ifdef INTEL_CUSTOMIZATION
 static cl::list<std::string> ArgsAfter(
     "extra-arg",
     cl::desc("Additional argument to append to the migration command line, example:\n"
              "--extra-arg=\"-I /path/to/header\". The options that can be passed this way can\n"
              "be found with the dpct -- -help command."),
     cl::value_desc("string"), cl::cat(Category), cl::sub(*cl::AllSubCommands));

  static cl::list<std::string> ArgsBefore(
     "extra-arg-before",
     cl::desc("Additional argument to prepend to the compiler command line.\n"
              "Refer to extra-arg option.\n"),
     cl::cat(Category), cl::sub(*cl::AllSubCommands), llvm::cl::Hidden);
#else
  static cl::list<std::string> ArgsAfter(
      "extra-arg",
      cl::desc("Additional argument to append to the compiler command line"),
      cl::cat(Category), cl::sub(*cl::AllSubCommands));

  static cl::list<std::string> ArgsBefore(
      "extra-arg-before",
      cl::desc("Additional argument to prepend to the compiler command line"),
      cl::cat(Category), cl::sub(*cl::AllSubCommands));
#endif
  cl::ResetAllOptionOccurrences();

  cl::HideUnrelatedOptions(Category);

  std::string ErrorMessage;
  Compilations =
      FixedCompilationDatabase::loadFromCommandLine(argc, argv, ErrorMessage);
  if (!ErrorMessage.empty())
    ErrorMessage.append("\n");
  llvm::raw_string_ostream OS(ErrorMessage);
  // Stop initializing if command-line option parsing failed.
  if (!cl::ParseCommandLineOptions(argc, argv, Overview, &OS)) {
    OS.flush();
#ifdef INTEL_CUSTOMIZATION
    return llvm::make_error<llvm::StringError>(ErrorMessage,
                                               llvm::inconvertibleErrorCode());
#else
    return llvm::make_error<llvm::StringError>("[CommonOptionsParser]: " +
                                                   ErrorMessage,
                                               llvm::inconvertibleErrorCode());
#endif
  }

  cl::PrintOptionValues();

  SourcePathList = SourcePaths;
#ifdef INTEL_CUSTOMIZATION
  if(!SourcePathList.empty()) {
    clang::tooling::FormatSearchPath = SourcePaths[0];
  }
  DatabaseStatus ErrCode =
      CannotFindDatabase; // map to MigrationErrorCannotFindDatabase in DPCT
#if _WIN32
  VcxprojFilePath = VcxprojFile;
  // In Windows, the option "-p" and "-vcxproj" are mutually exclusive, user can
  // only give one of them. If both of them exist, dpct will exit with
  // -1 (.i.e MigrationError).
  if (!BuildPath.empty() && !VcxprojFile.empty()) {
    ErrorMessage =
        "The option \"-p\" and \"-vcxproj\" are set together, please specify one of them.\n";
    llvm::errs() << ErrorMessage;
    exit(-1);
  }

  // In Windows, If the option
  // "--cuda-path" specifies SDK path in the command line, duplicate find the
  // the compilation database in the directory of the vcxproj file.
  if (!VcxprojFile.empty()) {
    SmallString<1024> AbsolutePath(getAbsolutePath(VcxprojFile));
    StringRef Directory = llvm::sys::path::parent_path(AbsolutePath);
    std::string BuildDir = Directory.str();
    DoParserHandle(BuildDir, VcxprojFile);
    Compilations = CompilationDatabase::autoDetectFromDirectory(
        BuildDir, ErrorMessage, ErrCode);
    clang::tooling::FormatSearchPath = BuildDir;
  }
#endif
#endif
  if ((OccurrencesFlag == cl::ZeroOrMore || OccurrencesFlag == cl::Optional) &&
      SourcePathList.empty())
    return llvm::Error::success();
  if (!Compilations) {
    if (!BuildPath.empty()) {
      Compilations = CompilationDatabase::autoDetectFromDirectory(
          BuildPath, ErrorMessage, ErrCode);
#ifdef INTEL_CUSTOMIZATION
      clang::tooling::FormatSearchPath = BuildPath;
#endif
    }
#ifdef INTEL_CUSTOMIZATION
    // if neither option "-p" or target source file names exist in the
    // command line, e.g "dpct -in-root=./ -out-root=out", dpct will not
    // continue.
    else if (BuildPath.empty() && SourcePathList.empty()) {
      Compilations = nullptr;
    }
#endif
    else {
      Compilations = CompilationDatabase::autoDetectFromSource(SourcePaths[0],
                                                               ErrorMessage);
#ifdef INTEL_CUSTOMIZATION
      clang::tooling::FormatSearchPath = SourcePaths[0];
#endif
    }
    if (!Compilations) {
#ifdef INTEL_CUSTOMIZATION
      if (SourcePaths.size() == 0 && !BuildPath.getValue().empty()){
        std::string buf;
        llvm::raw_string_ostream OS(buf);
        OS << "Error while trying to load a compilation database:\n"
           << ErrorMessage;
        DoPrintHandler(OS.str(), true);
        // The ErrCode is set to CannotParseDatabase(-101) from
        // findCompilationDatabaseFromDirectory in autoDetectFromDirectory when
        // database file exists but it cannot be parsed successfully. No other
        // value will set be to ErrCode. So the situation will be either
        // "CannotParseDatabase" or "CannotFindDatabase".

        if (ErrCode == CannotParseDatabase
          /*map to MigrationErrorCannotParseDatabase in DPCT*/) {
          return llvm::make_error<DPCTError>(
              CannotParseDatabase
              /*map to MigrationErrorCannotParseDatabase in DPCT*/);
        } else {
          return llvm::make_error<DPCTError>(
              CannotFindDatabase
              /*map to MigrationErrorCannotFindDatabase in DPCT*/);
        }
      } else if (SourcePaths.size() == 1 && BuildPath.getValue().empty()) {
        // need add -x cuda option for not using database
        IsCudaFile = true;
        using namespace llvm::sys;
        SmallString<256> Name = StringRef(SourcePaths[0]);
        StringRef File, Path;
        if (fs::make_absolute(Name) != std::error_code()) {
          std::string buf;
          llvm::raw_string_ostream OS(buf);
          OS << "Could not get absolute path from '" << Name << "'\n";
          DoPrintHandler(OS.str(), true);
        } else {
          File = path::filename(Name);
          Path = path::parent_path(Name);
        }
        std::string buf;
        llvm::raw_string_ostream OS(buf);
        OS << "NOTE: Could not auto-detect compilation database for"
           << " file '" << File << "' in '" << Path
           << "' or any parent directory.\n";
        DoPrintHandler(OS.str(), true);
      } else {
        if (SourcePaths.size() >= 2 && BuildPath.getValue().empty()) {
          // need add -x cuda option for not using database
          IsCudaFile = true;
        }
        if (!hasHelpOption(OriArgc, argv)) {
          std::string buf;
          llvm::raw_string_ostream OS(buf);
          OS << "Error while trying to load a compilation database:\n"
             << ErrorMessage;
          DoPrintHandler(OS.str(), true);
        }
      }
#else
      llvm::errs() << "Error while trying to load a compilation database:\n"
                   << ErrorMessage << "Running without flags.\n";
#endif
      Compilations.reset(
          new FixedCompilationDatabase(".", std::vector<std::string>()));
    }
  }
  auto AdjustingCompilations =
      std::make_unique<ArgumentsAdjustingCompilations>(
          std::move(Compilations));
  Adjuster =
      getInsertArgumentAdjuster(ArgsBefore, ArgumentInsertPosition::BEGIN);
  Adjuster = combineAdjusters(
      std::move(Adjuster),
      getInsertArgumentAdjuster(ArgsAfter, ArgumentInsertPosition::END));
#ifdef INTEL_CUSTOMIZATION
  for (auto &I : ArgsAfter) {
    if (I.size() > 2 && I.substr(0, 2) == "-x") {
      IsCudaFile = false;
      Adjuster = combineAdjusters(
          std::move(Adjuster),
          getInsertArgumentAdjuster(I.c_str(), ArgumentInsertPosition::BEGIN));
    }
  }
  if (IsCudaFile) {
    Adjuster = combineAdjusters(
        std::move(Adjuster),
        getInsertArgumentAdjuster("-xcuda", ArgumentInsertPosition::BEGIN));
  }
#endif
  AdjustingCompilations->appendArgumentsAdjuster(Adjuster);
  Compilations = std::move(AdjustingCompilations);
  return llvm::Error::success();
}

llvm::Expected<CommonOptionsParser> CommonOptionsParser::create(
    int &argc, const char **argv, llvm::cl::OptionCategory &Category,
    llvm::cl::NumOccurrencesFlag OccurrencesFlag, const char *Overview) {
  CommonOptionsParser Parser;
  llvm::Error Err =
      Parser.init(argc, argv, Category, OccurrencesFlag, Overview);
  if (Err)
    return std::move(Err);
  return std::move(Parser);
}

#ifdef INTEL_CUSTOMIZATION
bool CommonOptionsParser::hasHelpOption(int argc, const char **argv) {
  for (auto i = 0; i < argc; i++) {
    int Res1 = strcmp(argv[i], "-help");
    int Res2 = strcmp(argv[i], "--help");
    int Res3 = strcmp(argv[i], "--help-hidden");
    if (Res1 == 0 || Res2 == 0 || Res3 == 0)
      return true;
  }
  return false;
}
#endif

CommonOptionsParser::CommonOptionsParser(
    int &argc, const char **argv, cl::OptionCategory &Category,
    llvm::cl::NumOccurrencesFlag OccurrencesFlag, const char *Overview) {
  llvm::Error Err = init(argc, argv, Category, OccurrencesFlag, Overview);
  if (Err) {
    llvm::report_fatal_error(
        "CommonOptionsParser: failed to parse command-line arguments. " +
        llvm::toString(std::move(Err)));
  }
}
