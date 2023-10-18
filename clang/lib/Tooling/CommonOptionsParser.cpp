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


#ifdef SYCLomatic_CUSTOMIZATION
namespace clang {
namespace tooling {
#ifdef _WIN32
std::string VcxprojFilePath;
#endif

static std::string FormatSearchPath = "";
std::string getFormatSearchPath() { return FormatSearchPath; }
extern bool SpecifyLanguageInOption;
void emitDefaultLanguageWarningIfNecessary(const std::string &FileName,
                                           bool SpecifyLanguageInOption);
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
#endif // SYCLomatic_CUSTOMIZATION


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
#ifdef SYCLomatic_CUSTOMIZATION
  bool IsCudaFile = false;
  int OriArgc = argc;
  SpecifyLanguageInOption = false;
#define DPCT_OPTIONS_IN_CLANG_TOOLING
#define DPCT_OPT_TYPE(...) __VA_ARGS__
#define DPCT_NON_ENUM_OPTION(OPT_TYPE, OPT_VAR, OPTION_NAME, ...)  \
OPT_TYPE OPT_VAR(OPTION_NAME, __VA_ARGS__);
#include "clang/DPCT/DPCTOptions.inc"
#undef DPCT_NON_ENUM_OPTION
#undef DPCT_OPT_TYPE
#undef DPCT_OPTIONS_IN_CLANG_TOOLING

  static llvm::cl::list<std::string> SourcePaths(
      llvm::cl::Positional, llvm::cl::desc("[<source0> ... <sourceN>]"), llvm::cl::ZeroOrMore,
      llvm::cl::cat(Category), llvm::cl::sub(*llvm::cl::AllSubCommands));

  static cl::list<std::string> ArgsBefore(
     "extra-arg-before",
     cl::desc("Additional argument to prepend to the compiler command line.\n"
              "Refer to extra-arg option.\n"),
     cl::cat(Category), cl::sub(*cl::AllSubCommands), llvm::cl::Hidden);
#else
  static cl::opt<std::string> BuildPath("p", cl::desc("Build path"),
                                        cl::Optional, cl::cat(Category),
                                        cl::sub(cl::SubCommand::getAll()));

  static cl::list<std::string> SourcePaths(
      cl::Positional, cl::desc("<source0> [... <sourceN>]"), OccurrencesFlag,
      cl::cat(Category), cl::sub(cl::SubCommand::getAll()));

  static cl::list<std::string> ArgsAfter(
      "extra-arg",
      cl::desc("Additional argument to append to the compiler command line"),
      cl::cat(Category), cl::sub(cl::SubCommand::getAll()));

  static cl::list<std::string> ArgsBefore(
      "extra-arg-before",
      cl::desc("Additional argument to prepend to the compiler command line"),
      cl::cat(Category), cl::sub(cl::SubCommand::getAll()));
#endif // SYCLomatic_CUSTOMIZATION

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
    return llvm::make_error<llvm::StringError>(ErrorMessage,
                                               llvm::inconvertibleErrorCode());
  }

  cl::PrintOptionValues();

  SourcePathList = SourcePaths;
#ifdef SYCLomatic_CUSTOMIZATION
#ifndef _WIN32
  if (std::string(argv[1]) == "--intercept-build" ||
      std::string(argv[1]) == "-intercept-build" ||
      std::string(argv[1]) == "intercept-build") {
    return llvm::Error::success();
  }
#endif
  if(!SourcePathList.empty()) {
    clang::tooling::FormatSearchPath = SourcePaths[0];
  }
  DatabaseStatus ErrCode =
      CannotFindDatabase; // map to MigrationErrorCannotFindDatabase in DPCT
  IsPSpecified = BuildPath.getNumOccurrences();
  for (auto &I : ArgsAfter) {
    if (I.size() > 2 && I.substr(0, 2) == "-x") {
      SpecifyLanguageInOption = true;
    }
  }
#if _WIN32
  VcxprojFilePath = VcxprojFile;
  IsVcxprojfileSpecified = VcxprojFile.getNumOccurrences();
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
        BuildDir, ErrorMessage, ErrCode, CompilationsDir);
    clang::tooling::FormatSearchPath = BuildDir;
  }
#endif
#endif // SYCLomatic_CUSTOMIZATION
  if ((OccurrencesFlag == cl::ZeroOrMore || OccurrencesFlag == cl::Optional) &&
      SourcePathList.empty())
    return llvm::Error::success();
  if (!Compilations) {
    if (!BuildPath.empty()) {
#ifdef SYCLomatic_CUSTOMIZATION
      Compilations = CompilationDatabase::autoDetectFromDirectory(
          BuildPath, ErrorMessage, ErrCode, CompilationsDir);
      clang::tooling::FormatSearchPath = BuildPath;
#else
      Compilations = CompilationDatabase::autoDetectFromDirectory(
          BuildPath, ErrorMessage, ErrCode);
#endif // SYCLomatic_CUSTOMIZATION
    }
#ifdef SYCLomatic_CUSTOMIZATION
    // if neither option "-p" or target source file names exist in the
    // command line, e.g "dpct -in-root=./ -out-root=out", dpct will not
    // continue.
    else if (BuildPath.empty() && SourcePathList.empty()) {
      Compilations = nullptr;
    } else {
      Compilations = CompilationDatabase::autoDetectFromSource(
          SourcePaths[0], ErrorMessage, CompilationsDir);
      clang::tooling::FormatSearchPath = SourcePaths[0];
#else
    else {
      Compilations = CompilationDatabase::autoDetectFromSource(SourcePaths[0],
                                                               ErrorMessage);
#endif // SYCLomatic_CUSTOMIZATION
    }

    if (!Compilations) {
#ifdef SYCLomatic_CUSTOMIZATION
      if (SourcePaths.size() == 0 && !BuildPath.getValue().empty()){
        std::string buf;
        llvm::raw_string_ostream OS(buf);
        OS << "Error while trying to load a compilation database:\n";
        // The ErrCode is set to CannotParseDatabase(-101) from
        // findCompilationDatabaseFromDirectory in autoDetectFromDirectory when
        // database file exists but it cannot be parsed successfully. No other
        // value will set be to ErrCode. So the situation will be either
        // "CannotParseDatabase" or "CannotFindDatabase".

        if (ErrCode == CannotParseDatabase
          /*map to MigrationErrorCannotParseDatabase in DPCT*/) {
          OS << ErrorMessage;
          DoPrintHandle(OS.str(), true);
          return llvm::make_error<DPCTError>(
              CannotParseDatabase
              /*map to MigrationErrorCannotParseDatabase in DPCT*/);
        } else {
          bool IsProcessAllSet = false;
          for (auto &OM : cl::getRegisteredOptions(*cl::TopLevelSubCommand)) {
            cl::Option *O = OM.second;
            if (O->ArgStr == "process-all") {
              IsProcessAllSet = O->getNumOccurrences();
              break;
            }
          }

          if (!IsProcessAllSet) {
            // If no compilation database is found in the dir user specifies, dpct will
            // exit with "code: -19 (Error: Cannot find compilation database)", so
            // the sub misleading msg below should be removed.
            std::string Sub = "Migration initiated without compilation "
                      "database from directory \"" +
                      BuildPath + "\"\n";
            std::string::size_type Pos = ErrorMessage.find(Sub);
            if (Pos != std::string::npos)
              ErrorMessage.erase(Pos, ErrorMessage.length());

            OS << ErrorMessage;
            DoPrintHandle(OS.str(), true);

            return llvm::make_error<DPCTError>(
                CannotFindDatabase
                /*map to MigrationErrorCannotFindDatabase in DPCT*/);
          } else {
            // if -process-all specified, emit a warning msg of no compilation
            // database found, and try to migrate or copy all files from
            // directory specified by -in-root.
            OS << ErrorMessage;
            DoPrintHandle(OS.str(), true);
          }
        }
      } else if (SourcePaths.size() == 1 && BuildPath.getValue().empty()) {
        // need add -x cuda option for not using database
        IsCudaFile = true;
        emitDefaultLanguageWarningIfNecessary(SourcePaths[0],
                                              SpecifyLanguageInOption);
        using namespace llvm::sys;
        SmallString<256> Name = StringRef(SourcePaths[0]);
        StringRef File, Path;
        if (fs::make_absolute(Name) != std::error_code()) {
          std::string buf;
          llvm::raw_string_ostream OS(buf);
          OS << "Could not get absolute path from '" << Name << "'\n";
          DoPrintHandle(OS.str(), true);
        } else {
          File = path::filename(Name);
          Path = path::parent_path(Name);
        }
        std::string buf;
        llvm::raw_string_ostream OS(buf);
        OS << "NOTE: Could not auto-detect compilation database for"
           << " file '" << File << "' in '" << Path
           << "' or any parent directory.\n";
        DoPrintHandle(OS.str(), true);
      } else {
        if (SourcePaths.size() >= 2 && BuildPath.getValue().empty()) {
          // need add -x cuda option for not using database
          IsCudaFile = true;
          for (const auto &SourcePath : SourcePaths) {
            emitDefaultLanguageWarningIfNecessary(SourcePath,
                                                  SpecifyLanguageInOption);
          }
        }
        if (!hasHelpOption(OriArgc, argv)) {
          std::string buf;
          llvm::raw_string_ostream OS(buf);
          if (!BuildPath.getValue().empty())
            OS << "Error while trying to load a compilation database:\n";
          OS << ErrorMessage;
          DoPrintHandle(OS.str(), true);
        }
      }
#else
      llvm::errs() << "Error while trying to load a compilation database:\n"
                   << ErrorMessage << "Running without flags.\n";
#endif // SYCLomatic_CUSTOMIZATION
      Compilations.reset(
          new FixedCompilationDatabase(".", std::vector<std::string>()));
    }
  }
#ifdef SYCLomatic_CUSTOMIZATION
  if (!SourcePathList.empty() &&
              Compilations->getAllCompileCommands().size() != 0) {
    for (auto &Path : SourcePathList) {
      // Add the -x cuda for the case not in database.
      if (Compilations->getCompileCommands(Path).empty()) {
        IsCudaFile = true;
        emitDefaultLanguageWarningIfNecessary(Path, SpecifyLanguageInOption);
        break;
      }
    }
    if (IsCudaFile) {
      Compilations = std::make_unique<ExpandedCompilationDatabase>(
                                                      std::move(Compilations));
    }
  }
#endif
  auto AdjustingCompilations =
      std::make_unique<ArgumentsAdjustingCompilations>(
          std::move(Compilations));
  Adjuster =
      getInsertArgumentAdjuster(ArgsBefore, ArgumentInsertPosition::BEGIN);
  Adjuster = combineAdjusters(
      std::move(Adjuster),
      getInsertArgumentAdjuster(ArgsAfter, ArgumentInsertPosition::END));
#ifdef SYCLomatic_CUSTOMIZATION
  for (auto &I : ArgsAfter) {
    if (I.size() > 2 && I.substr(0, 2) == "-x") {
      IsCudaFile = false;
      Adjuster = combineAdjusters(
          std::move(Adjuster),
          getInsertArgumentAdjuster(I.c_str(), ArgumentInsertPosition::BEGIN));
    } else if(I.size() > 2 && I.substr(0, 2) == "-I") {
      std::string IncPath = I.substr(2);
      const auto StartPos = IncPath.find_first_not_of(" ");
      if (StartPos != std::string::npos)
        IncPath = IncPath.substr(StartPos);
      ExtraIncPathList.push_back(IncPath);
    }
  }

  if (AdjustingCompilations) {
    for (const auto &SourceFile : AdjustingCompilations->getAllFiles()) {
      std::vector<CompileCommand> CompileCommandsForFile =
          AdjustingCompilations->getCompileCommands(SourceFile);
      for (CompileCommand &CompileCommand : CompileCommandsForFile) {
        for (auto &I : CompileCommand.CommandLine) {
          if (I.size() > 2 && I.substr(0, 2) == "-I") {
            std::string IncPath = I.substr(2);
            const auto StartPos = IncPath.find_first_not_of(" ");
            if (StartPos != std::string::npos)
              IncPath = IncPath.substr(StartPos);
            ExtraIncPathList.push_back(IncPath);
          }
        }
      }
    }
  }

  if (IsCudaFile) {
    Adjuster = combineAdjusters(
        std::move(Adjuster),
        getInsertArgumentAdjuster("-xcuda", ArgumentInsertPosition::BEGIN));
  }
#endif // SYCLomatic_CUSTOMIZATION
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

#ifdef SYCLomatic_CUSTOMIZATION
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
#endif // SYCLomatic_CUSTOMIZATION

CommonOptionsParser::CommonOptionsParser(
    int &argc, const char **argv, cl::OptionCategory &Category,
    llvm::cl::NumOccurrencesFlag OccurrencesFlag, const char *Overview) {
  llvm::Error Err = init(argc, argv, Category, OccurrencesFlag, Overview);
  if (Err) {
    llvm::report_fatal_error(
        Twine("CommonOptionsParser: failed to parse command-line arguments. ") +
        llvm::toString(std::move(Err)));
  }
}
