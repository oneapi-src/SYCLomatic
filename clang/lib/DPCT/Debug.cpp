//===--- Debug.cpp--------- --------------------------------*- C++ -*---===//
////
//// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
////
//// The information and source code contained herein is the exclusive
//// property of Intel Corporation and may not be disclosed, examined
//// or reproduced in whole or in part without explicit written authorization
//// from the company.
////
////===-----------------------------------------------------------------===//
#include "Debug.h"
#include "ASTTraversal.h"
#include "SaveNewFiles.h"

#include <numeric>
#include <unordered_set>

#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace dpct {

// std::string -> file name
// std::array<unsigned int, 3> =>
// array[0]: count LOC(Lines Of Code) to API
// array[1]: count LOC(Lines Of Code) to SYCL
// array[2]: count API not support
std::unordered_map<std::string, std::array<unsigned int, 3>> LOCStaticsMap;

// std::string -> APIName ,types information
// unsigned int -> Times met
std::map<std::string, unsigned int> SrcAPIStaticsMap;

int VerboseLevel = NonVerbose;

void DebugInfo::printMigrationRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &TRs) {
#ifdef DPCT_DEBUG_BUILD
  auto print = [&]() {
    DpctDiags() << "Migration Rules:\n";

    constexpr char Indent[] = "  ";
    if (TRs.empty()) {
      DpctDiags() << Indent << "None\n";
      return;
    }

    size_t NumRules = 0;
    for (auto &TR : TRs) {
      if (auto I = dyn_cast<MigrationRule>(&*TR)) {
        DpctDiags() << Indent << I->getName() << "\n";
        ++NumRules;
      }
    }
    DpctDiags() << "# of MigrationRules: " << NumRules << "\n";
  };

  if (VerboseLevel > NonVerbose) {
    print();
  }

#endif // DPCT_DEBUG_BUILD
}

#ifdef DPCT_DEBUG_BUILD
// Start of debug build
static void printMatchedRulesDebugImpl(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
  // Verbose level lower than "High" doesn't show migration rules' information
  if (VerboseLevel < VerboseHigh) {
    return;
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<MigrationRule>(&*MR)) {
      TR->print(DpctDiags());
    }
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<MigrationRule>(&*MR)) {
      TR->printStatistics(DpctDiags());
    }
  }
}

static void printReplacementsDebugImpl(const TransformSetTy &TS,
                                       ASTContext &Context) {
  // Verbos level lower than "High" doesn't show detailed replacements'
  // information
  if (VerboseLevel < VerboseHigh) {
    return;
  }

  for (auto &TM : TS) {
    TM->print(DpctDiags(), Context);
  }

  std::unordered_map<int, size_t> NameCountMap;
  for (auto &TM : TS) {
    ++(NameCountMap.insert(std::make_pair((int)TM->getID(), 0)).first->second);
  }

  if (NameCountMap.empty())
    return;

  const size_t NumRepls =
      std::accumulate(NameCountMap.begin(), NameCountMap.end(), 0,
                      [](const size_t &a, const std::pair<int, size_t> &obj) {
                        return a + obj.second;
                      });
  for (const auto &Pair : NameCountMap) {
    auto &ID = Pair.first;
    auto &Numbers = Pair.second;
    DpctDiags() << "# of replacement <" << TextModification::TMNameMap.at((int)ID)
                << ">: " << Numbers << " (" << Numbers << "/" << NumRepls
                << ")\n";
  }
}

// End of debug build
#endif

void DebugInfo::printMatchedRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
#ifdef DPCT_DEBUG_BUILD // Debug build
  printMatchedRulesDebugImpl(MatchedRules);
#endif
}

void DebugInfo::printReplacements(const TransformSetTy &TS,
                                  ASTContext &Context) {
#ifdef DPCT_DEBUG_BUILD // Debug build
  printReplacementsDebugImpl(TS, Context);
#endif
}

// Log buffer, default size 4096, when running out of memory, dynamic memory
// allocation is handled by SmallVector internally.
static llvm::SmallVector<char, /* default buffer size */ 4096> DpctLogBuffer;
static llvm::raw_svector_ostream DpctLogStream(DpctLogBuffer);
static llvm::SmallVector<char, /* default buffer size */ 4096> DpctStatsBuffer;
static llvm::raw_svector_ostream DpctStatsStream(DpctStatsBuffer);
static llvm::SmallVector<char, /* default buffer size */ 4096> DpctDiagsBuffer;
static llvm::raw_svector_ostream DpctDiagsStream(DpctDiagsBuffer);

static llvm::SmallVector<char, /* default buffer size */ 4096> DpctTermBuffer;
static llvm::raw_svector_ostream DpctTermStream(DpctTermBuffer);

llvm::raw_ostream &DpctLog() { return DpctLogStream; }
llvm::raw_ostream &DpctStats() { return DpctStatsStream; }
llvm::raw_ostream &DpctDiags() { return DpctDiagsStream; }
llvm::raw_ostream &DpctTerm() { return DpctTermStream; }
std::string getDpctStatsStr() { return DpctStatsStream.str().str(); }
std::string getDpctDiagsStr() { return DpctDiagsStream.str().str(); }
std::string getDpctTermStr() { return DpctTermStream.str().str(); }

void DebugInfo::ShowStatus(int Status) {

  std::string StatusString;
  switch (Status) {
  case MigrationSuccessExpParingOrRuntimeErr:
    StatusString = "Migration process completed, except parsing/runtime error(s)";
    break;
  case MigrationSucceeded:
    StatusString = "Migration process completed";
    break;
  case MigrationNoCodeChangeHappen:
    StatusString = "Migration not necessary";
    break;
  case MigrationSkipped:
    StatusString = "Some migration rules were skipped";
    break;
  case MigrationError:
    StatusString = "Migration error happened";
    break;
  case MigrationSaveOutFail:
    StatusString = "Error: Saving output file(s)";
    break;
  case MigrationErrorInvalidSDKPath:
    StatusString = "Error: Path for CUDA header files is invalid or "
                   "not available. Specify with --cuda-include-path";
    break;
  case MigrationErrorInvalidInRootOrOutRoot:
    StatusString = "Error: Invalid --in-root or --out-root path";
    break;
  case MigrationErrorInvalidInRootPath:
    StatusString = "Error: Invalid --in-root path";
    break;
  case MigrationErrorInvalidReportArgs:
    StatusString = "Error: Bad value provided for report option(s)";
    break;
  case MigrationErrorInvalidWarningID:
    StatusString = "Error: Invalid warning ID or range; "
                   "valid warning IDs range from " +
                   std::to_string((size_t)Warnings::BEGIN) + " to " +
                   std::to_string((size_t)Warnings::END - 1);
    break;
  case MigrationOptionParsingError:
    StatusString = "Option parsing error,"
                   " run 'dpct --help' to see supported options and values";
    break;
  case MigrationErrorPathTooLong:
#if defined(_WIN32)
    StatusString = "Error: Path is too long; should be less than _MAX_PATH (" +
                   std::to_string(_MAX_PATH) + ")";
#else
    StatusString = "Error: Path is too long; should be less than PATH_MAX (" +
                   std::to_string(PATH_MAX) + ")";
#endif
    break;
  case MigrationErrorFileParseError:
    StatusString = "Error: Cannot parse input file(s)";
    break;
  case MigrationErrorCannotFindDatabase:
    StatusString = "Error: Cannot find compilation database";
    break;
  case MigrationErrorCannotParseDatabase:
    StatusString = "Error: Cannot parse compilation database";
    break;
  case MigrationErrorNoExplicitInRoot:
    StatusString = "Error: --process-all option requires --in-root to be "
                   "specified explicitly. Specify --in-root.";
    break;
  case MigrationErrorSpecialCharacter:
    StatusString = "Error: Prefix contains special characters;"
                   " only alphabetical characters, digits and underscore "
                   "character are allowed";
    break;
  case MigrationErrorNameTooLong:
#if defined(_WIN32)
    StatusString =
        "Error: File name is too long; should be less than _MAX_FNAME (" +
        std::to_string(_MAX_FNAME) + ")";
#else
    StatusString =
        "Error: File name is too long; should be less than NAME_MAX (" +
        std::to_string(NAME_MAX) + ")";
#endif
    break;
  case MigrationErrorPrefixTooLong:
    StatusString = "Error: Prefix is too long; should be less than 128";
    break;
  case MigrationErrorNoFileTypeAvail:
      StatusString = "Error: File Type not available for input file";
      break;
  default:
    DpctLog() << "Unknown error\n";
    exit(-1);
  }

  if (Status != 0) {
    DpctLog() << "dpct exited with code: " << Status << " (" << StatusString
              << ")\n";
  }

  llvm::dbgs() << DpctLogStream.str() << "\n";
  return;
}
// Currently, set IsPrintOnNormal false only at the place where messages about
// start and end of file parsing are produced,
//.i.e in the place "lib/Tooling:int ClangTool::run(ToolAction *Action)".
void PrintMsg(const std::string &Msg, bool IsPrintOnNormal) {
  if (!OutputFile.empty()) {
    //  Redirects stdout/stderr output to <file>
    DpctTerm() << Msg;
  }

  switch (OutputVerbosity) {
  case OutputVerbosityLev::detailed:
  case OutputVerbosityLev::diagnostics:
    llvm::outs() << Msg;
    break;
  case OutputVerbosityLev::normal:
    if (IsPrintOnNormal) {
      llvm::outs() << Msg;
    }
    break;
  case OutputVerbosityLev::silent:
  default:
    break;
  }
}

} // namespace dpct
} // namespace clang
