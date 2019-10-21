//===--- Debug.cpp--------- --------------------------------*- C++ -*---===//
////
//// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
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

#ifdef DPCT_DEBUG_BUILD // Debug build
bool ShowDebugLevelFlag = false;

static llvm::cl::opt<bool, true>
    ShowDebugLevel("show-debug-levels",
                   llvm::cl::desc("Show dpct debug level hierarchy"),
                   llvm::cl::Hidden, llvm::cl::location(ShowDebugLevelFlag));

enum class DebugLevel : int { Low = 1, Median, High };

static DebugLevel DbgLevel = DebugLevel::Low;

struct DebugLevelOpt {
  void operator=(const int &Val) {
    llvm::DebugFlag = true;
    const DebugLevel InputVal = static_cast<DebugLevel>(Val);
    if (InputVal < DebugLevel::Low) {
      DbgLevel = DebugLevel::Low;
    } else if (InputVal > DebugLevel::High) {
      DbgLevel = DebugLevel::High;
    } else {
      DbgLevel = InputVal;
    }
  }
};

static DebugLevelOpt DebugLevelOptLoc;

static llvm::cl::opt<DebugLevelOpt, true, llvm::cl::parser<int>>
    DebugLevelSelector(
        "debug-level",
        llvm::cl::desc("Specify debug level from 1 to 3 [default 3]"),
        llvm::cl::Hidden, llvm::cl::location(DebugLevelOptLoc));

static std::vector<std::pair<std::string, std::unordered_set<std::string>>>
    Levels = {
        // Debug informations not in level 2 and level 3.
        // Explicitly specified DPCT_DEBUG or DPCT_DEBUG_WITH_TYPE in dpct
        // falls in
        // this level.
        {"Debug information from DPCT_DEBUG/DPCT_DEBUG_WITH_TYPE",
         {
             // Elements here are registed dynamically, see DebugTypeRegister
             // and DPCT_DEBUG_WTIH_TYPE
         }},
        // Migration rules regards as level 2
        {"Matched migration rules and corresponding information",
         {
// Statically registed elements, no dynamic registation demanding so far
#define RULE(TYPE) #TYPE,
#include "MigrationRules.inc"
#undef RULE
         }},
        // TextModifications regards as level 3
        {"Detailed information of replacements",
         {
// Statically registed elements, no dynamic registation demanding so far
#define TRANSFORMATION(TYPE) #TYPE,
#include "Transformations.inc"
#undef TRANSFORMATION
         }}};

DebugTypeRegister::DebugTypeRegister(const std::string &type) {
  std::unordered_set<std::string> &Level1Set = Levels[0].second;
  Level1Set.emplace(type);
}

static void ShowDebugLevels() {
  constexpr char Indent[] = "  ";
  for (size_t i = 0; i < Levels.size(); ++i) {
    const std::string &Description = Levels[i].first;
    const std::unordered_set<std::string> &Set = Levels[i].second;
    DpctDiags() << "Level " << i + 1 << " - " << Description << "\n";
    for (const std::string &Str : Set) {
      DpctDiags() << Indent << Str << "\n";
    }
  }
}
#endif // Debug build

void DebugInfo::printMigrationRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &TRs) {
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

  DPCT_DEBUG_WITH_TYPE("MigrationRules", print());
}

#ifdef DPCT_DEBUG_BUILD
// Start of debug build
static void printMatchedRulesDebugImpl(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
  if (VerboseLevel == VerboseLow) {
    DbgLevel = DebugLevel::Low;
  } else if (VerboseLevel == VerboseHigh) {
    llvm::DebugFlag = true;
    DbgLevel = DebugLevel::High;
  }

  // Debug level lower than "Median" doesn't show migration rules' information
  if (DbgLevel < DebugLevel::Median) {
    return;
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<MigrationRule>(&*MR)) {
#define RULE(TYPE)                                                             \
  if (TR->getName() == #TYPE) {                                                \
    DEBUG_WITH_TYPE(#TYPE, TR->print(DpctDiags()));                            \
    continue;                                                                  \
  }
#include "MigrationRules.inc"
#undef RULE
    }
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<MigrationRule>(&*MR)) {
#define RULE(TYPE)                                                             \
  if (TR->getName() == #TYPE) {                                                \
    DEBUG_WITH_TYPE(#TYPE, TR->printStatistics(DpctDiags()));                  \
    continue;                                                                  \
  }
#include "MigrationRules.inc"
#undef RULE
    }
  }
}

static void printReplacementsDebugImpl(ReplacementFilter &ReplFilter,
                                       clang::ASTContext &Context) {
  if (VerboseLevel == VerboseLow) {
    DbgLevel = DebugLevel::Low;
  } else if (VerboseLevel == VerboseHigh) {
    llvm::DebugFlag = true;
    DbgLevel = DebugLevel::High;
  }

  // Debug level lower than "High" doesn't show detailed replacements'
  // information
  if (DbgLevel < DebugLevel::High) {
    return;
  }

  for (const ExtReplacement &Repl : ReplFilter) {
    const TextModification *TM = nullptr;
#define TRANSFORMATION(TYPE)                                                   \
  TM = Repl.getParentTM();                                                     \
  if (TM && TMID::TYPE == TM->getID()) {                                       \
    DEBUG_WITH_TYPE(#TYPE, TM->print(DpctDiags(), Context));                   \
    continue;                                                                  \
  }
#include "Transformations.inc"
#undef TRANSFORMATION
  }

  std::unordered_map<std::string, size_t> NameCountMap;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      MigratedFiles;
  for (const ExtReplacement &Repl : ReplFilter) {
    const TextModification *TM = nullptr;
#define TRANSFORMATION(TYPE)                                                   \
  TM = Repl.getParentTM();                                                     \
  if (TM && TMID::TYPE == TM->getID()) {                                       \
    if (NameCountMap.count(#TYPE) == 0) {                                      \
      NameCountMap.emplace(std::make_pair(#TYPE, 1));                          \
    } else {                                                                   \
      ++NameCountMap[#TYPE];                                                   \
    }                                                                          \
    MigratedFiles[Repl.getFilePath()].emplace(#TYPE);                          \
    continue;                                                                  \
  }
#include "Transformations.inc"
#undef TRANSFORMATION
  }

  if (NameCountMap.empty()) {
    return;
  }

  const size_t NumRepls = std::accumulate(
      NameCountMap.begin(), NameCountMap.end(), 0,
      [](const size_t &a, const std::pair<std::string, size_t> &obj) {
        return a + obj.second;
      });
  for (const auto &Pair : NameCountMap) {
    const std::string &Name = Pair.first;
    const size_t &Numbers = Pair.second;
#define TRANSFORMATION(TYPE)                                                   \
  if (Name == #TYPE) {                                                         \
    DEBUG_WITH_TYPE(#TYPE, DpctDiags() << "# of replacement <" << #TYPE        \
                                       << ">: " << Numbers << " (" << Numbers  \
                                       << "/" << NumRepls << ")\n");           \
    continue;                                                                  \
  }
#include "Transformations.inc"
#undef TRANSFORMATION
  }
}

// End of debug build
#else
// Start of release build
static void printMatchedRulesReleaseImpl(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
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

static void printReplacementsReleaseImpl(ReplacementFilter &ReplFilter,
                                         clang::ASTContext &Context) {
  if (VerboseLevel < VerboseHigh) {
    return;
  }

  std::unordered_map<std::string, size_t> NameCountMap;
  for (const ExtReplacement &Repl : ReplFilter) {
    const TextModification *TM = nullptr;
#define TRANSFORMATION(TYPE)                                                   \
  TM = Repl.getParentTM();                                                     \
  if (TM && TMID::TYPE == TM->getID()) {                                       \
    if (NameCountMap.count(#TYPE) == 0) {                                      \
      NameCountMap.emplace(std::make_pair(#TYPE, 1));                          \
    } else {                                                                   \
      ++NameCountMap[#TYPE];                                                   \
    }                                                                          \
    continue;                                                                  \
  }
#include "Transformations.inc"
#undef TRANSFORMATION
  }

  if (NameCountMap.empty()) {
    return;
  }

  const size_t NumRepls =
      std::accumulate(NameCountMap.begin(), NameCountMap.end(), 0,
                      [](size_t a, const std::pair<std::string, size_t> obj) {
                        return a + obj.second;
                      });
  for (const auto &Pair : NameCountMap) {
    const std::string &Name = Pair.first;
    const size_t &Numbers = Pair.second;
    DpctDiags() << "# of replacement <" << Name << ">: " << Numbers << " ("
                << Numbers << "/" << NumRepls << ")\n";
  }
}
// End of release Build
#endif

void DebugInfo::printMatchedRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
#ifdef DPCT_DEBUG_BUILD // Debug build
  printMatchedRulesDebugImpl(MatchedRules);
#else // Release build
  printMatchedRulesReleaseImpl(MatchedRules);
#endif
}

void DebugInfo::printReplacements(ReplacementFilter &ReplFilter,
                                  clang::ASTContext &Context) {
#ifdef DPCT_DEBUG_BUILD // Delease build
  printReplacementsDebugImpl(ReplFilter, Context);
#else // Release build
  printReplacementsReleaseImpl(ReplFilter, Context);
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
std::string getDpctStatsStr() { return DpctStatsStream.str(); }
std::string getDpctDiagsStr() { return DpctDiagsStream.str(); }
std::string getDpctTermStr() { return DpctTermStream.str(); }

void DebugInfo::ShowStatus(int Status) {
#ifdef DPCT_DEBUG_BUILD // Debug build
  if (ShowDebugLevelFlag) {
    ShowDebugLevels();
  }
#endif // Debug build

  std::string StatusString;
  switch (Status) {
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
      " only alphabets, digits and underscores are allowed";
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

} // namespace dpct
} // namespace clang
