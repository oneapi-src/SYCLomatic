#include "Debug.h"
#include "ASTTraversal.h"
#include "SaveNewFiles.h"

#include <numeric>
#include <unordered_set>

#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace syclct {

// std::string -> file name
// std::array<unsigned int, 3> =>
// array[0]: count LOC(Lines Of Code) to API
// array[1]: count LOC(Lines Of Code) to SYCL
// array[2]: count API not support
std::map<std::string, std::array<unsigned int, 3>> LOCStaticsMap;

int VerboseLevel = NonVerbose;

#ifdef SYCLCT_DEBUG_BUILD // Debug build
bool ShowDebugLevelFlag = false;

static llvm::cl::opt<bool, true>
    ShowDebugLevel("show-debug-levels",
                   llvm::cl::desc("Show syclct debug level hierarchy"),
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
        // Explicitly specified SYCLCT_DEBUG or SYCLCT_DEBUG_WITH_TYPE in syclct
        // falls in
        // this level.
        {"Debug information from SYCLCT_DEBUG/SYCLCT_DEBUG_WITH_TYPE",
         {
             // Elements here are registed dynamically, see DebugTypeRegister
             // and SYCLCT_DEBUG_WTIH_TYPE
         }},
        // Migration rules regards as level 2
        {"Matched migration rules and corresponding information",
         {
// Statically registed elements, no dynamic registation demanding so far
#define RULE(TYPE) #TYPE,
#include "TranslationRules.inc"
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
    SyclctDbgs() << "Level " << i + 1 << " - " << Description << "\n";
    for (const std::string &Str : Set) {
      SyclctDbgs() << Indent << Str << "\n";
    }
  }
}
#endif // Debug build

void DebugInfo::printTranslationRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &TRs) {
  auto print = [&]() {
    SyclctDbgs() << "Migration Rules:\n";

    constexpr char Indent[] = "  ";
    if (TRs.empty()) {
      SyclctDbgs() << Indent << "None\n";
      return;
    }

    size_t NumRules = 0;
    for (auto &TR : TRs) {
      if (auto I = dyn_cast<TranslationRule>(&*TR)) {
        SyclctDbgs() << Indent << I->getName() << "\n";
        ++NumRules;
      }
    }
    SyclctDbgs() << "# of MigrationRules: " << NumRules << "\n";
  };

  if (VerboseLevel == VerboseHigh) {
    print();
  }

  SYCLCT_DEBUG_WITH_TYPE("MigrationRules", print());
}

#ifdef SYCLCT_DEBUG_BUILD
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
    if (auto TR = dyn_cast<TranslationRule>(&*MR)) {
#define RULE(TYPE)                                                             \
  if (TR->getName() == #TYPE) {                                                \
    DEBUG_WITH_TYPE(#TYPE, TR->print(SyclctDbgs()));                           \
    continue;                                                                  \
  }
#include "TranslationRules.inc"
#undef RULE
    }
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<TranslationRule>(&*MR)) {
#define RULE(TYPE)                                                             \
  if (TR->getName() == #TYPE) {                                                \
    DEBUG_WITH_TYPE(#TYPE, TR->printStatistics(SyclctDbgs()));                 \
    continue;                                                                  \
  }
#include "TranslationRules.inc"
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
    DEBUG_WITH_TYPE(#TYPE, TM->print(SyclctDbgs(), Context));                  \
    continue;                                                                  \
  }
#include "Transformations.inc"
#undef TRANSFORMATION
  }

  std::unordered_map<std::string, size_t> NameCountMap;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      TranslatedFiles;
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
    TranslatedFiles[Repl.getFilePath()].emplace(#TYPE);                        \
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
    DEBUG_WITH_TYPE(#TYPE, SyclctDbgs() << "# of replacement <" << #TYPE       \
                                        << ">: " << Numbers << " (" << Numbers \
                                        << "/" << NumRepls << ")\n");          \
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
    if (auto TR = dyn_cast<TranslationRule>(&*MR)) {
      TR->print(SyclctDbgs());
    }
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<TranslationRule>(&*MR)) {
      TR->printStatistics(SyclctDbgs());
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
    SyclctDbgs() << "# of replacement <" << Name << ">: " << Numbers << " ("
                 << Numbers << "/" << NumRepls << ")\n";
  }
}
// End of release Build
#endif

void DebugInfo::printMatchedRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
#ifdef SYCLCT_DEBUG_BUILD // Debug build
  printMatchedRulesDebugImpl(MatchedRules);
#else // Release build
  printMatchedRulesReleaseImpl(MatchedRules);
#endif
}

void DebugInfo::printReplacements(ReplacementFilter &ReplFilter,
                                  clang::ASTContext &Context) {
#ifdef SYCLCT_DEBUG_BUILD // Delease build
  printReplacementsDebugImpl(ReplFilter, Context);
#else // Release build
  printReplacementsReleaseImpl(ReplFilter, Context);
#endif
}

// Log buffer, default size 4096, when running out of memory, dynamic memory
// allocation is handled by SmallVector internally.
static llvm::SmallVector<char, /* default buffer size */ 4096> SyclctLogBuffer;
static llvm::raw_svector_ostream SyclctLogStream(SyclctLogBuffer);

llvm::raw_ostream &SyclctDbgs() { return SyclctLogStream; }

void DebugInfo::ShowStatus(int Status) {
#ifdef SYCLCT_DEBUG_BUILD // Debug build
  if (ShowDebugLevelFlag) {
    ShowDebugLevels();
  }
#endif // Debug build

  std::string StatusString;
  switch (Status) {
  case MigrationSucceeded:
    StatusString = "Migration succeed";
    break;
  case MigrationNoCodeChangeHappen:
    StatusString = "Migration keep input file unchanged";
    break;
  case MigrationSkipped:
    StatusString = "Migration skip for input file";
    break;
  case MigrationError:
    StatusString = "Migration error happen";
    break;
  default:
    syclct_unreachable("no valid stats");
  }

  if (Status != 0) {
    SyclctDbgs() << "Syclct exited with code: " << Status << " ("
                 << StatusString << ")\n";
  }

  llvm::dbgs() << SyclctLogStream.str() << "\n";
  return;
}

} // namespace syclct
} // namespace clang
