#include "Debug.h"
#include "ASTTraversal.h"

#include <unordered_set>

namespace clang {
namespace syclct {

bool ShowDebugLevelFlag = false;

static llvm::cl::opt<bool, true>
    ShowDebugLevel("show-debug-levels",
                   llvm::cl::desc("Show syclct debug level hierarchy"),
                   llvm::cl::Hidden, llvm::cl::location(ShowDebugLevelFlag));

enum class DebugLevel : int { Low = 1, Median, High };

static DebugLevel DbgLevel = DebugLevel::High;

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
        // Translation rules regards as level 2
        {"Matched translation rules and corresponding information",
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
  if (IsReleaseBuild) {
    return;
  }

  std::unordered_set<std::string> &Level1Set = Levels[0].second;
  Level1Set.emplace(type);
}

static void ShowDebugLevels() {
  constexpr char Indent[] = "  ";
  for (size_t i = 0; i < Levels.size(); ++i) {
    const std::string &Description = Levels[i].first;
    const std::unordered_set<std::string> &Set = Levels[i].second;
    llvm::dbgs() << "Level " << i + 1 << " - " << Description << "\n";
    for (const std::string &Str : Set) {
      llvm::dbgs() << Indent << Str << "\n";
    }
  }
}

void DebugInfo::ShowStatistics(int status) {
  if (IsDebugBuild && ShowDebugLevelFlag) {
    ShowDebugLevels();
  }

  if (status != 0) {
    llvm::dbgs() << "Syclct failed with code: " << status << "\n";
  }
  return;
}

void DebugInfo::printTranslationRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &TRs) {
  if (IsReleaseBuild) {
    return;
  }

  auto print = [&]() {
    llvm::dbgs() << "Translation Rules:\n";

    constexpr char Indent[] = "  ";
    if (TRs.empty()) {
      llvm::dbgs() << Indent << "None\n";
      return;
    }

    size_t NumRules = 0;
    for (auto &TR : TRs) {
      if (auto I = dyn_cast<TranslationRule>(&*TR)) {
        llvm::dbgs() << Indent << I->getName() << "\n";
        ++NumRules;
      }
    }
    llvm::dbgs() << "# of TranslationRules: " << NumRules << "\n";
  };

  SYCLCT_DEBUG_WITH_TYPE("TranslationRules", print());
}

static void printMatchedRulesReleaseImpl(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
  // TODO
}

static void printMatchedRulesDebugImpl(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
  // Debug level lower than "Median" doesn't show translation rules' information
  if (IsDebugBuild && DbgLevel < DebugLevel::Median) {
    return;
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<TranslationRule>(&*MR)) {
#define RULE(TYPE)                                                             \
  if (TR->getName() == #TYPE) {                                                \
    DEBUG_WITH_TYPE(#TYPE, TR->print(llvm::dbgs()));                           \
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
    DEBUG_WITH_TYPE(#TYPE, TR->printStatistics(llvm::dbgs()));                 \
    continue;                                                                  \
  }
#include "TranslationRules.inc"
#undef RULE
    }
  }
}

void DebugInfo::printMatchedRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
  if (IsReleaseBuild) {
    printMatchedRulesReleaseImpl(MatchedRules);
  } else {
    printMatchedRulesDebugImpl(MatchedRules);
  }
}

static void printReplacementsReleaseImpl(ReplacementFilter &ReplFilter,
                                         clang::ASTContext &Context) {
  // TODO
}

static void printReplacementsDebugImpl(ReplacementFilter &ReplFilter,
                                       clang::ASTContext &Context) {
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
    DEBUG_WITH_TYPE(#TYPE, TM->print(llvm::dbgs(), Context));                  \
    continue;                                                                  \
  }
#include "Transformations.inc"
#undef TRANSFORMATION
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

  const size_t NumRepls =
      std::accumulate(NameCountMap.begin(), NameCountMap.end(), 0,
                      [](size_t a, const std::pair<std::string, size_t> obj) {
                        return a + obj.second;
                      });
  for (const auto &Pair : NameCountMap) {
    const std::string &Name = Pair.first;
    const size_t &Numbers = Pair.second;
#define TRANSFORMATION(TYPE)                                                   \
  if (Name == #TYPE) {                                                         \
    DEBUG_WITH_TYPE(#TYPE, llvm::dbgs() << "# of replacement <" << #TYPE       \
                                        << ">: " << Numbers << " (" << Numbers \
                                        << "/" << NumRepls << ")\n");          \
    continue;                                                                  \
  }
#include "Transformations.inc"
#undef TRANSFORMATION
  }
}

void DebugInfo::printReplacements(ReplacementFilter &ReplFilter,
                                  clang::ASTContext &Context) {
  if (IsReleaseBuild) {
    printReplacementsReleaseImpl(ReplFilter, Context);
  } else {
    printReplacementsDebugImpl(ReplFilter, Context);
  }
}

} // namespace syclct
} // namespace clang
