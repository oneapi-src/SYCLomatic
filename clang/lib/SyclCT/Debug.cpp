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

static std::vector<std::unordered_set<std::string>> Levels = {
    // Debug informations not in level 2 and level 3.
    // Explicitly specified SYCLCT_DEBUG or DEBUG_WITH_TYPE in syclct falls in
    // this level.
    {},
    {
// Translation rules regards as level 2
#define RULE(TYPE) #TYPE,
#include "TranslationRules.inc"
#undef RULE
    },
    {
// TextModifications regards as level 3
#define TRANSFORMATION(TYPE) #TYPE,
#include "Transformations.inc"
#undef TRANSFORMATION
    }};

DebugTypeRegister::DebugTypeRegister(const std::string &type) {
  if (IsReleaseBuild) {
    return;
  }

  Levels[0].emplace(type);
}

static void ShowDebugLevels() {
  constexpr char Indent[] = "  ";
  for (size_t i = 0; i < Levels.size(); ++i) {
    llvm::dbgs() << "Level " << i + 1 << "\n";
    for (const std::string &Str : Levels[i]) {
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

    for (auto &TR : TRs) {
      if (auto I = dyn_cast<TranslationRule>(&*TR)) {
        llvm::dbgs() << Indent << I->getName() << "\n";
      }
    }
  };

  SYCLCT_DEBUG_WITH_TYPE("TranslationRules", print());
}

void DebugInfo::printMatchedRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
  if (IsReleaseBuild) {
    return;
  }

  // Debug level lower than "Median" doesn't show translation rules' information
  if (DbgLevel < DebugLevel::Median) {
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
}

void DebugInfo::printReplacements(ReplacementFilter &ReplFilter,
                                  clang::ASTContext &Context) {
  if (IsReleaseBuild) {
    return;
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
    DEBUG_WITH_TYPE(#TYPE, TM->print(llvm::dbgs(), Context));                  \
    continue;                                                                  \
  }
#include "Transformations.inc"
#undef TRANSFORMATION
  }
}

} // namespace syclct
} // namespace clang
