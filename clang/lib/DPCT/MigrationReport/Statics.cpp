//===--------------- Statics.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "MigrationReport/Statics.h"
#include "ASTTraversal.h"
#include "RulesInclude/InclusionHeaders.h"

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

int VerboseLevel = VL_NonVerbose;

void StaticsInfo::printMigrationRules(
    const std::vector<std::unique_ptr<MigrationRule>> &TRs) {
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
      DpctDiags() << Indent << TR->getName() << "\n";
      ++NumRules;
    }
    DpctDiags() << "# of MigrationRules: " << NumRules << "\n";
  };

  if (VerboseLevel > VL_NonVerbose) {
    print();
  }

#endif // DPCT_DEBUG_BUILD
}

#ifdef DPCT_DEBUG_BUILD
// Start of debug build
static void printMatchedRulesStaticsImpl(
    const std::vector<std::unique_ptr<MigrationRule>> &MatchedRules) {
  // Verbose level lower than "High" doesn't show migration rules' information
  if (VerboseLevel < VL_VerboseHigh) {
    return;
  }

  for (auto &MR : MatchedRules) {
    MR->print(DpctDiags());
  }

  for (auto &MR : MatchedRules) {
    MR->printStatistics(DpctDiags());
  }
}

static void printReplacementsStaticsImpl(const TransformSetTy &TS,
                                         ASTContext &Context) {
  // Verbose level lower than "High" doesn't show detailed replacements'
  // information
  if (VerboseLevel < VL_VerboseHigh) {
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
    DpctDiags() << "# of replacement <"
                << TextModification::TMNameMap.at((int)ID) << ">: " << Numbers
                << " (" << Numbers << "/" << NumRepls << ")\n";
  }
}

// End of debug build
#endif

void StaticsInfo::printMatchedRules(
    const std::vector<std::unique_ptr<MigrationRule>> &MatchedRules) {
#ifdef DPCT_DEBUG_BUILD // Debug build
  printMatchedRulesStaticsImpl(MatchedRules);
#endif
}

void StaticsInfo::printReplacements(const TransformSetTy &TS,
                                    ASTContext &Context) {
#ifdef DPCT_DEBUG_BUILD // Debug build
  printReplacementsStaticsImpl(TS, Context);
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

llvm::raw_ostream &DpctStats() { return DpctStatsStream; }
llvm::raw_ostream &DpctDiags() { return DpctDiagsStream; }
llvm::raw_ostream &DpctTerm() { return DpctTermStream; }
llvm::raw_ostream &DpctLog() { return DpctLogStream; }
llvm::raw_ostream &DpctDebugs() {
#ifdef DPCT_DEBUG_BUILD
  return llvm::errs();
#else
  return llvm::nulls();
#endif
}
std::string getDpctStatsStr() { return DpctStatsStream.str().str(); }
std::string getDpctDiagsStr() { return DpctDiagsStream.str().str(); }
std::string getDpctTermStr() { return DpctTermStream.str().str(); }
std::string getDpctLogStr() { return DpctLogStream.str().str(); }

// Currently, set IsPrintOnNormal false only at the place where messages about
// start and end of file parsing are produced,
//.i.e in the place "lib/Tooling:int ClangTool::run(ToolAction *Action)".
void PrintMsg(const std::string &Msg, bool IsPrintOnNormal) {
  if (!OutputFile.empty()) {
    //  Redirects stdout/stderr output to <file>
    DpctTerm() << Msg;
  }

  switch (OutputVerbosity) {
  case OutputVerbosityLevel::OVL_Detailed:
  case OutputVerbosityLevel::OVL_Diagnostics:
    llvm::outs() << Msg;
    return;
  case OutputVerbosityLevel::OVL_Normal:
    if (IsPrintOnNormal) {
      llvm::outs() << Msg;
    }
    return;
  case OutputVerbosityLevel::OVL_Silent:
    return;
  }
  DpctDebugs() << "[OutputVerbosityLevel] Unexpected value: "
               << static_cast<std::underlying_type_t<OutputVerbosityLevel>>(
                      OutputVerbosity.getValue())
               << "\n";
  assert(0);
}

enum {
  LowEffort = static_cast<unsigned>(EffortLevel::EL_Low),
  MediumEffort = static_cast<unsigned>(EffortLevel::EL_Medium),
  HighEffort = static_cast<unsigned>(EffortLevel::EL_High),
  EffortNum = static_cast<unsigned>(EffortLevel::EL_NUM),
  NoEffort = EffortNum,
  CounterNum,
};

class LineStream {
  const static StringRef NL;
  llvm::raw_ostream &OS;

public:
  LineStream(llvm::raw_ostream &OS, unsigned Indent) : OS(OS) {
    OS.indent(Indent);
  }
  ~LineStream() { OS << NL; }

  template <class T> LineStream &operator<<(T &&Input) {
    OS << std::forward<T>(Input);
    return *this;
  }
};
const StringRef LineStream::NL = getNL();

struct AnalysisModeSummary {
  static const unsigned IndentIncremental = 2;
  static const unsigned NumberWidth = 3;

  StringRef Name;
  unsigned Total = 0;
  unsigned Counter[CounterNum] = {0};

  AnalysisModeSummary(StringRef Name) : Name(Name) {}

  AnalysisModeSummary &operator+=(const AnalysisModeSummary &Other) {
    Total += Other.Total;
    Counter[HighEffort] += Other.Counter[HighEffort];
    Counter[MediumEffort] += Other.Counter[MediumEffort];
    Counter[LowEffort] += Other.Counter[LowEffort];
    Counter[NoEffort] += Other.Counter[NoEffort];
    return *this;
  }

  void dump(llvm::raw_ostream &OS, unsigned Indent) const {
    LineStream(OS, Indent) << llvm::raw_ostream::Colors::BLUE << Name << ':'
                           << llvm::raw_ostream::Colors::RESET;
    Indent += IndentIncremental;
    printClassify(OS, Indent, "will be automatically migrated", NoEffort,
                  LowEffort, MediumEffort);
    printClassify(OS, Indent, "will not be automatically migrated", HighEffort);
  }

private:
  void printClassifyMsg(llvm::raw_ostream &OS, unsigned Indent,
                        StringRef ClassifyMsg, unsigned ClassifyNum) const {
    LineStream(OS, Indent) << '+'
                           << llvm::format_decimal(ClassifyNum, NumberWidth)
                           << " lines of code ("
                           << llvm::format_decimal(
                                  Total ? (ClassifyNum * 100) / Total : 0, 3)
                           << "%) " << ClassifyMsg << '.';
  }
  template <class... LevelTys>
  unsigned sum(unsigned FirstLevel, LevelTys... RestLevels) const {
    return Counter[FirstLevel] + sum(RestLevels...);
  }
  unsigned sum(unsigned Level) const { return Counter[Level]; }

  template <class... LevelTys>
  void printClassify(llvm::raw_ostream &OS, unsigned Indent,
                     StringRef ClassifyMsg, LevelTys... Levels) const {
    printClassifyMsg(OS, Indent, ClassifyMsg, sum(Levels...));
    printLevel(OS, Indent + IndentIncremental, Levels...);
  }
  template <class... LevelTys>
  void printLevel(llvm::raw_ostream &OS, unsigned Indent, unsigned FirstLevel,
                  LevelTys... RestLevels) const {
    printLevel(OS, Indent, FirstLevel);
    printLevel(OS, Indent, RestLevels...);
  }
  void printLevel(llvm::raw_ostream &OS, unsigned Indent,
                  unsigned Level) const {
    const static std::string PostMsgs[] = {
        "High manual effort for code fixing",
        "Medium manual effort for code fixing",
        "Low manual effort for checking and code fixing", "No manual effort"};
    LineStream(OS, Indent) << '-'
                           << llvm::format_decimal(Counter[Level], NumberWidth)
                           << " APIs/Types - " << PostMsgs[Level] << '.';
  }
};

class AnalysisModeStats {
  static const std::string LastMsg;
  static llvm::StringMap<AnalysisModeStats> AnalysisModeStaticsMap;

  static struct {
    uint8_t Flag = 0;
    bool operator[](uint8_t Idx) { return Flag & (1 << Idx); }
  } DependencyBits;

  struct EffortLevelWrap {
    unsigned EL;
    EffortLevelWrap() : EL(NoEffort) {}
    EffortLevelWrap &operator=(EffortLevel Other) {
      if (auto O = static_cast<unsigned>(Other); EL > O)
        EL = O;
      return *this;
    }

    operator unsigned() const { return EL; }
  };

  std::map<unsigned, EffortLevelWrap> FileEffortsMap;

  AnalysisModeSummary getSummary(StringRef Name) const {
    AnalysisModeSummary Summary(Name);
    for (auto &Entry : FileEffortsMap) {
      ++Summary.Counter[Entry.second];
      ++Summary.Total;
    }
    return Summary;
  }

  void recordEffort(unsigned Offset, EffortLevel Level) {
    FileEffortsMap[Offset] = Level;
  }
  void recordApisOrTypes(unsigned Offset) { (void)FileEffortsMap[Offset]; }

public:
  static void dump(llvm::raw_ostream &OS) {
    static const unsigned Indent = 0;
    AnalysisModeSummary Total("Total Project");
    for (const auto &Entry : AnalysisModeStaticsMap) {
      auto Summary = Entry.second.getSummary(Entry.first());
      Summary.dump(OS, Indent);
      Total += Summary;
    }
    if (AnalysisModeStaticsMap.size() > 1)
      Total.dump(OS, Indent);
    if (DependencyBits.Flag) {
      LineStream(OS, Indent) << llvm::raw_ostream::Colors::BLUE
                             << "Library Dependencies of SYCL Project:"
                             << llvm::raw_ostream::Colors::RESET;
      struct DependenceNames {
        llvm::StringRef Strs[static_cast<uint8_t>(LibraryDependencies::NUMS)];
        DependenceNames() noexcept {
#define LIBRARY(LIBNAME, LIBDESC, ...)                                         \
  Strs[static_cast<uint8_t>(LibraryDependencies::LD_##LIBNAME)] = LIBDESC;
#include "Libraries.inc"
        }
      } Names;
      auto EntryIndent = Indent + AnalysisModeSummary::IndentIncremental;
      for (unsigned i = 0; i < static_cast<uint8_t>(LibraryDependencies::NUMS);
           ++i) {
        if (DependencyBits[i])
          LineStream(OS, EntryIndent) << "- The Intel " << Names.Strs[i];
      }
    }
    LineStream(OS, Indent) << LastMsg;
  }

  static void recordApisOrTypes(SourceLocation SL) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(SL);
    AnalysisModeStaticsMap[LocInfo.first.getPath()].recordApisOrTypes(
        LocInfo.second);
  }
  static void recordEffort(SourceLocation SL, EffortLevel EL) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(SL);
    recordEffort(LocInfo.first, LocInfo.second, EL);
  }
  static void recordEffort(const tooling::UnifiedPath &Filename,
                           unsigned Offset, EffortLevel EL) {
    AnalysisModeStaticsMap[Filename.getPath()].recordEffort(Offset, EL);
  }
  static void setDependencies(const RuleGroups &Group) noexcept {
    for (unsigned i = 0; i < static_cast<uint8_t>(LibraryDependencies::NUMS);
         ++i) {
      DependencyBits.Flag |=
          Group.isDependsOn(static_cast<LibraryDependencies>(i)) << i;
    }
  }
};

decltype(AnalysisModeStats::DependencyBits) AnalysisModeStats::DependencyBits;

const std::string AnalysisModeStats::LastMsg =
    "See "
    "https://www.intel.com/content/www/us/en/docs/dpcpp-compatibility-tool/"
    "developer-guide-reference/current/overview.html for more details.";
llvm::StringMap<AnalysisModeStats> AnalysisModeStats::AnalysisModeStaticsMap;

void dumpAnalysisModeStatics(llvm::raw_ostream &OS) {
  if (!DpctGlobalInfo::isAnalysisModeEnabled())
    return;

  AnalysisModeStats::dump(OS);
}

void recordAnalysisModeEffort(SourceLocation SL, EffortLevel EL) {
  AnalysisModeStats::recordEffort(SL, EL);
}
void recordAnalysisModeEffort(const clang::tooling::UnifiedPath &Filename,
                              unsigned Offset, EffortLevel EL) {
  AnalysisModeStats::recordEffort(Filename, Offset, EL);
}

void recordRecognizedAPI(const CallExpr *CE) {
  if (DpctGlobalInfo::isAnalysisModeEnabled())
    AnalysisModeStats::recordApisOrTypes(CE->getBeginLoc());
}
void recordRecognizedType(TypeLoc TL) {
  if (DpctGlobalInfo::isAnalysisModeEnabled())
    AnalysisModeStats::recordApisOrTypes(TL.getBeginLoc());
}
void setDependenciesInfo(const RuleGroups &Group) noexcept {
  AnalysisModeStats::setDependencies(Group);
}

} // namespace dpct
} // namespace clang
