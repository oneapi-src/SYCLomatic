//===--- Statics.cpp--------- --------------------------------*- C++ -*---===//
////
//// Copyright (C) Intel Corporation. All rights reserved.
////
//// The information and source code contained herein is the exclusive
//// property of Intel Corporation and may not be disclosed, examined
//// or reproduced in whole or in part without explicit written authorization
//// from the company.
////
////===-----------------------------------------------------------------===//
#include "Statics.h"
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

int VerboseLevel = VL_NonVerbose;

void StaticsInfo::printMigrationRules(
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

  if (VerboseLevel > VL_NonVerbose) {
    print();
  }

#endif // DPCT_DEBUG_BUILD
}

#ifdef DPCT_DEBUG_BUILD
// Start of debug build
static void printMatchedRulesStaticsImpl(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
  // Verbose level lower than "High" doesn't show migration rules' information
  if (VerboseLevel < VL_VerboseHigh) {
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

static void printReplacementsStaticsImpl(const TransformSetTy &TS,
                                         ASTContext &Context) {
  // Verbos level lower than "High" doesn't show detailed replacements'
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
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
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

llvm::raw_ostream &DpctLog() { return DpctLogStream; }
llvm::raw_ostream &DpctStats() { return DpctStatsStream; }
llvm::raw_ostream &DpctDiags() { return DpctDiagsStream; }
llvm::raw_ostream &DpctTerm() { return DpctTermStream; }
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
    break;
  case OutputVerbosityLevel::OVL_Normal:
    if (IsPrintOnNormal) {
      llvm::outs() << Msg;
    }
    break;
  case OutputVerbosityLevel::OVL_Silent:
  default:
    break;
  }
}

} // namespace dpct
} // namespace clang
