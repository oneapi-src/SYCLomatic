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
namespace c2s {

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
#ifdef C2S_DEBUG_BUILD
  auto print = [&]() {
    C2SDiags() << "Migration Rules:\n";

    constexpr char Indent[] = "  ";
    if (TRs.empty()) {
      C2SDiags() << Indent << "None\n";
      return;
    }

    size_t NumRules = 0;
    for (auto &TR : TRs) {
      if (auto I = dyn_cast<MigrationRule>(&*TR)) {
        C2SDiags() << Indent << I->getName() << "\n";
        ++NumRules;
      }
    }
    C2SDiags() << "# of MigrationRules: " << NumRules << "\n";
  };

  if (VerboseLevel > VL_NonVerbose) {
    print();
  }

#endif // C2S_DEBUG_BUILD
}

#ifdef C2S_DEBUG_BUILD
// Start of debug build
static void printMatchedRulesStaticsImpl(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
  // Verbose level lower than "High" doesn't show migration rules' information
  if (VerboseLevel < VL_VerboseHigh) {
    return;
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<MigrationRule>(&*MR)) {
      TR->print(C2SDiags());
    }
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<MigrationRule>(&*MR)) {
      TR->printStatistics(C2SDiags());
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
    TM->print(C2SDiags(), Context);
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
    C2SDiags() << "# of replacement <"
                << TextModification::TMNameMap.at((int)ID) << ">: " << Numbers
                << " (" << Numbers << "/" << NumRepls << ")\n";
  }
}

// End of debug build
#endif

void StaticsInfo::printMatchedRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
#ifdef C2S_DEBUG_BUILD // Debug build
  printMatchedRulesStaticsImpl(MatchedRules);
#endif
}

void StaticsInfo::printReplacements(const TransformSetTy &TS,
                                    ASTContext &Context) {
#ifdef C2S_DEBUG_BUILD // Debug build
  printReplacementsStaticsImpl(TS, Context);
#endif
}

// Log buffer, default size 4096, when running out of memory, dynamic memory
// allocation is handled by SmallVector internally.
static llvm::SmallVector<char, /* default buffer size */ 4096> C2SLogBuffer;
static llvm::raw_svector_ostream C2SLogStream(C2SLogBuffer);
static llvm::SmallVector<char, /* default buffer size */ 4096> C2SStatsBuffer;
static llvm::raw_svector_ostream C2SStatsStream(C2SStatsBuffer);
static llvm::SmallVector<char, /* default buffer size */ 4096> C2SDiagsBuffer;
static llvm::raw_svector_ostream C2SDiagsStream(C2SDiagsBuffer);

static llvm::SmallVector<char, /* default buffer size */ 4096> C2STermBuffer;
static llvm::raw_svector_ostream C2STermStream(C2STermBuffer);

llvm::raw_ostream &C2SLog() { return C2SLogStream; }
llvm::raw_ostream &C2SStats() { return C2SStatsStream; }
llvm::raw_ostream &C2SDiags() { return C2SDiagsStream; }
llvm::raw_ostream &C2STerm() { return C2STermStream; }
std::string getC2SStatsStr() { return C2SStatsStream.str().str(); }
std::string getC2SDiagsStr() { return C2SDiagsStream.str().str(); }
std::string getC2STermStr() { return C2STermStream.str().str(); }
std::string getC2SLogStr() { return C2SLogStream.str().str(); }

// Currently, set IsPrintOnNormal false only at the place where messages about
// start and end of file parsing are produced,
//.i.e in the place "lib/Tooling:int ClangTool::run(ToolAction *Action)".
void PrintMsg(const std::string &Msg, bool IsPrintOnNormal) {
  if (!OutputFile.empty()) {
    //  Redirects stdout/stderr output to <file>
    C2STerm() << Msg;
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

} // namespace c2s
} // namespace clang
