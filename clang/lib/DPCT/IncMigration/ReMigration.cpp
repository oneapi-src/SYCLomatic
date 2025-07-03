//===----------------------- ReMigration.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Workflow: 
//      CUDA code v1
//           |
//           |  dpct (--format-range=off) => MainSourceFiles.yaml and *.h.yaml
//           v
//      SYCL code v1
//           |  1. git init and commit
//           |  2. manual format with clang-format
//           |  3. manual fix
//           v  4. gitdiff2yaml => UserChange.yaml
//      SYCL code v1.1
//
//      CUDA code v2
//           |
//           |  dpct (--format-range=off)
//           |    + MainSourceFiles.yaml and *.h.yaml
//           |    + UserChange.yaml
//           v
//      SYCL code v2
//===----------------------------------------------------------------------===//

#include "ReMigration.h"
#include "AnalysisInfo.h"
#include "ExternalReplacement.h"
#include "TextModification.h"

#include "clang/AST/Expr.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Core/UnifiedPath.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <unordered_map>

std::optional<
    std::function<llvm::StringRef(clang::tooling::UnifiedPath, unsigned)>>
    getLineStringHook = std::nullopt;
std::optional<std::function<unsigned(clang::tooling::UnifiedPath, unsigned)>>
    getLineNumberHook = std::nullopt;
std::optional<std::function<unsigned(clang::tooling::UnifiedPath, unsigned)>>
    getLineBeginOffsetHook = std::nullopt;

namespace clang::dpct {
using namespace clang::tooling;
static GitDiffChanges UpstreamChanges;
static GitDiffChanges UserChanges;
static TranslationUnitReplacements LastMigration;
AddFileHunk::AddFileHunk(std::string NewFilePath)
    : Hunk(AddFile), NewFilePath(UnifiedPath(NewFilePath).getCanonicalPath()) {}
DeleteFileHunk::DeleteFileHunk(std::string OldFilePath)
    : Hunk(DeleteFile),
      OldFilePath(UnifiedPath(OldFilePath).getCanonicalPath()) {}
GitDiffChanges &getUpstreamChanges() { return UpstreamChanges; }
GitDiffChanges &getUserChanges() { return UserChanges; }
TranslationUnitReplacements &getLastMigration() { return LastMigration; }
static void dumpGitDiffChanges(const GitDiffChanges &GHC) {
  llvm::errs() << "GitDiffChanges:\n";
  llvm::errs() << "  ModifyFileHunks:\n";
  for (const auto &Hunk : GHC.ModifyFileHunks) {
    llvm::errs() << "    - FilePath:        " << Hunk.getFilePath() << "\n";
    llvm::errs() << "      Offset:          " << Hunk.getOffset() << "\n";
    llvm::errs() << "      Length:          " << Hunk.getLength() << "\n";
    llvm::errs() << "      ReplacementText: " << Hunk.getReplacementText()
                 << "\n";
  }
  llvm::errs() << "  AddFileHunks:\n";
  for (const auto &Hunk : GHC.AddFileHunks) {
    llvm::errs() << "    - NewFilePath: " << Hunk.getNewFilePath() << "\n";
  }
  llvm::errs() << "  DeleteFileHunks:\n";
  for (const auto &Hunk : GHC.DeleteFileHunks) {
    llvm::errs() << "    - OldFilePath: " << Hunk.getOldFilePath() << "\n";
  }
  llvm::errs() << "  MoveFileHunks:\n";
  for (const auto &Hunk : GHC.MoveFileHunks) {
    llvm::errs() << "    - FilePath:        " << Hunk.getFilePath() << "\n";
    llvm::errs() << "      Offset:          " << Hunk.getOffset() << "\n";
    llvm::errs() << "      Length:          " << Hunk.getLength() << "\n";
    llvm::errs() << "      ReplacementText: " << Hunk.getReplacementText()
                 << "\n";
    llvm::errs() << "      NewFilePath:     " << Hunk.getNewFilePath() << "\n";
  }
}

bool tryLoadingUpstreamChangesAndUserChanges() {
  unsigned int Found = 0;
  llvm::SmallString<128> UpstreamChangesFilePath(
      DpctGlobalInfo::getInRoot().getCanonicalPath());
  llvm::SmallString<128> UserChangesFilePath(
      DpctGlobalInfo::getInRoot().getCanonicalPath());
  llvm::SmallString<128> LastMigrationFilePath(
      DpctGlobalInfo::getInRoot().getCanonicalPath());
  llvm::sys::path::append(UpstreamChangesFilePath, "UpstreamChanges.yaml");
  llvm::sys::path::append(UserChangesFilePath, "UserChanges.yaml");
  llvm::sys::path::append(LastMigrationFilePath, "LastMigration.yaml");

  if (llvm::sys::fs::exists(UpstreamChangesFilePath)) {
    ::loadGDCFromYaml(UpstreamChangesFilePath, getUpstreamChanges());
    Found++;
  }
  if (llvm::sys::fs::exists(UserChangesFilePath)) {
    ::loadGDCFromYaml(UserChangesFilePath, getUserChanges());
    Found++;
  }
  if (llvm::sys::fs::exists(LastMigrationFilePath)) {
    ::loadTUFromYaml(LastMigrationFilePath, getLastMigration());
    Found++;
  }
  return (Found == 3) ? true : false;
}

/// Below 3 functions only used for merging Repl_B and repl_D. So they only
/// need to read CUDA code v2.
static StringRef getLineString(UnifiedPath FilePath, unsigned LineNumber) {
#ifndef NDEBUG
  if (getLineStringHook.has_value()) {
    return getLineStringHook.value()(FilePath, LineNumber);
  }
#endif
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(FilePath);
  StringRef Line = FileInfo->getLineString(LineNumber);
  return Line;
}

static unsigned getLineNumber(UnifiedPath FilePath, unsigned Offset) {
#ifndef NDEBUG
  if (getLineNumberHook.has_value()) {
    return getLineNumberHook.value()(FilePath, Offset);
  }
#endif
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(FilePath);
  return FileInfo->getLineNumber(Offset);
}

static unsigned getLineBeginOffset(UnifiedPath FilePath, unsigned LineNumber) {
#ifndef NDEBUG
  if (getLineBeginOffsetHook.has_value()) {
    return getLineBeginOffsetHook.value()(FilePath, LineNumber);
  }
#endif
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(FilePath);
  return FileInfo->getLineInfo(LineNumber).Offset;
}

/// Calculate the new Repls of the input \p NewRepl after \p Repls is applied to
/// the files.
/// This shfit may have conflicts.
/// Since \p NewRepl (from Repl_C_y) is line based and \p Repls (from Repl_A) is
/// character based, we assume that if there is a conflict, the range from
/// Repl_A is always covering the range from Repl_C_y. Then we just ignore the
/// \p NewRepl (from Repl_C_y) since the old CUDA code is changed, so the
/// migration repl is out-of-date.
/// \param Repls Replacements to apply.
/// \param NewRepl Replacements before applying \p Repls.
/// \return The result Repls.
clang::tooling::Replacements
calculateUpdatedRanges(const clang::tooling::Replacements &Repls,
                       const clang::tooling::Replacements &NewRepl) {
  // Assumption: no overlap in the each groups.
  clang::tooling::Replacements Result;
  for (const auto &R : NewRepl) {
    // Check if the range (BOffset, EOffset - BOffset) is overlapped with any
    // repl in Repls
    std::optional<Replacement> MaxNotGreater = std::nullopt;
    for (const auto &ExistingR : Repls) {
      if (ExistingR.getOffset() <= R.getOffset())
        MaxNotGreater = ExistingR;
      else
        break;
    }
    if (MaxNotGreater.has_value()) {
      if (MaxNotGreater->getOffset() + MaxNotGreater->getLength() >
          R.getOffset())
        continue; // has overlap
    }

    unsigned int BOffset = Repls.getShiftedCodePosition(R.getOffset());
    unsigned int EOffset =
        Repls.getShiftedCodePosition(R.getOffset() + R.getLength());
    if (BOffset > EOffset)
      continue;
    llvm::cantFail(Result.add(Replacement(
        R.getFilePath(), BOffset, EOffset - BOffset, R.getReplacementText())));
  }
  return Result;
}

std::map<std::string, std::vector<Replacement>>
groupReplcementsByFile(const std::vector<Replacement> &Repls) {
  std::map<std::string, std::vector<Replacement>> Result;
  for (const auto &R : Repls) {
    Result[R.getFilePath().str()].push_back(R);
  }
  return Result;
}

// If repl range is cross lines, we treat the \n itself belongs to current line.
// case:
/*
aaaaa
bbbbb
ccccc
ddddd
*/
// replacement is: offset:6, length:11, text:xxx\nyyy
//
// If the splited repl of line 2 does not end with \n, if there is conflict at
// this line, the mark will not correct.
// In other words, if there is no conflict, the repl is correct since there is
// \n after the repl. While the conflict mark assumes that the repl is endding
// with \n so it can at the beginning of each line, this is the problem.
// So the solution is:
// 1. If the newtext is not ending with \n, we merge the left part of the last
// line into the repl (the first line).
// 2. If the newtext is ending with \n, we still keep the last line at the last
// line.
std::vector<Replacement>
splitReplInOrderToNotCrossLines(const std::vector<Replacement> &InRepls) {
  std::string FilePath = InRepls[0].getFilePath().str();
  std::vector<Replacement> Result;

  for (const auto &Repl : InRepls) {
    unsigned StartOffset = Repl.getOffset();
    unsigned EndOffset = StartOffset + Repl.getLength();
    unsigned StartLine = getLineNumber(FilePath, StartOffset);
    unsigned EndLine = getLineNumber(FilePath, EndOffset);

    if (StartLine == EndLine) {
      // Single line replacement
      Result.push_back(Repl);
      continue;
    }

    // Cross-line replacement
    unsigned CurrentOffset = StartOffset;

    // The first line
    // We need merge the left part of the last line into the replacement if the
    // newText is not ending with \n.
    bool IsFisrtLineEndingWithNL = Repl.getReplacementText().ends_with('\n');
    unsigned LineEndOffset = getLineBeginOffset(FilePath, StartLine + 1);
    unsigned FirstLineLength = LineEndOffset - StartOffset;
    Replacement ReplFisrtLine(FilePath, CurrentOffset, FirstLineLength,
                              Repl.getReplacementText());
    if (IsFisrtLineEndingWithNL)
      Result.push_back(ReplFisrtLine);
    CurrentOffset += FirstLineLength;

    // middle lines
    for (unsigned Line = StartLine + 1; Line < EndLine; ++Line) {
      LineEndOffset = getLineBeginOffset(FilePath, Line + 1);
      unsigned LineLength = LineEndOffset - CurrentOffset;
      Result.emplace_back(Repl.getFilePath(), CurrentOffset, LineLength, "");
      CurrentOffset += LineLength;
    }

    // The last line
    std::string LastLineStr = getLineString(FilePath, EndLine).str();
    unsigned LastLineLength = EndOffset - CurrentOffset;
    if (IsFisrtLineEndingWithNL) {
      if (LastLineLength > 0) {
        Result.emplace_back(Repl.getFilePath(), CurrentOffset, LastLineLength,
                            "");
      }
    } else {
      Result.emplace_back(Repl.getFilePath(), ReplFisrtLine.getOffset(),
                          ReplFisrtLine.getLength(),
                          ReplFisrtLine.getReplacementText().str() +
                              LastLineStr.substr(LastLineLength));
      Result.emplace_back(Repl.getFilePath(), CurrentOffset, LastLineStr.size(),
                          "");
    }
  }

  return Result;
}

std::map<unsigned, std::string>
convertReplcementsLineString(const std::vector<Replacement> &InRepls) {
  std::vector<Replacement> Replacements =
      splitReplInOrderToNotCrossLines(InRepls);
  UnifiedPath FilePath(InRepls[0].getFilePath());

  // group replacement by line
  std::map<unsigned, std::vector<Replacement>> ReplacementsByLine;
  for (const auto &Repl : Replacements) {
    unsigned LineNum = getLineNumber(FilePath, Repl.getOffset());
    ReplacementsByLine[LineNum].push_back(Repl);
  }

  // process each line
  std::map<unsigned, std::string> Result;
  for (auto &[LineNum, Repls] : ReplacementsByLine) {
    std::sort(Repls.begin(), Repls.end());

    std::string OriginalLineStr = getLineString(FilePath, LineNum).str();
    unsigned LineStartOffset = getLineBeginOffset(FilePath, LineNum);
    std::string NewLineStr;
    unsigned Pos = 0;
    for (const auto &Repl : Repls) {
      unsigned StrOffset = Repl.getOffset() - LineStartOffset;
      NewLineStr += OriginalLineStr.substr(Pos, StrOffset - Pos);
      NewLineStr += Repl.getReplacementText().str();
      Pos = StrOffset + Repl.getLength();
    }
    NewLineStr += OriginalLineStr.substr(Pos);
    Result[LineNum] = NewLineStr;
  }
  return Result;
}

static std::map<unsigned, std::string>
convertReplcementsLineString(const tooling::Replacements &Repls) {
  std::vector<Replacement> ReplsVec;
  for (const auto &R : Repls) {
    ReplsVec.push_back(R);
  }
  return convertReplcementsLineString(ReplsVec);
}

std::vector<Replacement>
mergeMapsByLine(const std::map<unsigned, std::string> &MapA,
                const std::map<unsigned, std::string> &MapB,
                const UnifiedPath &FilePath) {
  auto genReplacement = [&](unsigned LineNumber,
                            const std::string &LineContent) {
    unsigned Offset = getLineBeginOffset(FilePath, LineNumber);
    unsigned StrLen = getLineString(FilePath, LineNumber).size();
    return Replacement(FilePath.getCanonicalPath(), Offset, StrLen,
                       LineContent);
  };

  std::vector<Replacement> Result;
  auto ItA = MapA.begin();
  auto ItB = MapB.begin();

  while (ItA != MapA.end() || ItB != MapB.end()) {
    if (ItA == MapA.end()) {
      Result.push_back(genReplacement(ItB->first, ItB->second));
      ++ItB;
    } else if (ItB == MapB.end()) {
      Result.push_back(genReplacement(ItA->first, ItA->second));
      ++ItA;
    } else if (ItA->first < ItB->first) {
      Result.push_back(genReplacement(ItA->first, ItA->second));
      ++ItA;
    } else if (ItB->first < ItA->first) {
      Result.push_back(genReplacement(ItB->first, ItB->second));
      ++ItB;
    } else {
      // ItB->first == ItA->first
      if (ItB->second == ItA->second) {
        // No conflict, just add one of them
        Result.push_back(genReplacement(ItA->first, ItA->second));
        ++ItA;
        ++ItB;
        continue;
      }
      // Conflict line(s)
      std::vector<std::string> ConflictA;
      std::vector<std::string> ConflictB;
      unsigned ConflictOffset = getLineBeginOffset(FilePath, ItA->first);
      unsigned ConflictLength = 0;

      // Collect continuous conflicting lines
      unsigned CurrentLineNum = ItA->first;
      while (ItA != MapA.end() && ItB != MapB.end() &&
             CurrentLineNum == ItA->first && CurrentLineNum == ItB->first) {
        ConflictA.push_back(ItA->second);
        ConflictB.push_back(ItB->second);
        ConflictLength += getLineString(FilePath, ItA->first).size();
        ++ItA;
        ++ItB;
        ++CurrentLineNum;
      }

      // generate merged string
      std::string Merged = "<<<<<<<\n";
      for (const auto &L : ConflictA)
        Merged += L;
      Merged += "=======\n";
      for (const auto &L : ConflictB)
        Merged += L;
      Merged += ">>>>>>>\n";

      Result.emplace_back(FilePath.getCanonicalPath(), ConflictOffset,
                          ConflictLength, Merged);
    }
  }
  return Result;
}

static bool hasConflict(const Replacement &R1, const Replacement &R2) {
  if (R1.getFilePath() != R2.getFilePath())
    return false;
  if (R1.getOffset() == R2.getOffset()) {
    if (R1.getLength() && R2.getLength()) {
      return true;
    }
  }
  if ((R1.getOffset() < R2.getOffset() &&
       R1.getOffset() + R1.getLength() > R2.getOffset()) ||
      (R2.getOffset() < R1.getOffset() &&
       R2.getOffset() + R2.getLength() > R1.getOffset())) {
    return true;
  }
  return false;
}

namespace {
// Repl group A is the replacements generated by the first run of dpct
// Repl group B is the replacements written by the user manaully based on the
// SYCL code generated by  the first run of dpct.

// We name the original CUDA code TEXT0, the migrated code TEXT1, the manually
// fixed SYCL code TEXT2.

// Generate a mapping: (offset in TEXT1) -> (offset in TEXT0).

// Build the mappings based on the replacements in group A.
std::pair<std::vector<std::pair<unsigned, unsigned>>,
          std::vector<std::tuple<unsigned, unsigned, std::string>>>
buildMapping(const std::vector<Replacement> &A) {
  std::vector<std::pair<unsigned, unsigned>> OffsetMap;
  std::vector<std::tuple<unsigned, unsigned, std::string>> ModifiedRangeList;
  unsigned OrigPos = 0;
  unsigned MigratedCodePos = 0;

  auto SortedA = A;
  std::sort(SortedA.begin(), SortedA.end());

  for (const auto &Repl : SortedA) {
    // Add mapping before repl
    if (Repl.getOffset() > OrigPos) {
      unsigned len = Repl.getOffset() - OrigPos;
      OffsetMap.emplace_back(MigratedCodePos, OrigPos);
      MigratedCodePos += len;
      OrigPos += len;
    }

    // record the repl range
    ModifiedRangeList.emplace_back(MigratedCodePos, Repl.getLength(),
                                   Repl.getReplacementText().str());

    // Add mapping after repl
    OffsetMap.emplace_back(MigratedCodePos, Repl.getOffset());
    MigratedCodePos += Repl.getReplacementText().size();
    OrigPos += Repl.getLength();
  }

  // the last fragment
  OffsetMap.emplace_back(MigratedCodePos, OrigPos);
  return {OffsetMap, ModifiedRangeList};
}

// Map the offset in TEXT1 to the offset in TEXT0.
unsigned
mapToOriginalOffset(unsigned MigratedCodeOffset,
                    const std::vector<std::pair<unsigned, unsigned>> &Mapping) {
  auto It = std::upper_bound(
      Mapping.begin(), Mapping.end(), std::make_pair(MigratedCodeOffset, 0),
      [](const auto &a, const auto &b) { return a.first < b.first; });

  if (It != Mapping.begin())
    --It;
  return It->second + (MigratedCodeOffset - It->first);
}

std::vector<Replacement> mergeC1AndC2Impl(const std::vector<Replacement> &A,
                                          const std::vector<Replacement> &B) {
  // Build 2 mappings
  auto [OffsetMap, ModifiedRangeList] = buildMapping(A);
  std::string FilePath = A[0].getFilePath().str();

  std::vector<Replacement> Result;
  std::vector<Replacement> AGroupRepls;
  for (const auto &repl : ModifiedRangeList) {
    AGroupRepls.push_back(
        {FilePath, std::get<0>(repl), std::get<1>(repl), std::get<2>(repl)});
  }
  std::vector<Replacement> BGroupRepls = B;
  std::sort(
      AGroupRepls.begin(),
      AGroupRepls.end()); // The length in replA is based on original text, so
                          // for below calculation which is based on modified
                          // text, we need to use text.size() instead.
  std::sort(BGroupRepls.begin(), BGroupRepls.end());

  auto IterA = AGroupRepls.begin();
  auto IterB = BGroupRepls.begin();

  std::optional<std::tuple<unsigned /*offset(based on TEXT1)*/,
                           unsigned /*length*/, std::string>>
      UnfinishedRepl = std::nullopt;
  while (IterA != AGroupRepls.end() || IterB != BGroupRepls.end()) {
    bool NoA = IterA == AGroupRepls.end();
    bool NoB = IterB == BGroupRepls.end();
    if (UnfinishedRepl) {
      unsigned Off = std::get<0>(*UnfinishedRepl);
      unsigned Len = std::get<1>(*UnfinishedRepl);
      std::string Text = std::get<2>(*UnfinishedRepl);
      if ((NoA || (Off + Len) < IterA->getOffset()) &&
          (NoB || (Off + Len) < IterB->getOffset())) {
        // Finished
        Result.push_back({FilePath, mapToOriginalOffset(Off, OffsetMap),
                          mapToOriginalOffset(Off + Len, OffsetMap) -
                              mapToOriginalOffset(Off, OffsetMap),
                          Text});
        UnfinishedRepl = std::nullopt;
        continue;
      }
      if (!NoB && (Off + Len) >= IterB->getOffset()) {
        // Unfinished repl overlapping with next B repl.
        // We process B first because it has higher priority.
        std::get<1>(*UnfinishedRepl) =
            IterB->getLength() + IterB->getOffset() - Off;
        std::get<2>(*UnfinishedRepl) =
            Text.substr(0, IterB->getOffset() - Off) +
            IterB->getReplacementText().str();
        ++IterB;
        continue;
      }
      if (!NoA && (Off + Len) >= IterA->getOffset()) {
        if ((Off + Len) >=
            (IterA->getOffset() + IterA->getReplacementText().size())) {
          // case 1: totally cover IterA
          ++IterA;
          continue;
        }
        // case 2: partially cover IterA
        std::get<1>(*UnfinishedRepl) =
            IterA->getReplacementText().size() + IterA->getOffset() - Off;
        std::get<2>(*UnfinishedRepl) =
            Text + IterA->getReplacementText().str().substr(Off + Len -
                                                            IterA->getOffset());
        ++IterA;
        continue;
      }
    }
    if (NoA) {
      // only left B group replacements
      Result.push_back(
          {FilePath, mapToOriginalOffset(IterB->getOffset(), OffsetMap),
           mapToOriginalOffset(IterB->getOffset() + IterB->getLength(),
                               OffsetMap) -
               mapToOriginalOffset(IterB->getOffset(), OffsetMap),
           IterB->getReplacementText()});
      ++IterB;
      continue;
    }
    if (NoB) {
      // only left A group replacements
      Result.push_back(*IterA);
      ++IterA;
      continue;
    }
    if (IterA->getOffset() < IterB->getOffset()) {
      UnfinishedRepl = std::make_tuple(IterA->getOffset(),
                                       IterA->getReplacementText().size(),
                                       IterA->getReplacementText().str());
      ++IterA;
    } else {
      UnfinishedRepl = std::make_tuple(IterB->getOffset(), IterB->getLength(),
                                       IterB->getReplacementText().str());
      ++IterB;
    }
  }

  // TODO: figure out why we need this?
  if (UnfinishedRepl) {
    Result.push_back(
        {FilePath, mapToOriginalOffset(std::get<0>(*UnfinishedRepl), OffsetMap),
         mapToOriginalOffset(std::get<0>(*UnfinishedRepl) +
                                 std::get<1>(*UnfinishedRepl),
                             OffsetMap) -
             mapToOriginalOffset(std::get<0>(*UnfinishedRepl), OffsetMap),
         std::get<2>(*UnfinishedRepl)});
  }

  std::sort(Result.begin(), Result.end());
  return Result;
}
} // namespace

// Merge Repl_C1 and Repl_C2. If has conflict, keep repl from Repl_C2.
std::vector<Replacement> mergeC1AndC2(
    const std::vector<Replacement> &Repl_C1, const GitDiffChanges &Repl_C2,
    const std::map<UnifiedPath /*SYCL name*/, UnifiedPath /*CUDA name*/>
        &FileNameMap) {
  std::vector<Replacement> Result;
  std::vector<Replacement> Repl_C2_vec;
  std::for_each(Repl_C2.ModifyFileHunks.begin(), Repl_C2.ModifyFileHunks.end(),
                [&Repl_C2_vec, FileNameMap](const Replacement &Hunk) {
                  UnifiedPath OldFilePath =
                      FileNameMap.at(UnifiedPath(Hunk.getFilePath()));
                  Replacement Replacement(OldFilePath.getCanonicalPath(),
                                          Hunk.getOffset(), Hunk.getLength(),
                                          Hunk.getReplacementText());
                  Repl_C2_vec.push_back(Replacement);
                });

  auto C1 = groupReplcementsByFile(Repl_C1);
  auto C2 = groupReplcementsByFile(Repl_C2_vec);
  for (const auto &[FilePath, Repls_C1] : C1) {
    auto ItC2 = C2.find(FilePath);
    if (ItC2 == C2.end()) {
      // No replacements in C2 for this file, just add C1
      Result.insert(Result.end(), Repls_C1.begin(), Repls_C1.end());
      continue;
    }
    // Merge replacements in C1 and C2 for this file
    auto Repls_C2 = ItC2->second;
    auto Merged = mergeC1AndC2Impl(Repls_C1, Repls_C2);
    Result.insert(Result.end(), Merged.begin(), Merged.end());
  }

  return Result;
}

// clang-format off
//                               Repl A
// [CUDA code 1] -----------------------------------------> [CUDA code 2]
//       |                                                  /     |
//       | Repl C1                                         /      | Repl B
//       |                                        ________/       |
//       V                                       /                V
// [SYCL code 1]                                /           [SYCL code 2]
//       |                                     /                  |
//       | Repl C2                     Repl D /                   |
//       |                                   /                    |
//       V      shift (may have conlifct)   V            merge    |
// [SYCL code 1.1] ----------------> [SYCL code 1.1]  ----------> |
// (based on CUDA code 1)         (based on CUDA code 2)          |
//                                                                V
//                                                         [SYCL code 2.1]
//
// Repl_A: Read from gitdiff2yaml generated files.
// Repl_B: Curent in-memory migration replacements.
// Repl C1: Read from MainSourceFiles.yaml (and *.h.yaml) file(s).
// Repl_C2: Read from gitdiff2yaml generated files.
//
// Repl_A has 4 parts:
//   Repl_A_1: New added files.
//   Repl_A_2: Replacements in modified files.
//   Repl_A_3: Deleted files.
//   Repl_A_4: Replacements in moved files.
//
// clang-format on
//
// Merge process:
// 1. Merge Repl_C1 and Repl_C2 directly, named Repl_C. If there is conlict,
// we keep Repl_C2.
//    Repl_C can be divided in to 2 parts:
//      Repl_C_x: Replacements which in ranges of Repl_A_3 or delete hunks in
//                Repl_A_2/Repl_A_4.
//      Repl_C_y: Other replacements.
//    Repl_C_x will be ignored during this merge.
// 2. Shfit Repl_C_y with Repl_A, called Repl_D.
// 3. Merge Repl_D and Repl_B. May have conflicts.
std::map<std::string, std::vector<Replacement>> reMigrationMerge(
    const GitDiffChanges &Repl_A, const std::vector<Replacement> &Repl_B,
    const std::vector<Replacement> &Repl_C1, const GitDiffChanges &Repl_C2,
    const std::map<UnifiedPath /*SYCL name*/, UnifiedPath /*CUDA name*/>
        &FileNameMap) {
  // TODO: is this assumption true? Will the manual fix delete/add/move file?
  assert(Repl_C2.AddFileHunks.empty() && Repl_C2.DeleteFileHunks.empty() &&
         Repl_C2.MoveFileHunks.empty() &&
         "Repl_C2 should only have ModifiyFileHunks.");
  // Merge Repl_C1 and Repl_C2. If has conflict, keep repl from Repl_C2.
  // 1. Repl_C1 has name like file1.cpp, file2.cpp, file3.cu, file4.cuh
  //    but Repl_C2 has name like file1.cpp, file2.cpp.dp.cpp, file3.dp.cpp,
  //    file4.dp.hpp. We need convert the filename in Repl_C2 to CUDA style
  // 2. Repl_C1 is based on CUDA code, Repl_C2 is based on the migrated SYCL
  //    code, the result Repl_C should based on the original CUDA code.
  std::vector<Replacement> Repl_C = mergeC1AndC2(Repl_C1, Repl_C2, FileNameMap);

  // Convert vector in Repl_A to map for quick lookup.
  std::map<std::string, std::map<unsigned /*Offset*/, unsigned /*Length*/>>
      DeletedParts;
  std::map<std::string, clang::tooling::Replacements> ModifiedParts;
  for (const auto &Hunk : Repl_A.ModifyFileHunks) {
    if (Hunk.getLength() != 0 && Hunk.getReplacementText().size() == 0) {
      DeletedParts[Hunk.getFilePath().str()][Hunk.getOffset()] =
          Hunk.getLength();
    }
    llvm::cantFail(ModifiedParts[Hunk.getFilePath().str()].add(
        Replacement(Hunk.getFilePath().str(), Hunk.getOffset(),
                    Hunk.getLength(), Hunk.getReplacementText())));
  }
  for (const auto &Hunk : Repl_A.MoveFileHunks) {
    if (Hunk.getLength() != 0 && Hunk.getReplacementText().size() == 0) {
      DeletedParts[Hunk.getFilePath().str()][Hunk.getOffset()] =
          Hunk.getLength();
    }
    llvm::cantFail(ModifiedParts[Hunk.getFilePath().str()].add(
        Replacement(Hunk.getFilePath().str(), Hunk.getOffset(),
                    Hunk.getLength(), Hunk.getReplacementText())));
  }
  for (const auto &Hunk : Repl_A.DeleteFileHunks) {
    DeletedParts[Hunk.getOldFilePath()] = std::map<unsigned, unsigned>();
  }

  // Get Repl_C_y
  std::map<std::string, clang::tooling::Replacements> Repl_C_y;

  for (const auto &Repl : Repl_C) {
    // The gitdiff changes are line-based while clang replacements are
    // character-based. So here assume there is no overlap (only repl totally
    // covered by delete hunk) between delete hunks and replacements.
    const auto &It = DeletedParts.find(Repl.getFilePath().str());
    if (It == DeletedParts.end()) {
      llvm::cantFail(Repl_C_y[Repl.getFilePath().str()].add(
          Replacement(Repl.getFilePath().str(), Repl.getOffset(),
                      Repl.getLength(), Repl.getReplacementText())));
      continue;
    }

    // Check if the replacement is in a deleted part.
    bool HasConflict = false;
    for (const auto &Part : It->second) {
      if (HasConflict =
              hasConflict(Repl, Replacement(Repl.getFilePath(), Part.first,
                                            Part.second, "")))
        break;
    }
    if (!HasConflict)
      llvm::cantFail(Repl_C_y[Repl.getFilePath().str()].add(
          Replacement(Repl.getFilePath().str(), Repl.getOffset(),
                      Repl.getLength(), Repl.getReplacementText())));
  }

  // Shift Repl_C_y with Repl_A(ModifiedParts)
  std::map<std::string, clang::tooling::Replacements> Repl_D;
  for (const auto &Item : Repl_C_y) {
    const auto &FilePath = Item.first;
    const auto &Repls = Item.second;
    // Check if the file has modified parts.
    const auto &It = ModifiedParts.find(FilePath);
    if (It == ModifiedParts.end()) {
      Repl_D[FilePath] = Repls;
      continue;
    }
    Repl_D[FilePath] = calculateUpdatedRanges(It->second, Repls);
  }

  // Group Repl_B by file
  const auto Repl_B_by_file = groupReplcementsByFile(Repl_B);
  // Merge Repl_D and Repl_B
  // Convert the repls to a map <line_number, new_text> then merge line by line
  std::map<std::string, std::vector<Replacement>> Result;
  for (const auto &Pair : Repl_B_by_file) {
    std::map<unsigned, std::string> ReplBInLines =
        convertReplcementsLineString(Pair.second);
    if (Repl_D.find(Pair.first) != Repl_D.end()) {
      // Merge Repl_D and Repl_B
      std::map<unsigned, std::string> ReplDInLines =
          convertReplcementsLineString(Repl_D[Pair.first]);
      Result[Pair.first] =
          mergeMapsByLine(ReplBInLines, ReplDInLines, Pair.first);
    } else {
      // No Repl_D for this file, just add Repl_B.
      Result[Pair.first] = Pair.second;
    }
  }

  return Result;
}
} // namespace clang::dpct
