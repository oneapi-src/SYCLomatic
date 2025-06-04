//===----------------------- ReMigration.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReMigration.h"
#include "AnalysisInfo.h"
#include "ExternalReplacement.h"

#include "clang/AST/Expr.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Core/UnifiedPath.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <unordered_map>

namespace clang::dpct {
static GitDiffChanges UpstreamChanges;
static GitDiffChanges UserChanges;
AddFileHunk::AddFileHunk(std::string NewFilePath)
    : Hunk(AddFile),
      NewFilePath(tooling::UnifiedPath(NewFilePath).getCanonicalPath()) {}
DeleteFileHunk::DeleteFileHunk(std::string OldFilePath)
    : Hunk(DeleteFile),
      OldFilePath(tooling::UnifiedPath(OldFilePath).getCanonicalPath()) {}
GitDiffChanges &getUpstreamChanges() { return UpstreamChanges; }
GitDiffChanges &getUserChanges() { return UserChanges; }
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

void tryLoadingUpstreamChangesAndUserChanges() {
  llvm::SmallString<128> UpstreamChangesFilePath(
      DpctGlobalInfo::getInRoot().getCanonicalPath());
  llvm::SmallString<128> UserChangesFilePath(
      DpctGlobalInfo::getInRoot().getCanonicalPath());
  llvm::sys::path::append(UpstreamChangesFilePath, "UpstreamChanges.yaml");
  llvm::sys::path::append(UserChangesFilePath, "UserChanges.yaml");

  if (llvm::sys::fs::exists(UpstreamChangesFilePath)) {
    ::loadGDCFromYaml(UpstreamChangesFilePath, getUpstreamChanges());
  }
  if (llvm::sys::fs::exists(UserChangesFilePath)) {
    ::loadGDCFromYaml(UserChangesFilePath, getUserChanges());
  }
}

/// Calculate the new Repls of the input \p NewRepl after \p Repls is applied to
/// the files.
/// \param Repls Replacements to apply.
/// \param NewRepl Replacements before applying \p Repls.
/// \return The result Repls.
clang::tooling::Replacements
calculateUpdatedRanges(const clang::tooling::Replacements &Repls,
                       const clang::tooling::Replacements &NewRepl) {
  clang::tooling::Replacements Result;
  for (const auto &R : NewRepl) {
    unsigned int BOffset = Repls.getShiftedCodePosition(R.getOffset());
    unsigned int EOffset =
        Repls.getShiftedCodePosition(R.getOffset() + R.getLength());
    if (BOffset > EOffset)
      continue;
    (void)Result.add(tooling::Replacement(
        R.getFilePath(), BOffset, EOffset - BOffset, R.getReplacementText()));
  }
  return Result;
}

std::map<std::string, std::vector<tooling::Replacement>>
groupReplcementsByFile(
    const std::vector<tooling::Replacement> &Repls) {
  std::map<std::string, std::vector<tooling::Replacement>> Result;
  for (const auto &R : Repls) {
    Result[R.getFilePath().str()].push_back(R);
  }
  return Result;
}

StringRef getLineString(clang::tooling::UnifiedPath FilePath,
                        unsigned LineNumber) {
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(FilePath);
  StringRef Line = FileInfo->getLineString(LineNumber);
  return Line;
}
unsigned getLineNumber(clang::tooling::UnifiedPath FilePath, unsigned Offset) {
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(FilePath);
  return FileInfo->getLineNumber(Offset);
}
unsigned getLineBeginOffset(clang::tooling::UnifiedPath FilePath,
                            unsigned LineNumber) {
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(FilePath);
  return FileInfo->getLineInfo(LineNumber).Offset;
}

std::map<unsigned, std::string>
convertReplcementsLineString(const std::vector<tooling::Replacement> &Repls) {
  tooling::UnifiedPath FilePath(Repls[0].getFilePath());

  std::map<unsigned, std::string> Result;
  std::vector<tooling::Replacement> SortedRepls = Repls;

  std::sort(SortedRepls.begin(), SortedRepls.end(),
            [](const tooling::Replacement &A, const tooling::Replacement &B) {
              return A.getOffset() < B.getOffset();
            });

  for (const auto &Repl : SortedRepls) {
    unsigned StartLine = getLineNumber(FilePath, Repl.getOffset());
    unsigned EndLine =
        getLineNumber(FilePath, Repl.getOffset() + Repl.getLength() - 1);

    // process each line
    for (unsigned Line = StartLine; Line <= EndLine; ++Line) {
      std::string LineContent = getLineString(FilePath, Line).str();
      unsigned LineStartOffset = getLineBeginOffset(FilePath, Line);
      unsigned LineEndOffset = LineStartOffset + LineContent.size();

      // calculate the replacement range within the line
      unsigned ReplaceStart =
          std::max(Repl.getOffset(), LineStartOffset) - LineStartOffset;
      unsigned ReplaceEnd =
          std::min(Repl.getOffset() + Repl.getLength(), LineEndOffset) -
          LineStartOffset;

      // do replace
      if (Line == StartLine) {
        LineContent.replace(
            ReplaceStart, ReplaceEnd - ReplaceStart,
            Repl.getReplacementText().substr(0, ReplaceEnd - ReplaceStart));
      } else if (Line == EndLine) {
        unsigned textStart =
            Repl.getReplacementText().size() - (ReplaceEnd - ReplaceStart);
        LineContent.replace(0, ReplaceEnd,
                            Repl.getReplacementText().substr(textStart));
      } else {
        LineContent = Repl.getReplacementText().substr(
            ReplaceStart, LineEndOffset - LineStartOffset);
      }

      Result[Line] = LineContent;
    }
  }
  return Result;
}

std::map<unsigned, std::string>
convertReplcementsLineString(const tooling::Replacements &Repls) {
  std::vector<tooling::Replacement> ReplsVec;
  for (const auto &R : Repls) {
    ReplsVec.push_back(R);
  }
  return convertReplcementsLineString(ReplsVec);
}

std::vector<tooling::Replacement>
convertMapToReplacements(const std::map<unsigned, std::string> &Map,
                         const clang::tooling::UnifiedPath &FilePath) {
  std::vector<clang::tooling::Replacement> Result;
  for (const auto &Pair : Map) {
    unsigned LineNumber = Pair.first;
    StringRef LineContent = Pair.second;
    unsigned Offset = getLineBeginOffset(FilePath, LineNumber);
    Result.emplace_back(FilePath.getCanonicalPath(), Offset, LineContent.size(),
                        LineContent.str());
  }
  return Result;
}

std::vector<tooling::Replacement>
mergeMapsByLine(const std::map<unsigned, std::string> &MapA,
                const std::map<unsigned, std::string> &MapB,
                const clang::tooling::UnifiedPath &FilePath) {
  auto genReplacement = [&](unsigned LineNumber,
                            const std::string &LineContent) {
    unsigned Offset = getLineBeginOffset(FilePath, LineNumber);
    return tooling::Replacement(FilePath.getCanonicalPath(), Offset,
                                LineContent.size(), LineContent);
  };

  std::vector<tooling::Replacement> Result;
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
      // Conflict line(s)
      std::vector<std::string> ConflictA;
      std::vector<std::string> ConflictB;
      unsigned ConflictOffset = ItA->first;
      unsigned ConflictLength = 0;

      // Collect continuous conflicting lines
      while (ItA != MapA.end() && ItB != MapB.end() &&
             ItA->first == ItB->first) {
        ConflictA.push_back(ItA->second);
        ConflictB.push_back(ItB->second);
        ++ItA;
        ++ItB;
        ConflictLength += ItA->second.size();
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
//       V               shift              V            merge    |
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
// Merge process:
// 1. Merge Repl_C1 and Repl_C2 directly, named Repl_C. There is no conlict.
//    Repl_C can be divided in to 2 parts:
//      Repl_C_x: Replacements which in ranges of Repl_A_3 or delete hunks in
//                Repl_A_2/Repl_A_4.
//      Repl_C_y: Other replacements.
//    Repl_C_x will be ignored during this merge.
// 2. Shfit Repl_C_y with Repl_A, called Repl_D.
// 3. Merge Repl_D and Repl_B. May have conflicts.
std::map<std::string, std::vector<clang::tooling::Replacement>>
reMigrationMerge(const GitDiffChanges &Repl_A,
                 const std::vector<tooling::Replacement> &Repl_B,
                 const std::vector<tooling::Replacement> &Repl_C1,
                 const GitDiffChanges &Repl_C2) {
  assert(Repl_C2.AddFileHunks.empty() && Repl_C2.DeleteFileHunks.empty() &&
         Repl_C2.MoveFileHunks.empty() &&
         "Repl_C2 should only have ModifiyFileHunks.");
  std::vector<tooling::Replacement> Repl_C;
  // Merge Repl_C1 and Repl_C2
  Repl_C.insert(Repl_C.end(), Repl_C1.begin(), Repl_C1.end());
  for (const auto &Hunk : Repl_C2.ModifyFileHunks) {
    tooling::Replacement Replacement(
        Hunk.getFilePath(), Hunk.getOffset(), Hunk.getLength(),
        Hunk.getReplacementText());
    Repl_C.push_back(Replacement);
  }

  // Convert vector in Repl_A to map for quick lookup.
  std::map<std::string, std::map<unsigned /*Offset*/, unsigned /*Length*/>>
      DeletedParts;
  std::map<std::string, clang::tooling::Replacements> ModifiedParts;
  for (const auto &Hunk : Repl_A.ModifyFileHunks) {
    if (Hunk.getLength() != 0 && Hunk.getReplacementText().size() == 0) {
      DeletedParts[Hunk.getFilePath().str()][Hunk.getOffset()] =
          Hunk.getLength();
    }
    (void)ModifiedParts[Hunk.getFilePath().str()].add(
        tooling::Replacement(Hunk.getFilePath().str(),
                                        Hunk.getOffset(), Hunk.getLength(),
                                        Hunk.getReplacementText()));
  }
  for (const auto &Hunk : Repl_A.MoveFileHunks) {
    if (Hunk.getLength() != 0 && Hunk.getReplacementText().size() == 0) {
      DeletedParts[Hunk.getFilePath().str()][Hunk.getOffset()] = Hunk.getLength();
    }
    (void)ModifiedParts[Hunk.getFilePath().str()].add(
      tooling::Replacement(Hunk.getFilePath().str(),
                                      Hunk.getOffset(), Hunk.getLength(),
                                      Hunk.getReplacementText()));
  }
  for (const auto &Hunk : Repl_A.DeleteFileHunks) {
    DeletedParts[Hunk.getOldFilePath()] = std::map<unsigned, unsigned>();
  }

  // Get Repl_C_y
  std::map<std::string, clang::tooling::Replacements> Repl_C_y;
  for (const auto &Repl : Repl_B) {
    // The gitdiff changes are line-based while clang replacements are character-based.
    // So here assume there is no overlap between delete hunks and replacements.
    const auto &It = DeletedParts.find(Repl.getFilePath().str());
    if (It == DeletedParts.end()) {
      (void)Repl_C_y[Repl.getFilePath().str()].add(
          tooling::Replacement(Repl.getFilePath().str(),
                                          Repl.getOffset(), Repl.getLength(),
                                          Repl.getReplacementText()));
      continue;
    }

    // Check if the replacement is in a deleted part.
    // TODO: Use Interval Tree to speed up the lookup.
    for (const auto &Part : It->second) {
      if (Repl.getOffset() >= Part.first &&
          Repl.getOffset() + Repl.getLength() <= Part.first + Part.second) {
        break;
      }
    }
    (void)Repl_C_y[Repl.getFilePath().str()].add(
        tooling::Replacement(Repl.getFilePath().str(),
                                        Repl.getOffset(), Repl.getLength(),
                                        Repl.getReplacementText()));
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
  // 1. we should convert the replacements to a map <line_number, new_text>. We will have 2 maps.
  // 2. we need a vector<offset /*line end offset*/> for current file (CUDA code 2)
  // 3. merge line by line
  std::map<std::string, std::vector<clang::tooling::Replacement>> Result;
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
