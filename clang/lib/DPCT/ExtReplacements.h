//===--- ExtReplacements.h ------------------------------*- C++-*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

// ExtReplacements.h declare class ExtReplacements, which stores ExtReplacement
// set, with merging pass

#ifndef DPCT_EXTREPLACEMENTS_H
#define DPCT_EXTREPLACEMENTS_H

#include "TextModification.h"
#include "Utility.h"

namespace clang {
namespace dpct {
class DpctFileInfo;

class ExtReplacements {

  // Save pair replacements status and the first encountered replacement in the
  // pair. The pair is dead only when all the replacements are dead.
  struct PairReplsStatus {
    // Dead: the replacements are dead.
    // Alive: the replacements are alive or merged.
    enum StatusKind { Dead, Alive };
    // Pair status is initialized with the first encountered replacement status.
    PairReplsStatus(std::shared_ptr<ExtReplacement> Repl, StatusKind Status)
        : Repl(Repl), Status(Status) {}
    // The first encountered replacement
    std::shared_ptr<ExtReplacement> Repl;
    // Pair status.
    StatusKind Status;
  };

public:
  ExtReplacements(DpctFileInfo *FileInfo);

  void addReplacement(std::shared_ptr<ExtReplacement> Repl);
  void emplaceIntoReplSet(tooling::Replacements &ReplSet);

  inline bool empty() { return ReplMap.empty(); }

  struct SourceLineRange {
    unsigned SrcBeginLine = 0, SrcEndLine = 0, SrcBeginOffset = 0;
  };

private:
  using ReplIterator =
      std::multimap<unsigned, std::shared_ptr<ExtReplacement>>::iterator;

private:
  bool isInvalid(std::shared_ptr<ExtReplacement> Repl);

  inline bool checkLiveness(std::shared_ptr<ExtReplacement> Repl) {
    if (isAlive(Repl))
      // If a replacement in the same pair is alive, merge it anyway.
      return true;
    // Check if it is duplicate replacement.
    return !isDuplicated(Repl, ReplMap.lower_bound(Repl->getOffset()),
                     ReplMap.upper_bound(Repl->getOffset()));
  }

  bool isDuplicated(std::shared_ptr<ExtReplacement> Repl, ReplIterator Begin,
                ReplIterator End);

  std::shared_ptr<ExtReplacement> inline mergeAtSameOffset(
      std::shared_ptr<ExtReplacement> First,
      std::shared_ptr<ExtReplacement> Second) {
    if (!First)
      return Second;
    if (!Second)
      return First;
    bool ShorterIsFirst = First->getLength() < Second->getLength();
    return ShorterIsFirst ? mergeComparedAtSameOffset(First, Second)
                          : mergeComparedAtSameOffset(Second, First);
  }

  /// Do merge for Short replacement and Longer replacement.
  ///
  /// Return the merged replacemtent.
  /// Prerequisite: Shorter replacement's length should be not more than Longer
  /// replacement's.
  std::shared_ptr<ExtReplacement>
  mergeComparedAtSameOffset(std::shared_ptr<ExtReplacement> Shorter,
                            std::shared_ptr<ExtReplacement> Longer);

  inline std::shared_ptr<ExtReplacement>
  mergeReplacement(std::shared_ptr<ExtReplacement> First,
                   std::shared_ptr<ExtReplacement> Second) {
    return std::make_shared<ExtReplacement>(
        FilePath, First->getOffset(),
        Second->getOffset() + Second->getLength() - First->getOffset(),
        (First->getReplacementText() + Second->getReplacementText()).str(),
        nullptr);
  }
  std::shared_ptr<ExtReplacement>
  filterOverlappedReplacement(std::shared_ptr<ExtReplacement> Repl,
                              unsigned &PrevEnd);

  std::vector<std::shared_ptr<ExtReplacement>> mergeReplsAtSameOffset();

  void buildOriginCodeReplacements();

  // Remove comments in the source code.
  void removeCommentsInSrcCode(const std::string &SrcCode, std::string &Result,
                               bool &BlockComment);

  std::shared_ptr<ExtReplacement>
  buildOriginCodeReplacement(const SourceLineRange &LineRange);

  bool isEndWithSlash(unsigned LineNumber);
  size_t findCR(const std::string &Line);

  // Mark a replacement as dead.
  void markAsDead(std::shared_ptr<ExtReplacement> Repl) {
    if (auto PairID = Repl->getPairID())
      PairReplsMap[PairID] =
          std::make_shared<PairReplsStatus>(Repl, PairReplsStatus::Dead);
  }

  // Mark a replacement as alive and insert into ReplMap
  // If it is not the first encountered replacement and the first one is
  // dead, insert the first one into ReplMap, too.
  void markAsAlive(std::shared_ptr<ExtReplacement> Repl);

  // Check if its pair has a replacement inserted.
  bool isAlive(std::shared_ptr<ExtReplacement> Repl) {
    if (auto PairID = Repl->getPairID()) {
      if (auto &R = PairReplsMap[PairID])
        return R->Status == PairReplsStatus::Alive;
    }
    return false;
  }

  const std::string &FilePath;
  DpctFileInfo *FileInfo;
  ///<Offset, ExtReplacement>
  std::multimap<unsigned, std::shared_ptr<ExtReplacement>> ReplMap;
  ///<PairID, PairStatus>
  std::map<unsigned, std::shared_ptr<PairReplsStatus>> PairReplsMap;

  const static StringRef NullStr;
};
} // namespace dpct
} // namespace clang

#endif
