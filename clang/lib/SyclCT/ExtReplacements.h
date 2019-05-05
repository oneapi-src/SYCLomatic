//===--- ExtReplacements.h -------------------------------------*- C++
//-*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

// ExtReplacements.h declare class ExtReplacements, which stores ExtReplacement
// set, with merging pass

#ifndef SYCLCT_EXTREPLACEMENTS_H
#define SYCLCT_EXTREPLACEMENTS_H

#include "TextModification.h"
#include "Utility.h"

namespace clang {
namespace syclct {
class SyclctFileInfo;

class ExtReplacements {
public:
  ExtReplacements(SyclctFileInfo *FileInfo);

  void addReplacement(std::shared_ptr<ExtReplacement> Repl);
  void emplaceIntoReplSet(tooling::Replacements &ReplSet);

  inline bool empty() { return ReplMap.empty(); }

  struct SourceLineRange {
    unsigned SrcBeginLine = 0, SrcEndLine = 0, SrcBeginOffset = 0;
  };

private:
  bool isInvalid(std::shared_ptr<ExtReplacement> Repl);

  std::shared_ptr<ExtReplacement> inline mergeAtSameOffset(
      std::shared_ptr<ExtReplacement> First,
      std::shared_ptr<ExtReplacement> Second) {
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

  void buildOriginCodeReplacements();

  // Remove comments in the source code.
  void removeCommentsInSrcCode(const std::string &SrcCode, std::string &Result,
                               bool &BlockComment);

  std::shared_ptr<ExtReplacement>
  buildOriginCodeReplacement(const SourceLineRange &LineRange);

  bool isEndWithSlash(unsigned LineNumber);
  size_t findCR(const std::string &Line);

  SyclctFileInfo *FileInfo;
  const std::string &FilePath;
  std::map<unsigned, std::shared_ptr<ExtReplacement>> ReplMap;

  const static StringRef NullStr;
};
} // namespace syclct
} // namespace clang

#endif