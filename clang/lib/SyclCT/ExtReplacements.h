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

class ExtReplacements {
public:
  ExtReplacements(const std::string &FilePath) : FilePath(FilePath) {}

  void addReplacement(std::shared_ptr<ExtReplacement> Repl);
  void emplaceIntoReplSet(tooling::Replacements &ReplSet);

private:
  bool isInvalid(std::shared_ptr<ExtReplacement> Repl);

  std::shared_ptr<ExtReplacement> inline mergeAtSameOffset(
      std::shared_ptr<ExtReplacement> First,
      std::shared_ptr<ExtReplacement> Second) {
    bool ShorterIsFirst = First->getLength() < Second->getLength();
    return ShorterIsFirst ? mergeComparedAtSameOffset(First, Second)
                          : mergeComparedAtSameOffset(Second, First);
  }
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

  const std::string &FilePath;
  std::map<unsigned, std::shared_ptr<ExtReplacement>> ReplMap;

  const static StringRef NullStr;
};
} // namespace syclct
} // namespace clang

#endif