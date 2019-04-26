//===--- ExprReplacements.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "ExtReplacements.h"
#include "Debug.h"

namespace clang {
namespace syclct {

bool ExtReplacements::isInvalid(std::shared_ptr<ExtReplacement> Repl) {
  if (Repl->getFilePath().empty())
    return true;
  if (Repl->getLength() == 0 && Repl->getReplacementText().empty())
    return true;
  return false;
}

/// Do merge for Short replacement and Longer replacement.
///
/// Return the merged replacemtent.
/// Prerequisite: Shorter replacement's length should be not more than Longer
/// replacement's.
std::shared_ptr<ExtReplacement> ExtReplacements::mergeComparedAtSameOffset(
    std::shared_ptr<ExtReplacement> Shorter,
    std::shared_ptr<ExtReplacement> Longer) {
  if (Shorter->getLength() == Longer->getLength()) {
    if (Longer->getReplacementText().equals(Shorter->getReplacementText()) &&
        Longer->getLength()) {
      // Fully equal replacements, just reserve one.
      return Longer;
    } else if (Longer->getReplacementText().equals(
                   Shorter->getReplacementText()) &&
               Shorter->getReplacementText().find(StringRef("(")) ==
                   StringRef::npos &&
               Shorter->getReplacementText().find(StringRef(")")) ==
                   StringRef::npos &&
               Longer->getLength() == 0) {
      // Fully equal insert,  just reserve one. if not "( )".
      // Todo:  need further figout the rule.
      return Longer;

    } else {
      // Both Shorter and Longer are insert replacements, do merge.
      // Or same length but different code replacement text, do merge.
      // inset replacement could be "namespace::", "(type cast)",  ")"  "(".
      return (Longer->getInsertPosition() == InsertPositionLeft)
                 ? mergeReplacement(Longer, Shorter)
                 : mergeReplacement(Shorter, Longer);
    }
  } else if (!Shorter->getLength()) {
    // Shorter is insert replacement, Longer is code replacement, do merge.
    return mergeReplacement(Shorter, Longer);
  } else {
    // Shorter is a sub replacement of Longer, just reserve Longer.
    // TODO: Currently not make sure "keep the longer, remove shorter" is
    // correct, need to do in the future.
    return Longer;
  }
}

std::shared_ptr<ExtReplacement> ExtReplacements::filterOverlappedReplacement(
    std::shared_ptr<ExtReplacement> Repl, unsigned &PrevEnd) {
  auto ReplEnd = Repl->getOffset() + Repl->getLength();
  if (PrevEnd > ReplEnd)
    return std::shared_ptr<ExtReplacement>();
  if (PrevEnd == ReplEnd && Repl->getLength())
    return std::shared_ptr<ExtReplacement>();
  if ((Repl->getOffset() < PrevEnd) && !Repl->getReplacementText().empty())
    syclct_unreachable("overlapped replacements");

  PrevEnd = ReplEnd;
  return Repl;
}

void ExtReplacements::addReplacement(std::shared_ptr<ExtReplacement> Repl) {
  if (isInvalid(Repl))
    return;
  auto &R = ReplMap[Repl->getOffset()];

  if (R)
    R = mergeAtSameOffset(R, Repl);
  else
    R = Repl;
}

void ExtReplacements::emplaceIntoReplSet(tooling::Replacements &ReplSet) {
  // TODO: Original code should be output when required
  unsigned PrevEnd = 0;
  for (auto &R : ReplMap) {
    if (auto Repl = filterOverlappedReplacement(R.second, PrevEnd)) {
      if (auto Err = ReplSet.add(*Repl)) {
        llvm::dbgs() << Err << "\n";
        syclct_unreachable("Adding the replacement: Error occured ");
      }
    }
  }
}
} // namespace syclct
} // namespace clang
