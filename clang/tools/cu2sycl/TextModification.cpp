//===--- TextModification.cpp ---------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "TextModification.h"

#include "clang/AST/Attr.h"

using namespace clang;
using namespace clang::cu2sycl;
using namespace clang::tooling;

Replacement ReplaceStmt::getReplacement(const SourceManager &SM) const {
  return Replacement(SM, TheStmt, ReplacementString);
}

Replacement RemoveAttr::getReplacement(const SourceManager &SM) const {
  SourceRange AttrRange = TheAttr->getRange();
  SourceLocation ARB = AttrRange.getBegin();
  SourceLocation ARE = AttrRange.getEnd();
  SourceLocation ExpB = SM.getExpansionLoc(ARB);
  // No need to invoke getExpansionLoc again if the location is the same.
  SourceLocation ExpE = (ARB == ARE) ? ExpB : SM.getExpansionLoc(ARE);
  return Replacement(SM, CharSourceRange::getTokenRange(ExpB, ExpE), "");
}

Replacement
ReplaceTypeInVarDecl::getReplacement(const SourceManager &SM) const {
  TypeLoc TL = D->getTypeSourceInfo()->getTypeLoc();
  return Replacement(SM, &TL, T);
}

Replacement ReplaceReturnType::getReplacement(const SourceManager &SM) const {
  SourceRange SR = FD->getReturnTypeSourceRange();
  return Replacement(SM, CharSourceRange(SR, true), T);
}

Replacement
RenameFieldInMemberExpr::getReplacement(const SourceManager &SM) const {
  SourceLocation SL = ME->getLocEnd();
  return Replacement(SM, CharSourceRange(SourceRange(SL, SL), true), T);
}

Replacement InsertAfterStmt::getReplacement(const SourceManager &SM) const {
  SourceLocation Loc = S->getSourceRange().getEnd();
  clang::LangOptions Opts;
  unsigned Offs = Lexer::MeasureTokenLength(Loc, SM, Opts);
  return Replacement(SM, Loc.getLocWithOffset(Offs), 0, T);
}

bool ReplacementFilter::containsInterval(const IntervalSet &IS,
                                         const Interval &I) const {
  size_t Low = 0;
  size_t High = IS.size();

  while (High != Low) {
    size_t Mid = Low + (High - Low) / 2;

    if (IS[Mid].Offset <= I.Offset) {
      if (IS[Mid].Offset + IS[Mid].Length >= I.Offset + I.Length)
        return true;
      Low = Mid + 1;
    } else {
      High = Mid;
    }
  }

  return false;
}

bool ReplacementFilter::isDeletedReplacement(
    const tooling::Replacement &R) const {
  if (R.getReplacementText().empty())
    return false;
  auto Found = FileMap.find(R.getFilePath());
  if (Found == FileMap.end())
    return false;
  return containsInterval(Found->second, {R.getOffset(), R.getLength()});
}

size_t ReplacementFilter::findFirstNotDeletedReplacement(size_t Start) const {
  size_t Size = ReplSet.size();
  for (size_t Index = Start; Index < Size; ++Index)
    if (!isDeletedReplacement(ReplSet[Index]))
      return Index;
  return -1;
}

ReplacementFilter::ReplacementFilter(const std::vector<Replacement> &RS)
    : ReplSet(RS) {
  // TODO: Smaller Intervals should be discarded if they are completely
  // covered by a larger Interval, so that no intervals overlap in the set.
  for (const Replacement &R : ReplSet)
    if (R.getReplacementText().empty())
      FileMap[R.getFilePath()].push_back({R.getOffset(), R.getLength()});
  for (auto &FMI : FileMap)
    std::sort(FMI.second.begin(), FMI.second.end());
}

Replacement InsertBeforeStmt::getReplacement(const SourceManager &SM) const {
  return Replacement(SM, S->getSourceRange().getBegin(), 0, T);
}

Replacement RemoveArg::getReplacement(const SourceManager &SM) const {
  SourceRange SR = CE->getArg(N)->getSourceRange();
  SourceLocation Begin = SR.getBegin();
  SourceLocation End;
  bool IsLast = (N == (CE->getNumArgs() - 1));
  if (IsLast) {
      End = SR.getEnd();
  }
  else {
      End =  CE->getArg(N+1)->getSourceRange().getBegin().getLocWithOffset(-1);
  }
  return Replacement(SM, CharSourceRange(SourceRange(Begin, End), true), "");
}
