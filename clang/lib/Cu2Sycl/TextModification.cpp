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
#include "Utility.h"

#include "clang/AST/Attr.h"

using namespace clang;
using namespace clang::cu2sycl;
using namespace clang::tooling;

// Get textual representation of the Expr.
// This helper function is tricky. Ideally, we should use SourceLocation
// information in the expression to be able to access the actual character
// used for spelling of this expression in the source code (either before or
// after preprocessor). But the quality of the this information is bad.
// This should be addressed in the clang sources in the long run, but we need
// a working solution right now, so we use another way of getting the spelling.
// Specific example, when SourceLocation information is broken
//   - DeclRefExpr has valid information only about beginning of the expression,
//     pointers to the end of the expression point to the beginning.
std::string TextModification::getExprSpelling(const Expr *E,
                                              const ASTContext &Context) const {
  std::string StrBuffer;
  llvm::raw_string_ostream TmpStream(StrBuffer);
  auto LangOpts = Context.getLangOpts();
  E->printPretty(TmpStream, nullptr, PrintingPolicy(LangOpts), 0, &Context);
  return TmpStream.str();
}

Replacement ReplaceStmt::getReplacement(const ASTContext &Context) const {
  return Replacement(Context.getSourceManager(), TheStmt, ReplacementString);
}

Replacement RemoveAttr::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  SourceRange AttrRange = TheAttr->getRange();
  SourceLocation ARB = AttrRange.getBegin();
  SourceLocation ARE = AttrRange.getEnd();
  SourceLocation ExpB = SM.getExpansionLoc(ARB);
  // No need to invoke getExpansionLoc again if the location is the same.
  SourceLocation ExpE = (ARB == ARE) ? ExpB : SM.getExpansionLoc(ARE);
  return Replacement(SM, CharSourceRange::getTokenRange(ExpB, ExpE), "");
}

Replacement
ReplaceTypeInVarDecl::getReplacement(const ASTContext &Context) const {
  TypeLoc TL = D->getTypeSourceInfo()->getTypeLoc();
  return Replacement(Context.getSourceManager(), &TL, T);
}

Replacement ReplaceReturnType::getReplacement(const ASTContext &Context) const {
  SourceRange SR = FD->getReturnTypeSourceRange();
  return Replacement(Context.getSourceManager(), CharSourceRange(SR, true), T);
}

Replacement
RenameFieldInMemberExpr::getReplacement(const ASTContext &Context) const {
  SourceLocation SL = ME->getLocEnd();
  return Replacement(Context.getSourceManager(),
                     CharSourceRange(SourceRange(SL, SL), true), T);
}

Replacement InsertAfterStmt::getReplacement(const ASTContext &Context) const {
  CharSourceRange CSR = CharSourceRange(S->getSourceRange(), false);
  SourceLocation Loc = CSR.getEnd();
  auto &SM = Context.getSourceManager();
  auto &Opts = Context.getLangOpts();
  SourceLocation SpellLoc = SM.getSpellingLoc(Loc);
  unsigned Offs = Lexer::MeasureTokenLength(SpellLoc, SM, Opts);
  SourceLocation LastTokenBegin = Lexer::GetBeginningOfToken(Loc, SM, Opts);
  SourceLocation End = LastTokenBegin.getLocWithOffset(Offs);

  return Replacement(SM, CharSourceRange(SourceRange(End, End), false), T);
}

Replacement ReplaceInclude::getReplacement(const ASTContext &Context) const {
  return Replacement(Context.getSourceManager(), Range, T);
}

Replacement InsertComment::getReplacement(const ASTContext &Context) const {
  auto NL = getNL(SL, Context.getSourceManager());
  return Replacement(Context.getSourceManager(), SL, 0,
                     (llvm::Twine("/*") + NL + Text + NL + "*/" + NL).str());
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

Replacement ReplaceCallExpr::getReplacement(const ASTContext &Context) const {
  std::string NewString = Name + "(";
  if (Types.empty()) {
    for (auto A = Args.cbegin(); A != Args.cend(); A++) {
      NewString += getExprSpelling(*A, Context);
      if (A + 1 != Args.cend()) {
        NewString += ", ";
      }
    }
  } else {
    for (auto A = Args.cbegin(); A != Args.cend(); A++) {
      auto B = Types.cbegin();
      NewString += (*B + "(" + getExprSpelling(*A, Context) + ")");
      if (A + 1 != Args.cend()) {
        NewString += ", ";
      }
      B++;
    }
  }
  NewString += ")";
  return Replacement(Context.getSourceManager(), C, NewString);
}

Replacement InsertArgument::getReplacement(const ASTContext &Context) const {
  auto FNameLoc = FD->getNameInfo().getEndLoc();
  // TODO: Investigate what happens in macro expansion
  auto tkn =
      Lexer::findNextToken(FNameLoc, Context.getSourceManager(), LangOptions())
          .getValue();
  // TODO: Investigate if its possible to not have l_paren as next token
  assert(tkn.is(tok::TokenKind::l_paren));
  // Emit new argument at the end of l_paren token
  auto OutStr = ArgName;
  if (!FD->parameters().empty())
    OutStr = ArgName + ", ";
  return Replacement(Context.getSourceManager(), tkn.getEndLoc(), 0, OutStr);
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

Replacement InsertBeforeStmt::getReplacement(const ASTContext &Context) const {
  SourceLocation Begin = S->getSourceRange().getBegin();
  return Replacement(Context.getSourceManager(),
                     CharSourceRange(SourceRange(Begin, Begin), false), T);
}

Replacement RemoveArg::getReplacement(const ASTContext &Context) const {
  SourceRange SR = CE->getArg(N)->getSourceRange();
  SourceLocation Begin = SR.getBegin();
  SourceLocation End;
  bool IsLast = (N == (CE->getNumArgs() - 1));
  if (IsLast) {
    End = SR.getEnd();
  } else {
    End = CE->getArg(N + 1)->getSourceRange().getBegin().getLocWithOffset(-1);
  }
  return Replacement(Context.getSourceManager(),
                     CharSourceRange(SourceRange(Begin, End), true), "");
}
