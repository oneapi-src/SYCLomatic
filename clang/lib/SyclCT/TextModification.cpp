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

#include "Utility.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"

#include <sstream>

using namespace clang;
using namespace clang::syclct;
using namespace clang::tooling;

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

Replacement ReplaceToken::getReplacement(const ASTContext &Context) const {
  // Need to deal with the fact, that the type name might be a macro.
  return Replacement(Context.getSourceManager(),
                     // false means [Begin, End)
                     // true means [Begin, End]
                     CharSourceRange(SourceRange(Begin, Begin), true), T);
}

Replacement ReplaceCCast::getReplacement(const ASTContext &Context) const {
  auto Begin = Cast->getLParenLoc();
  auto End = Cast->getRParenLoc();
  return Replacement(Context.getSourceManager(),
                     CharSourceRange(SourceRange(Begin, End), true), TypeName);
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

template <typename ArgIterT>
std::string buildArgList(llvm::iterator_range<ArgIterT> Args,
                         const ASTContext &Context) {
  std::string List;
  for (auto A = begin(Args); A != end(Args); A++) {
    List += getStmtSpelling(*A, Context);
    if (A + 1 != end(Args)) {
      List += ", ";
    }
  }
  return List;
}

template <typename ArgIterT, typename TypeIterT>
std::string buildArgList(llvm::iterator_range<ArgIterT> Args,
                         llvm::iterator_range<TypeIterT> Types,
                         const ASTContext &Context) {
  std::string List;
  for (auto A = begin(Args); A != end(Args); A++) {
    auto B = begin(Types);
    List += (*B + "(" + getStmtSpelling(*A, Context) + ")");
    if (A + 1 != end(Args)) {
      List += ", ";
    }
    B++;
  }
  return List;
}

template <typename ArgIterT, typename TypeIterT>
std::string
buildCall(const std::string &Name, llvm::iterator_range<ArgIterT> Args,
          llvm::iterator_range<TypeIterT> Types, const ASTContext &Context) {
  std::string List;
  if (begin(Types) == end(Types)) {
    List = buildArgList(Args, Context);
  } else {
    List = buildArgList(Args, Types, Context);
  }
  return Name + "(" + List + ")";
}

Replacement
ReplaceKernelCallExpr::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  auto NL = getNL(KCall->getLocEnd(), SM);
  auto OrigIndent = getIndent(KCall->getLocStart(), SM).str();
  auto KName = KCall->getCalleeDecl()->getAsFunction()->getName().str();
  auto NDSize = KCall->getConfig()->getArg(0);
  auto WGSize = KCall->getConfig()->getArg(1);
  std::stringstream Header;
  std::stringstream Header2;
  std::stringstream Header3;
  Header << "{" << NL;
  auto Indent = OrigIndent + "  ";
  for (auto *Arg : KCall->arguments()) {
    if (Arg->getType()->isAnyPointerType()) {
      if (auto *DeclRef = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts())) {
        auto VarName = DeclRef->getNameInfo().getAsString();
        auto PointeeType = DeclRef->getDecl()->getType()->getPointeeType();
        // TODO check that no nested pointers in a structure
        assert(!PointeeType->isAnyPointerType());
        auto VarType = PointeeType.getCanonicalType().getAsString();
        Header << Indent << "std::pair<cl::sycl::buffer<char, 1 >*, size_t> "
               << VarName << "_buf = syclct::get_buffer_and_offset("
               << VarName + ");" << NL;
        Header << Indent << "size_t " << VarName
               << "_offset = " << VarName + "_buf.second;" << NL;
        Header2 << Indent << "    auto " << VarName << "_acc = " << VarName
                << "_buf.first->"
                   "get_access<cl::sycl::access::mode::read_write>("
                << "cgh);" << NL;
        Header3 << Indent << "        " << VarType << " *" << VarName << " = ("
                << VarType << "*)(&" << VarName << "_acc[0] + " << VarName
                << "_offset);" << NL;
      } else {
        assert(false && "unknown argumant expression");
      }
    }
  }
  // clang-format off
  std::stringstream Final;
  Final
  << Header.str()
  << Indent << "syclct::get_device_manager().current_device().default_queue().submit(" << NL
  << Indent <<  "  [=](cl::sycl::handler &cgh) {" << NL
  << Header2.str()
  << Indent <<  "    cgh.parallel_for<class " << KName << ">(" << NL
  << Indent <<  "      cl::sycl::nd_range<3>(" << getStmtSpelling(NDSize, Context) << ", "
                                               << getStmtSpelling(WGSize, Context) << ")," << NL
  << Indent <<  "      [=](cl::sycl::nd_item<3> it) {" << NL
  << Header3.str()
  << Indent <<  "        "
  << KName <<  "(it, " << buildArgList(KCall->arguments(), Context) << ");" <<  NL
  << Indent <<  "      });" <<  NL
  << Indent <<  "  })" <<  NL
  << OrigIndent << "}";
  // clang-format on

  return Replacement(
      SM,
      CharSourceRange(SourceRange(KCall->getLocStart(), KCall->getLocEnd()),
                      /*IsTokenRange=*/true),
      move(Final.str()));
}

Replacement ReplaceCallExpr::getReplacement(const ASTContext &Context) const {
  return Replacement(
      Context.getSourceManager(), C,
      buildCall(Name, llvm::iterator_range<decltype(begin(Args))>(Args),
                llvm::iterator_range<decltype(begin(Types))>(Types), Context));
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

Replacement
InsertBeforeCtrInitList::getReplacement(const ASTContext &Context) const {
  if (CDecl->init_begin() != CDecl->init_end()) {
    auto SLoc = CDecl->getBody()->getSourceRange().getBegin();
    auto LocInfo = Context.getSourceManager().getDecomposedLoc(SLoc);
    auto Buffer = Context.getSourceManager().getBufferData(LocInfo.first);
    auto begin = Buffer.find_last_of(':', LocInfo.second);
    if (begin == StringRef::npos) {
      begin = 0;
    }
    SLoc = SLoc.getLocWithOffset((int)begin - (int)LocInfo.second);
    return Replacement(Context.getSourceManager(), SLoc, 0, T);
  } else {
    SourceLocation Begin = CDecl->getBody()->getSourceRange().getBegin();
    return Replacement(Context.getSourceManager(), Begin, 0, T);
  }
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
