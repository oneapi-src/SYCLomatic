//===--- TextModification.cpp ---------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "TextModification.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Utility.h"

#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/Support/Path.h"

#include <sstream>

using namespace clang;
using namespace clang::dpct;
using namespace clang::tooling;

static std::unordered_set<std::string> DuplicateFilterAPI;
static std::unordered_set<std::string> DuplicateFilterSYCL;

// This function is used to collect the number of LOC (Lines Of Code) to DPC++
// and the number of LOC migrated to Compatibility API.
// When a Compatibility API replacement and a non-API replacement happen in
// the same line, only the LOC of Compatibility API is accumulated.
void recordMigrationInfo(const ASTContext &Context, const SourceLocation &SL,
                           bool IsCompatibilityAPI = false,
                           std::string APIName = "") {
  const SourceManager &SM = Context.getSourceManager();
  if (SL.isValid()) {
    const SourceLocation FileLoc = SM.getFileLoc(SL);
    std::string SLStr = FileLoc.printToString(SM);
    std::size_t PosCol = SLStr.rfind(':');
    std::size_t PosRow = SLStr.rfind(':', PosCol - 1);
    std::string FileName = SLStr.substr(0, PosRow);
    std::string LineNo = SLStr.substr(PosRow + 1, PosCol - PosRow - 1);
    std::string Key = FileName + ":" + LineNo;

    if (DuplicateFilterAPI.find(Key) == end(DuplicateFilterAPI) &&
        IsCompatibilityAPI == true) {
      LOCStaticsMap[FileName][0]++;
      DuplicateFilterAPI.insert(Key);
    }

    if (DuplicateFilterSYCL.find(Key) == end(DuplicateFilterSYCL) &&
        DuplicateFilterAPI.find(Key) == end(DuplicateFilterAPI) &&
        IsCompatibilityAPI == false) {
      LOCStaticsMap[FileName][1]++;
      DuplicateFilterSYCL.insert(Key);
    }
  }
}

std::shared_ptr<ExtReplacement>
ReplaceStmt::getReplacement(const ASTContext &Context) const {
  const SourceManager &SM = Context.getSourceManager();
  SourceLocation Begin(TheStmt->getBeginLoc()), End(TheStmt->getEndLoc());

  // If ReplaceStmt replaces calls to compatibility APIs, record the
  // OrigAPIName (not macro case)
  if (!IsProcessMacro) {
    if (IsReplaceCompatibilityAPI) {
      recordMigrationInfo(Context, TheStmt->getBeginLoc(), true, OrigAPIName);
    } else {
      recordMigrationInfo(Context, TheStmt->getBeginLoc());
    }
  }

  if (IsProcessMacro) {
    if (Begin.isMacroID()) {
      if (SM.isMacroArgExpansion(Begin))
        Begin = SM.getSpellingLoc(Begin);
      else
        Begin = SM.getExpansionLoc(Begin);
    }
    // If ReplaceStmt replaces calls to compatibility APIs, record the
    // OrigAPIName (macro case)
    if (IsReplaceCompatibilityAPI) {
      recordMigrationInfo(Context, Begin, true, OrigAPIName);
    } else {
      recordMigrationInfo(Context, Begin);
    }

    if (End.isMacroID()) {
      if (SM.isMacroArgExpansion(End))
        End = SM.getSpellingLoc(End);
      else
        End = SM.getExpansionLoc(End);
    }
    if (Begin == End) {
      End = Lexer::getLocForEndOfToken(End, 0, SM, LangOptions());
      End = End.getLocWithOffset(-1);
    }
    auto CallExprLength =
        SM.getCharacterData(End) - SM.getCharacterData(Begin) + 1;
    if (IsCleanup && ReplacementString.empty())
      return removeStmtWithCleanups(SM);
    return std::make_shared<ExtReplacement>(SM, Begin, CallExprLength,
                                            ReplacementString, this);
  } else {
    // When replacing a CallExpr with an empty string, also remove semicolons
    // and redundant spaces
    if (IsCleanup &&
        TheStmt->getStmtClass() == Stmt::StmtClass::CallExprClass &&
        ReplacementString.empty() && !IsSingleLineStatement(TheStmt)) {
      return removeStmtWithCleanups(SM);
    }
    return std::make_shared<ExtReplacement>(SM, TheStmt, ReplacementString,
                                            this);
  }
}

// Remove TheStmt together with the trailing semicolon and redundant spaces
// in the same line.
std::shared_ptr<ExtReplacement>
ReplaceStmt::removeStmtWithCleanups(const SourceManager &SM) const {
  unsigned TotalLen = 0;
  auto StmtLoc = TheStmt->getBeginLoc();
  if (StmtLoc.isInvalid() && !StmtLoc.isMacroID())
    return std::make_shared<ExtReplacement>(SM, TheStmt, ReplacementString,
                                            this);

  SourceLocation Begin(TheStmt->getBeginLoc()), Endt(TheStmt->getEndLoc());
  SourceLocation End(Lexer::getLocForEndOfToken(Endt, 0, SM, LangOptions()));
  SourceLocation LocBeforeStmt;
  const char *PosBeforeStmt;
  const char *LastLFPos;
  if (IsProcessMacro) {
    Begin = SM.getExpansionLoc(Begin);
    End = SM.getExpansionLoc(End);
    LocBeforeStmt = Begin.getLocWithOffset(-1);
    PosBeforeStmt = SM.getCharacterData(LocBeforeStmt);
    LastLFPos = PosBeforeStmt;
  } else {
    LocBeforeStmt = TheStmt->getBeginLoc().getLocWithOffset(-1);
    PosBeforeStmt = SM.getCharacterData(LocBeforeStmt);
    LastLFPos = PosBeforeStmt;
  }

  while (isspace(*LastLFPos) && *LastLFPos != '\n')
    --LastLFPos;

  SourceLocation PostLastLFLoc = Begin;
  // Get the length of spaces before the TheStmt
  if (*LastLFPos == '\n') {
    unsigned Lent = PosBeforeStmt - LastLFPos;
    PostLastLFLoc = Begin.getLocWithOffset(-Lent);
    TotalLen += Lent;
  }

  SourceLocation StmtBeginLoc = Begin;
  SourceLocation StmtEndLoc;
  const char *StmtBeginPos;
  const char *StmtEndPos;
  if (IsProcessMacro) {
    StmtEndLoc = End;
    StmtBeginPos = SM.getCharacterData(StmtBeginLoc);
    StmtEndPos = SM.getCharacterData(StmtEndLoc);
  } else {
    StmtEndLoc = TheStmt->getEndLoc();
    StmtBeginPos = SM.getCharacterData(StmtBeginLoc);
    StmtEndPos = SM.getCharacterData(StmtEndLoc);
  }

  // Get the length of TheStmt
  TotalLen += StmtEndPos - StmtBeginPos;

  // Get the length of spaces and the semicolon after the TheStmt
  SourceLocation PostStmtLoc;
  Optional<Token> TokSharedPtr;
  if (IsProcessMacro) {
    PostStmtLoc = End;
    TokSharedPtr =
        Lexer::findNextToken(End.getLocWithOffset(-1), SM, LangOptions());
  } else {
    PostStmtLoc = StmtEndLoc.getLocWithOffset(1);
    TokSharedPtr = Lexer::findNextToken(StmtEndLoc, SM, LangOptions());
  }

  if (TokSharedPtr.hasValue()) {
    Token Tok = TokSharedPtr.getValue();
    // If TheStmt has a trailing semicolon
    if (Tok.is(tok::TokenKind::semi)) {
      auto PostSemiLoc = Tok.getLocation().getLocWithOffset(1);
      auto PostSemiPos = SM.getCharacterData(PostSemiLoc);
      const char *EndPos = PostSemiPos;
      while (isspace(*EndPos) && *EndPos != '\n')
        ++EndPos;

      auto ReplaceBeginPos = SM.getCharacterData(PostStmtLoc);
      if (*EndPos == '\n') {
        unsigned Lent = EndPos - ReplaceBeginPos + 1;
        TotalLen += Lent;
      }
      if (*EndPos == '}') {
        unsigned Lent = EndPos - ReplaceBeginPos;
        TotalLen += Lent;
      }

      if (IsProcessMacro) {
        return std::make_shared<ExtReplacement>(SM, PostLastLFLoc, TotalLen, "",
                                                this);
      } else {
        return std::make_shared<ExtReplacement>(SM, PostLastLFLoc, TotalLen + 1,
                                                "", this);
      }
    }
  }

  // If semicolon is not found, just remove TheStmt
  if (IsProcessMacro) {
    return std::make_shared<ExtReplacement>(SM, Begin, TotalLen,
                                            ReplacementString, this);
  } else {
    return std::make_shared<ExtReplacement>(SM, TheStmt, ReplacementString,
                                            this);
  }
}

std::shared_ptr<ExtReplacement>
ReplaceCalleeName::getReplacement(const ASTContext &Context) const {
  const SourceManager &SM = Context.getSourceManager();
  recordMigrationInfo(Context, C->getBeginLoc(), true, OrigAPIName);
  return std::make_shared<ExtReplacement>(
      Context.getSourceManager(), SM.getSpellingLoc(C->getBeginLoc()),
      getCalleeName(Context).size(), ReplStr, this);
}

std::shared_ptr<ExtReplacement>
RemoveAttr::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  SourceRange AttrRange = TheAttr->getRange();
  SourceLocation ARB = AttrRange.getBegin();
  SourceLocation ARE = AttrRange.getEnd();
  SourceLocation ExpB = SM.getExpansionLoc(ARB);
  // No need to invoke getExpansionLoc again if the location is the same.
  SourceLocation ExpE = (ARB == ARE) ? ExpB : SM.getExpansionLoc(ARE);

  SourceLocation SpellingBegin = SM.getSpellingLoc(ExpB);
  SourceLocation SpellingEnd = SM.getSpellingLoc(ExpE);
  std::pair<FileID, unsigned> Start = SM.getDecomposedLoc(SpellingBegin);
  std::pair<FileID, unsigned> End = SM.getDecomposedLoc(SpellingEnd);
  End.second += Lexer::MeasureTokenLength(SpellingEnd, SM, LangOptions());
  unsigned Len = End.second - Start.second;
  // check the char after attribute, if it is empty then del it.
  //   -eg. will del the space in case  "__global__ "
  //   -eg. will not del the ";" in  case "__global__;"
  unsigned int I = 0;
  while (SM.getCharacterData(ExpB.getLocWithOffset(Len), 0)[I] == ' ' ||
         SM.getCharacterData(ExpB.getLocWithOffset(Len), 0)[I] == '\t') {
    I++;
  }
  Len += I;

  recordMigrationInfo(Context, TheAttr->getLocation());

  return std::make_shared<ExtReplacement>(
      SM, CharSourceRange::getCharRange(ExpB, ExpB.getLocWithOffset(Len)), "",
      this);
}

std::map<unsigned, ReplaceVarDecl *> ReplaceVarDecl::ReplaceMap;

std::shared_ptr<ExtReplacement>
ReplaceTypeInDecl::getReplacement(const ASTContext &Context) const {
  recordMigrationInfo(Context, DD->getBeginLoc());
  return std::make_shared<ExtReplacement>(Context.getSourceManager(), &TL, T,
                                          this);
}

ReplaceVarDecl *ReplaceVarDecl::getVarDeclReplacement(const VarDecl *VD,
                                                      std::string &&Text) {
  auto LocID = VD->getBeginLoc().getRawEncoding();
  auto Itr = ReplaceMap.find(LocID);
  if (Itr == ReplaceMap.end())
    return ReplaceMap
        .insert(std::map<unsigned, ReplaceVarDecl *>::value_type(
            LocID, new ReplaceVarDecl(VD, std::move(Text))))
        .first->second;
  Itr->second->addVarDecl(VD, std::move(Text));
  return nullptr;
}

ReplaceVarDecl::ReplaceVarDecl(const VarDecl *D, std::string &&Text)
    : TextModification(TMID::ReplaceVarDecl), D(D),
      SR(DpctGlobalInfo::getSourceManager().getExpansionRange(
          D->getSourceRange())),
      T(std::move(Text)),
      Indent(getIndent(SR.getBegin(), DpctGlobalInfo::getSourceManager())),
      NL(getNL()) {}

void ReplaceVarDecl::addVarDecl(const VarDecl *VD, std::string &&Text) {
  SourceManager &SM = DpctGlobalInfo::getSourceManager();
  CharSourceRange Range = SM.getExpansionRange(VD->getSourceRange());
  if (SM.getCharacterData(Range.getEnd()) > SM.getCharacterData(SR.getEnd()))
    SR = Range;
  T += NL + Indent + Text;
}

std::shared_ptr<ExtReplacement>
ReplaceVarDecl::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  size_t repLength;
  repLength =
      SM.getCharacterData(SR.getEnd()) - SM.getCharacterData(SR.getBegin()) + 1;
  // try to del  "    ;" in var declare
  auto DataAfter = SM.getCharacterData(SR.getBegin());
  auto Data = DataAfter[repLength];
  while (Data != ';')
    Data = DataAfter[++repLength];
  recordMigrationInfo(Context, SR.getBegin());
  return std::make_shared<ExtReplacement>(Context.getSourceManager(),
                                          SR.getBegin(), ++repLength, T, this);
}

std::shared_ptr<ExtReplacement>
ReplaceReturnType::getReplacement(const ASTContext &Context) const {
  SourceRange SR = FD->getReturnTypeSourceRange();
  recordMigrationInfo(Context, FD->getBeginLoc());
  return std::make_shared<ExtReplacement>(Context.getSourceManager(),
                                          CharSourceRange(SR, true), T, this);
}

std::shared_ptr<ExtReplacement>
ReplaceToken::getReplacement(const ASTContext &Context) const {
  recordMigrationInfo(Context, Begin);
  // Need to deal with the fact, that the type name might be a macro.
  return std::make_shared<ExtReplacement>(
      Context.getSourceManager(),
      // false means [Begin, End)
      // true means [Begin, End]
      CharSourceRange(SourceRange(Begin, End), true), T, this);
}

std::shared_ptr<ExtReplacement>
InsertText::getReplacement(const ASTContext &Context) const {
  recordMigrationInfo(Context, Begin);
  // Need to deal with the fact, that the type name might be a macro.
  auto R = std::make_shared<ExtReplacement>(
      Context.getSourceManager(),
      // false means [Begin, End)
      // true means [Begin, End]
      CharSourceRange(SourceRange(Begin, Begin), false), T, this);
  R->setPairID(PairID);
  return R;
}

std::shared_ptr<ExtReplacement>
ReplaceCCast::getReplacement(const ASTContext &Context) const {
  auto Begin = Cast->getLParenLoc();
  auto End = Cast->getRParenLoc();
  recordMigrationInfo(Context, Cast->getBeginLoc());
  return std::make_shared<ExtReplacement>(
      Context.getSourceManager(),
      CharSourceRange(SourceRange(Begin, End), true), TypeName, this);
}

std::shared_ptr<ExtReplacement>
RenameFieldInMemberExpr::getReplacement(const ASTContext &Context) const {
  SourceLocation SL = ME->getEndLoc();
  SourceLocation Begin = SL;
  if (PositionOfDot != 0) {
    // Cover dot position when migrate dim3.x/y/z to
    // cl::sycl::range<3>[0]/[1]/[2].
    Begin = ME->getBeginLoc();
    Begin = Begin.getLocWithOffset(PositionOfDot);
  }
  recordMigrationInfo(Context, ME->getBeginLoc());
  return std::make_shared<ExtReplacement>(
      Context.getSourceManager(), CharSourceRange(SourceRange(Begin, SL), true),
      T, this);
}

std::shared_ptr<ExtReplacement>
InsertAfterStmt::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  auto Loc = SM.getSpellingLoc(S->getEndLoc());
  Loc = Loc.getLocWithOffset(
      Lexer::MeasureTokenLength(Loc, SM, Context.getLangOpts()));
  if (DoMacroExpansion) {
    if (S->getEndLoc().isMacroID()) {
      auto TokenBegin = SM.getExpansionLoc(S->getEndLoc());
      // If current macro is inside another macro or is a macro arg
      // we can only modify the spelling part
      // BeginLoc is more accurate than EndLoc
      if (S->getBeginLoc().isMacroID() && !SM.isAtStartOfImmediateMacroExpansion(S->getBeginLoc())) {
        TokenBegin = SM.getSpellingLoc(S->getEndLoc());
      }
      auto Len = Lexer::MeasureTokenLength(TokenBegin, SM, LangOptions());
      Loc = TokenBegin.getLocWithOffset(Len);
    }
  }
  recordMigrationInfo(Context, Loc);
  auto R = std::make_shared<ExtReplacement>(SM, Loc, 0, T, this);
  R->setPairID(PairID);
  return R;
}

static int getExpansionRangeSize(const SourceManager &Sources,
                                 const CharSourceRange &Range,
                                 const LangOptions &LangOpts) {
  SourceLocation ExpansionBegin = Sources.getExpansionLoc(Range.getBegin());
  SourceLocation ExpansionEnd = Sources.getExpansionLoc(Range.getEnd());
  std::pair<FileID, unsigned> Start = Sources.getDecomposedLoc(ExpansionBegin);
  std::pair<FileID, unsigned> End = Sources.getDecomposedLoc(ExpansionEnd);
  if (Start.first != End.first)
    return -1;
  if (Range.isTokenRange())
    End.second += Lexer::MeasureTokenLength(ExpansionEnd, Sources, LangOpts);
  return End.second - Start.second;
}

static std::tuple<StringRef, unsigned, unsigned>
getReplacementInfo(const ASTContext &Context, const CharSourceRange &Range) {
  const auto &SM = Context.getSourceManager();
  const auto &ExpansionBegin = SM.getExpansionLoc(Range.getBegin());
  const std::pair<FileID, unsigned> DecomposedLocation =
      SM.getDecomposedLoc(ExpansionBegin);
  const FileEntry *Entry = SM.getFileEntryForID(DecomposedLocation.first);
  StringRef FilePath = Entry ? Entry->getName() : "";
  unsigned Offset = DecomposedLocation.second;
  unsigned Length = getExpansionRangeSize(SM, Range, LangOptions());
  return std::make_tuple(FilePath, Offset, Length);
}

std::shared_ptr<ExtReplacement>
ReplaceInclude::getReplacement(const ASTContext &Context) const {
  recordMigrationInfo(Context, Range.getBegin());
  // Make replacements for macros happen in expansion locations, rather than
  // spelling locations
  if (Range.getBegin().isMacroID() || Range.getEnd().isMacroID()) {
    StringRef FilePath;
    unsigned Offset, Length;
    std::tie(FilePath, Offset, Length) = getReplacementInfo(Context, Range);
    return std::make_shared<ExtReplacement>(FilePath, Offset, Length, T, this);
  }

  return std::make_shared<ExtReplacement>(Context.getSourceManager(), Range, T,
                                          this);
}

void ReplaceDim3Ctor::setRange() {
  if (isDecl) {
    SourceRange SR = Ctor->getParenOrBraceRange();
    if (SR.isInvalid()) {
      // dim3 a;
      auto CtorEndLoc = Lexer::getLocForEndOfToken(
          Ctor->getLocation(), 0, DpctGlobalInfo::getSourceManager(),
          DpctGlobalInfo::getContext().getLangOpts());
      CSR = CharSourceRange(SourceRange(CtorEndLoc, CtorEndLoc), false);
    } else {
      SourceRange SR1 =
          SourceRange(SR.getBegin().getLocWithOffset(1), SR.getEnd());
      CSR = CharSourceRange(SR1, false);
    }
  } else {
    // adjust the statement to replace if top-level constructor includes the
    // variable being defined
    const Stmt *S = getReplaceStmt(Ctor);
    if (!S) {
      return;
    }
    CSR = CharSourceRange::getTokenRange(S->getSourceRange());
  }
}

ReplaceInclude *ReplaceDim3Ctor::getEmpty() {
  return new ReplaceInclude(CSR, "");
}

// Strips possible Materialize and Cast operators from CXXConstructor
const CXXConstructExpr *ReplaceDim3Ctor::getConstructExpr(const Expr *E) {
  if (auto C = dyn_cast_or_null<CXXConstructExpr>(E)) {
    return C;
  } else if (isa<MaterializeTemporaryExpr>(E)) {
    return getConstructExpr(
        dyn_cast<MaterializeTemporaryExpr>(E)->GetTemporaryExpr());
  } else if (isa<CastExpr>(E)) {
    return getConstructExpr(dyn_cast<CastExpr>(E)->getSubExpr());
  } else {
    return nullptr;
  }
}

// Returns the full replacement string for the CXXConstructorExpr
std::string
ReplaceDim3Ctor::getSyclRangeCtor(const CXXConstructExpr *Ctor) const {
  ExprAnalysis Analysis(Ctor);
  Analysis.analyze();
  return Analysis.getReplacedString();
}

const Stmt *ReplaceDim3Ctor::getReplaceStmt(const Stmt *S) const {
  if (auto Ctor = dyn_cast_or_null<CXXConstructExpr>(S)) {
    if (Ctor->getNumArgs() == 1) {
      return getConstructExpr(Ctor->getArg(0));
    }
  }
  return S;
}

std::string ReplaceDim3Ctor::getReplaceString() const {
  if (isDecl) {
    // Get the new parameter list for the replaced constructor, without the
    // parens
    std::string ReplacedString = getSyclRangeCtor(Ctor);
    ReplacedString.replace(0, strlen("cl::sycl::range<3>("), "");
    ReplacedString.replace(ReplacedString.length() - 1, 1, "");
    if (Ctor->getParenOrBraceRange().isInvalid()) {
      // dim3 = a;
      ReplacedString = "(" + ReplacedString + ")";
    }
    return ReplacedString;
  } else {
    std::string S;
    if (FinalCtor) {
      S = getSyclRangeCtor(FinalCtor);
    } else {
      S = getSyclRangeCtor(Ctor);
    }
    return S;
  }
}

std::shared_ptr<ExtReplacement>
ReplaceDim3Ctor::getReplacement(const ASTContext &Context) const {
  recordMigrationInfo(Context, CSR.getBegin());
  // Make replacements for macros happen in expansion locations, rather than
  // spelling locations
  if (CSR.getBegin().isMacroID() || CSR.getEnd().isMacroID()) {
    StringRef FilePath;
    unsigned Offset, Length;
    std::tie(FilePath, Offset, Length) = getReplacementInfo(Context, CSR);
    return std::make_shared<ExtReplacement>(FilePath, Offset, Length,
                                            getReplaceString(), this);
  }

  ReplacementString = getReplaceString();
  return std::make_shared<ExtReplacement>(
      Context.getSourceManager(), CSR.getBegin(), 0, ReplacementString, this);
}

std::shared_ptr<ExtReplacement>
InsertComment::getReplacement(const ASTContext &Context) const {
  auto NL = getNL();
  auto OrigIndent = getIndent(SL, Context.getSourceManager()).str();
  return std::make_shared<ExtReplacement>(Context.getSourceManager(), SL, 0,
                                          (OrigIndent + llvm::Twine("/*") + NL +
                                           OrigIndent + Text + NL + OrigIndent +
                                           "*/" + NL)
                                              .str(),
                                          this);
}

std::string printTemplateArgument(const TemplateArgument &Arg,
                                  const PrintingPolicy &PP) {
  std::string Out;
  llvm::raw_string_ostream OS(Out);
  Arg.print(PP, OS);
  return OS.str();
}

SourceLocation InsertBeforeCtrInitList::getInsertLoc() const {
  auto Init = CDecl->init_begin();
  while (Init != CDecl->init_end()) {
    auto InitLoc = (*Init)->getSourceLocation();
    if (InitLoc.isValid() && (*Init)->isWritten()) {
      // Try to insert before ":"
      int i = 0;
      auto Data =
          DpctGlobalInfo::getSourceManager().getCharacterData(InitLoc);
      while (Data[i] != ':')
        --i;
      return InitLoc.getLocWithOffset(i);
    }
    ++Init;
  }
  return CDecl->getBody()->getBeginLoc();
}

std::shared_ptr<ExtReplacement>
InsertBeforeCtrInitList::getReplacement(const ASTContext &Context) const {
  recordMigrationInfo(Context, CDecl->getBeginLoc());
  return std::make_shared<ExtReplacement>(Context.getSourceManager(),
                                          getInsertLoc(), 0, T, this);
}

std::shared_ptr<ExtReplacement>
InsertBeforeStmt::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  SourceLocation Begin = S->getSourceRange().getBegin();
  if (DoMacroExpansion) {
    if (Begin.isMacroID()) {
      if (!SM.isAtStartOfImmediateMacroExpansion(Begin)) {
        // If current macro is inside another macro or is a macro arg
        // we can only modify the spelling part
        Begin = SM.getSpellingLoc(Begin);
      }
      else {
        Begin = SM.getExpansionLoc(Begin);
      }
    }
  }
  recordMigrationInfo(Context, Begin);
  auto R = std::make_shared<ExtReplacement>(
      Context.getSourceManager(),
      CharSourceRange(SourceRange(Begin, Begin), false), T, this);
  R->setPairID(PairID);
  return R;
}

std::shared_ptr<ExtReplacement>
RemoveArg::getReplacement(const ASTContext &Context) const {
  SourceRange SR = CE->getArg(N)->getSourceRange();
  SourceLocation Begin = SR.getBegin();
  SourceLocation End;
  bool IsLast = (N == (CE->getNumArgs() - 1));
  if (IsLast) {
    End = SR.getEnd();
  } else {
    End = CE->getArg(N + 1)->getSourceRange().getBegin().getLocWithOffset(0);
  }
  recordMigrationInfo(Context, CE->getBeginLoc());
  return std::make_shared<ExtReplacement>(
      Context.getSourceManager(),
      CharSourceRange(SourceRange(Begin, End), false), "", this);
}

std::shared_ptr<ExtReplacement>
InsertClassName::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  auto BeginLoc = CD->getBeginLoc();
  auto DataBegin = SM.getCharacterData(BeginLoc);

  unsigned i = 0;
  auto Data = DataBegin[i];
  while ((Data != ':') && (Data != '{'))
    Data = DataBegin[++i];

  Data = DataBegin[--i];
  while ((Data == ' ') || (Data == '\t') || (Data == '\n') || (Data == '\r'))
    Data = DataBegin[--i];

  recordMigrationInfo(Context, CD->getBeginLoc());
  return std::make_shared<ExtReplacement>(
      SM, BeginLoc.getLocWithOffset(i + 1), 0,
      " dpct_type_" +
          getHashAsString(BeginLoc.printToString(SM)).substr(0, 6),
      this);
}

std::shared_ptr<ExtReplacement>
ReplaceText::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  if (IsReplaceCompatibilityAPI) {
    recordMigrationInfo(Context, BeginLoc, true, OrigAPIName);
  } else {
    recordMigrationInfo(Context, BeginLoc);
  }
  return std::make_shared<ExtReplacement>(SM, BeginLoc, Len, T, this);
}

const std::unordered_map<int, std::string> TextModification::TMNameMap = {
#define TRANSFORMATION(TYPE) {(int)TMID::TYPE, #TYPE},
#include "Transformations.inc"
#undef TRANSFORMATION
};

const std::string &TextModification::getName() const {
  return TMNameMap.at((int)getID());
}

constexpr char TransformStr[] = " => ";
static void printHeader(llvm::raw_ostream &OS, const TMID &ID,
                        const char *ParentRuleID) {
  OS << "[";
  if (ParentRuleID) {
    OS << ASTTraversalMetaInfo::getNameTable()[ParentRuleID] << ":";
  }
  OS << TextModification::TMNameMap.at((int)ID);
  OS << "] ";
}

static void printLocation(llvm::raw_ostream &OS, const SourceLocation &SL,
                          ASTContext &Context, const bool PrintDetail) {
  const SourceManager &SM = Context.getSourceManager();
  if (PrintDetail) {
    SL.print(OS, SM);
  } else {
    const SourceLocation FileLoc = SM.getFileLoc(SL);
    std::string SLStr = FileLoc.printToString(SM);
    OS << llvm::sys::path::filename(SLStr);
  }
  OS << " ";
}

static void printInsertion(llvm::raw_ostream &OS,
                           const std::string &Insertion) {
  OS << TransformStr << Insertion << "\n";
}

static void printReplacement(llvm::raw_ostream &OS,
                             const std::string &Replacement) {
  OS << TransformStr;
  OS << "\"" << Replacement << "\"\n";
}

void ReplaceStmt::print(llvm::raw_ostream &OS, ASTContext &Context,
                        const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, TheStmt->getBeginLoc(), Context, PrintDetail);
  TheStmt->printPretty(OS, nullptr, PrintingPolicy(Context.getLangOpts()));
  printReplacement(OS, ReplacementString);
}

void ReplaceCalleeName::print(llvm::raw_ostream &OS, ASTContext &Context,
                              const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, C->getBeginLoc(), Context, PrintDetail);
  OS << getCalleeName(Context);
  printReplacement(OS, ReplStr);
}

void RemoveAttr::print(llvm::raw_ostream &OS, ASTContext &Context,
                       const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, TheAttr->getLocation(), Context, PrintDetail);
  TheAttr->printPretty(OS, PrintingPolicy(Context.getLangOpts()));
  printReplacement(OS, "");
}

void ReplaceTypeInDecl::print(llvm::raw_ostream &OS, ASTContext &Context,
                              const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, DD->getBeginLoc(), Context, PrintDetail);
  DD->print(OS, PrintingPolicy(Context.getLangOpts()));
  printReplacement(OS, T);
}

void ReplaceVarDecl::print(llvm::raw_ostream &OS, ASTContext &Context,
                           const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, SR.getBegin(), Context, PrintDetail);
  D->print(OS, PrintingPolicy(Context.getLangOpts()));
  printReplacement(OS, T);
}

void ReplaceReturnType::print(llvm::raw_ostream &OS, ASTContext &Context,
                              const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, FD->getBeginLoc(), Context, PrintDetail);
  FD->print(OS, PrintingPolicy(Context.getLangOpts()));
  printReplacement(OS, T);
}

void ReplaceToken::print(llvm::raw_ostream &OS, ASTContext &Context,
                         const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, Begin, Context, PrintDetail);
  printReplacement(OS, T);
}

void InsertText::print(llvm::raw_ostream &OS, ASTContext &Context,
                       const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, Begin, Context, PrintDetail);
  printInsertion(OS, T);
}

void ReplaceCCast::print(llvm::raw_ostream &OS, ASTContext &Context,
                         const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, Cast->getBeginLoc(), Context, PrintDetail);
  Cast->printPretty(OS, nullptr, PrintingPolicy(Context.getLangOpts()));
  printReplacement(OS, TypeName);
}

void RenameFieldInMemberExpr::print(llvm::raw_ostream &OS, ASTContext &Context,
                                    const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, ME->getBeginLoc(), Context, PrintDetail);
  ME->printPretty(OS, nullptr, PrintingPolicy(Context.getLangOpts()));
  printReplacement(OS, T);
}

void InsertAfterStmt::print(llvm::raw_ostream &OS, ASTContext &Context,
                            const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, S->getEndLoc(), Context, PrintDetail);
  printInsertion(OS, T);
}

void ReplaceInclude::print(llvm::raw_ostream &OS, ASTContext &Context,
                           const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, Range.getBegin(), Context, PrintDetail);
  // TODO: 1. Find a way to show replaced include briefly
  //       2. ReplaceDim3Ctor uses ReplaceInclude, need to clarification
  printReplacement(OS, T);
}

void ReplaceDim3Ctor::print(llvm::raw_ostream &OS, ASTContext &Context,
                            const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, CSR.getBegin(), Context, PrintDetail);
  Ctor->printPretty(OS, nullptr, PrintingPolicy(Context.getLangOpts()));
  printReplacement(OS, ReplacementString);
}

void InsertComment::print(llvm::raw_ostream &OS, ASTContext &Context,
                          const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, SL, Context, PrintDetail);
  printInsertion(OS, Text);
}

void InsertBeforeCtrInitList::print(llvm::raw_ostream &OS, ASTContext &Context,
                                    const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, CDecl->getBeginLoc(), Context, PrintDetail);
  CDecl->print(OS, PrintingPolicy(Context.getLangOpts()));
  printInsertion(OS, T);
}

void InsertBeforeStmt::print(llvm::raw_ostream &OS, ASTContext &Context,
                             const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  SourceLocation Begin = S->getSourceRange().getBegin();
  if (DoMacroExpansion) {
    auto &SM = Context.getSourceManager();
    if (Begin.isMacroID())
      Begin = SM.getExpansionLoc(Begin);
  }
  printLocation(OS, Begin, Context, PrintDetail);
  S->printPretty(OS, nullptr, PrintingPolicy(Context.getLangOpts()));
  printReplacement(OS, T);
}

void RemoveArg::print(llvm::raw_ostream &OS, ASTContext &Context,
                      const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, CE->getBeginLoc(), Context, PrintDetail);
  CE->printPretty(OS, nullptr, PrintingPolicy(Context.getLangOpts()));
  printReplacement(OS, "");
}

void InsertClassName::print(llvm::raw_ostream &OS, ASTContext &Context,
                            const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, CD->getBeginLoc(), Context, PrintDetail);
  CD->print(OS, PrintingPolicy(Context.getLangOpts()));
  printInsertion(OS, "");
}

void ReplaceText::print(llvm::raw_ostream &OS, ASTContext &Context,
                        const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, BeginLoc, Context, PrintDetail);
  printInsertion(OS, T);
}
