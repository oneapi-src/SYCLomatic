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
using namespace clang::syclct;
using namespace clang::tooling;

static std::unordered_set<std::string> DuplicateFilter;

void recordTranslationInfo(const ASTContext &Context, const SourceLocation &SL,
                           bool IsCompatibilityAPI = false,
                           std::string APIName = "") {
  const SourceManager &SM = Context.getSourceManager();
  if (SL.isValid()) {
    const SourceLocation FileLoc = SM.getFileLoc(SL);
    std::string SLStr = FileLoc.printToString(SM);

    std::size_t Pos = SLStr.find(':');
    std::string FileName = SLStr.substr(0, Pos);
    std::size_t PosNext = SLStr.find(':', Pos + 1);
    std::string LineNo = SLStr.substr(Pos + 1, PosNext - Pos - 1);

    std::string Key = FileName + ":" + LineNo;

    if (DuplicateFilter.find(Key) == end(DuplicateFilter) ||
        IsCompatibilityAPI == true) {
      if (IsCompatibilityAPI) {
        if (DuplicateFilter.find(Key) != end(DuplicateFilter)) {
          // when syclct api replacement and non-api SYCL replacement happen in
          // the same line, only count line number to syclct api accumulation.
          LOCStaticsMap[FileName][1]--;
        }
        LOCStaticsMap[FileName][0]++;

        if (!APIName.empty()) {
          std::string LocKey = APIName + "," + "true";
          APIStaticsMap[LocKey]++;
        }

      } else {
        LOCStaticsMap[FileName][1]++;
      }
      DuplicateFilter.insert(Key);
    }
  }
}

ExtReplacement ReplaceStmt::getReplacement(const ASTContext &Context) const {
  // If ReplaceStmt replaces calls to compatibility APIs, record the OrigAPIName
  if (IsReplaceCompatibilityAPI) {
    recordTranslationInfo(Context, TheStmt->getBeginLoc(), true, OrigAPIName);
  } else {
    recordTranslationInfo(Context, TheStmt->getBeginLoc());
  }
  return ExtReplacement(Context.getSourceManager(), TheStmt, ReplacementString,
                        this);
}

ExtReplacement
ReplaceCalleeName::getReplacement(const ASTContext &Context) const {
  const SourceManager &SM = Context.getSourceManager();
  recordTranslationInfo(Context, C->getBeginLoc(), true, OrigAPIName);
  return ExtReplacement(Context.getSourceManager(),
                        SM.getSpellingLoc(C->getBeginLoc()),
                        getCalleeName(Context).size(), ReplStr, this);
}

ExtReplacement RemoveAttr::getReplacement(const ASTContext &Context) const {
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

  recordTranslationInfo(Context, TheAttr->getLocation());

  return ExtReplacement(
      SM, CharSourceRange::getCharRange(ExpB, ExpB.getLocWithOffset(Len)), "",
      this);
}

std::map<unsigned, ReplaceVarDecl *> ReplaceVarDecl::ReplaceMap;

ExtReplacement
ReplaceTypeInDecl::getReplacement(const ASTContext &Context) const {
  if (D) {
    recordTranslationInfo(Context, D->getBeginLoc());
  } else {
    recordTranslationInfo(Context, FD->getBeginLoc());
  }
  return ExtReplacement(Context.getSourceManager(), &TL, T, this);
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
      SR(SyclctGlobalInfo::getSourceManager().getExpansionRange(
          D->getSourceRange())),
      T(std::move(Text)),
      Indent(getIndent(SR.getBegin(), SyclctGlobalInfo::getSourceManager())),
      NL(getNL(SR.getBegin(), SyclctGlobalInfo::getSourceManager())) {}

void ReplaceVarDecl::addVarDecl(const VarDecl *VD, std::string &&Text) {
  SourceManager &SM = SyclctGlobalInfo::getSourceManager();
  CharSourceRange Range = SM.getExpansionRange(VD->getSourceRange());
  if (SM.getCharacterData(Range.getEnd()) > SM.getCharacterData(SR.getEnd()))
    SR = Range;
  T += NL + Indent + Text;
}

ExtReplacement ReplaceVarDecl::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  size_t repLength;
  repLength =
      SM.getCharacterData(SR.getEnd()) - SM.getCharacterData(SR.getBegin()) + 1;
  // try to del  "    ;" in var declare
  auto DataAfter = SM.getCharacterData(SR.getBegin());
  auto Data = DataAfter[repLength];
  while (Data != ';')
    Data = DataAfter[++repLength];
  recordTranslationInfo(Context, SR.getBegin());
  return ExtReplacement(Context.getSourceManager(), SR.getBegin(), ++repLength,
                        T, this);
}

ExtReplacement
ReplaceReturnType::getReplacement(const ASTContext &Context) const {
  SourceRange SR = FD->getReturnTypeSourceRange();
  recordTranslationInfo(Context, FD->getBeginLoc());
  return ExtReplacement(Context.getSourceManager(), CharSourceRange(SR, true),
                        T, this);
}

ExtReplacement ReplaceToken::getReplacement(const ASTContext &Context) const {
  recordTranslationInfo(Context, Begin);
  // Need to deal with the fact, that the type name might be a macro.
  return ExtReplacement(Context.getSourceManager(),
                        // false means [Begin, End)
                        // true means [Begin, End]
                        CharSourceRange(SourceRange(Begin, End), true), T,
                        this);
}

ExtReplacement InsertText::getReplacement(const ASTContext &Context) const {
  recordTranslationInfo(Context, Begin);
  // Need to deal with the fact, that the type name might be a macro.
  return ExtReplacement(Context.getSourceManager(),
                        // false means [Begin, End)
                        // true means [Begin, End]
                        CharSourceRange(SourceRange(Begin, Begin), false), T,
                        this);
}

ExtReplacement
InsertNameSpaceInVarDecl::getReplacement(const ASTContext &Context) const {
  TypeLoc TL = D->getTypeSourceInfo()->getTypeLoc();
  ExtReplacement R(
      Context.getSourceManager(),
      CharSourceRange(SourceRange(TL.getBeginLoc(), TL.getBeginLoc()), false),
      T, this);
  R.setInsertPosition(InsertPositionRight);
  recordTranslationInfo(Context, D->getBeginLoc());
  return R;
}

ExtReplacement
InsertNameSpaceInCastExpr::getReplacement(const ASTContext &Context) const {
  recordTranslationInfo(Context, D->getBeginLoc());
  return ExtReplacement(
      Context.getSourceManager(),
      CharSourceRange(SourceRange(D->getLParenLoc().getLocWithOffset(1),
                                  D->getLParenLoc().getLocWithOffset(1)),
                      false),
      T, this);
}

ExtReplacement ReplaceCCast::getReplacement(const ASTContext &Context) const {
  auto Begin = Cast->getLParenLoc();
  auto End = Cast->getRParenLoc();
  recordTranslationInfo(Context, Cast->getBeginLoc());
  return ExtReplacement(Context.getSourceManager(),
                        CharSourceRange(SourceRange(Begin, End), true),
                        TypeName, this);
}

ExtReplacement
RenameFieldInMemberExpr::getReplacement(const ASTContext &Context) const {
  SourceLocation SL = ME->getEndLoc();
  SourceLocation Begin = SL;
  if (PositionOfDot != 0) {
    // Cover dot position when migrate dim3.x/y/z to
    // cl::sycl::range<3>[0]/[1]/[2].
    Begin = ME->getBeginLoc();
    Begin = Begin.getLocWithOffset(PositionOfDot);
  }
  recordTranslationInfo(Context, ME->getBeginLoc());
  return ExtReplacement(Context.getSourceManager(),
                        CharSourceRange(SourceRange(Begin, SL), true), T, this);
}

ExtReplacement
InsertAfterStmt::getReplacement(const ASTContext &Context) const {
  CharSourceRange CSR = CharSourceRange(S->getSourceRange(), false);
  SourceLocation Loc = CSR.getEnd();
  auto &SM = Context.getSourceManager();
  auto &Opts = Context.getLangOpts();
  SourceLocation SpellLoc = SM.getSpellingLoc(Loc);
  unsigned Offs = Lexer::MeasureTokenLength(SpellLoc, SM, Opts);
  SourceLocation LastTokenBegin = Lexer::GetBeginningOfToken(Loc, SM, Opts);
  SourceLocation End = LastTokenBegin.getLocWithOffset(Offs);
  recordTranslationInfo(Context, S->getEndLoc());
  return ExtReplacement(SM, CharSourceRange(SourceRange(End, End), false), T,
                        this);
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

ExtReplacement ReplaceInclude::getReplacement(const ASTContext &Context) const {
  recordTranslationInfo(Context, Range.getBegin());
  // Make replacements for macros happen in expansion locations, rather than
  // spelling locations
  if (Range.getBegin().isMacroID() || Range.getEnd().isMacroID()) {
    StringRef FilePath;
    unsigned Offset, Length;
    std::tie(FilePath, Offset, Length) = getReplacementInfo(Context, Range);
    return ExtReplacement(FilePath, Offset, Length, T, this);
  }

  return ExtReplacement(Context.getSourceManager(), Range, T, this);
}

void ReplaceDim3Ctor::setRange() {
  if (isDecl) {
    SourceRange SR = Ctor->getParenOrBraceRange();
    SourceRange SR1 =
        SourceRange(SR.getBegin().getLocWithOffset(1), SR.getEnd());
    CSR = CharSourceRange(SR1, false);
  } else {
    // adjust the statement to replace if top-level constructor includes the
    // variable being defined
    const Stmt *S = getReplaceStmt(Ctor);
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
std::string ReplaceDim3Ctor::getSyclRangeCtor(const CXXConstructExpr *Ctor,
                                              const ASTContext &Context) const {
  return "cl::sycl::range<3>(" + getParamsString(Ctor, Context) + ")";
}

// Returns the new parameter list for the replaced constructor, without the
// parens
std::string ReplaceDim3Ctor::getParamsString(const CXXConstructExpr *Ctor,
                                             const ASTContext &Context) const {
  std::string Params = "";

  if (Ctor->getNumArgs() == 1) {
    if (auto E = getConstructExpr(Ctor->getArg(0))) {
      return getSyclRangeCtor(E, Context);
    } else {
      return getStmtSpelling(Ctor->getArg(0), Context);
    }
  } else {
    for (const auto *Arg : Ctor->arguments()) {
      if (!Params.empty()) {
        Params += ", ";
      }
      if (isa<CXXDefaultArgExpr>(Arg)) {
        Params += "1";
      } else {
        if (Arg->getBeginLoc().isMacroID() || Arg->getEndLoc().isMacroID())
          Params += getStmtSpellingWithTransforms(Arg, Context, SSM, true);
        else
          Params += getStmtSpellingWithTransforms(Arg, Context, SSM);
      }
    }
    return Params;
  }
}

const Stmt *ReplaceDim3Ctor::getReplaceStmt(const Stmt *S) const {
  if (auto Ctor = dyn_cast_or_null<CXXConstructExpr>(S)) {
    if (Ctor->getNumArgs() == 1) {
      return getConstructExpr(Ctor->getArg(0));
    }
  }
  return S;
}

std::string ReplaceDim3Ctor::getReplaceString(const ASTContext &Context) const {
  if (isDecl) {
    return getParamsString(Ctor, Context);
  } else {
    std::string S;
    if (FinalCtor) {
      S = getSyclRangeCtor(FinalCtor, Context);
    } else {
      S = getSyclRangeCtor(Ctor, Context);
    }
    SSM->insert({Ctor, S});
    return S;
  }
}

ExtReplacement
ReplaceDim3Ctor::getReplacement(const ASTContext &Context) const {
  recordTranslationInfo(Context, CSR.getBegin());
  // Make replacements for macros happen in expansion locations, rather than
  // spelling locations
  if (CSR.getBegin().isMacroID() || CSR.getEnd().isMacroID()) {
    StringRef FilePath;
    unsigned Offset, Length;
    std::tie(FilePath, Offset, Length) = getReplacementInfo(Context, CSR);
    return ExtReplacement(FilePath, Offset, Length, getReplaceString(Context),
                          this);
  }

  ReplacementString = getReplaceString(Context);
  return ExtReplacement(Context.getSourceManager(), CSR.getBegin(), 0,
                        ReplacementString, this);
}

ExtReplacement InsertComment::getReplacement(const ASTContext &Context) const {
  auto NL = getNL(SL, Context.getSourceManager());
  auto OrigIndent = getIndent(SL, Context.getSourceManager()).str();
  return ExtReplacement(Context.getSourceManager(), SL, 0,
                        (OrigIndent + llvm::Twine("/*") + NL + OrigIndent +
                         Text + NL + OrigIndent + "*/" + NL)
                            .str(),
                        this, true /*true means comments replacement*/);
}

// TODO: Remove this workaround
//
//       Current kernel call's argument are generated separately (buildArgList)
//       from AST replacement rules, that is AST replacement rules are not
//       applied here, this workaround does the replacement again here
//
//       Kernel call replacement (ReplaceKernelCallExpr) should be implemented
//       with fine-grained replacement to work with other replacement rules
//       instead of generating the replacement at once.
static inline std::string ReplacedArgText(const Expr *A,
                                          const ASTContext &Context) {
  std::string Elem = getStmtSpelling(A, Context);
  if (const MemberExpr *ME = dyn_cast<MemberExpr>(A->IgnoreImpCasts())) {
    const std::string MemberName = ME->getMemberNameInfo().getAsString();
    auto Search = MapNames::Dim3MemberNamesMap.find(MemberName);
    if (Search != MapNames::Dim3MemberNamesMap.end()) {
      static constexpr char Dot[] = ".";
      assert(Elem.find_last_of(Dot) != std::string::npos);
      Elem.replace(Elem.find_last_of(Dot), MemberName.length() + strlen(Dot),
                   Search->second);
    }
  }
  return Elem;
}

template <typename ArgIterT>
std::string buildArgList(llvm::iterator_range<ArgIterT> Args,
                         const ASTContext &Context) {
  std::stringstream List;
  for (auto A = begin(Args); A != end(Args); A++) {
    std::string Elem = ReplacedArgText(*A, Context);
    if (!Elem.empty()) {
      // Fixed bug in the situation:
      // funciton declaration is "void fun(int a, int b, int c=0)",
      // and, "fun(a, b)" migrated "fun(a,b,"")"
      List << Elem;
      if (A + 1 != end(Args)) {
        List << ", ";
      }
    }
  }
  return List.str();
}

template <typename ArgIterT, typename TypeIterT>
std::string buildArgList(llvm::iterator_range<ArgIterT> Args,
                         llvm::iterator_range<TypeIterT> Types,
                         const ASTContext &Context) {
  std::stringstream List;
  auto B = begin(Types);
  bool IsCommaNeeded = true;
  for (auto A = begin(Args); A != end(Args); A++) {
    if (*A != nullptr && !(*B).empty()) {
      // General case, both are not empty
      std::string Elem = ReplacedArgText(*A, Context);
      if (!Elem.empty()) {
        // Fixed bug in the situation:
        // funciton declaration is "void fun(int a, int b, int c=0)",
        // and, "fun(a, b)" migrated "fun(a,b,"")"
        List << *B << "(" << Elem << ")";
      } else {
        IsCommaNeeded = false;
      }
    } else if (*A != nullptr && (*B).empty()) {
      // No type, just argument
      std::string Elem = ReplacedArgText(*A, Context);
      if (!Elem.empty()) {
        // Fixed bug in the situation:
        // funciton declaration is "void fun(int a, int b, int c=0)",
        // and, "fun(a, b)" migrated "fun(a,b,"")"
        List << Elem;
      } else {
        IsCommaNeeded = false;
      }
    } else if (*A == nullptr && !(*B).empty()) {
      // Just use "type", which is desired textual representation
      // of argument in this case.
      List << *B;
    } else {
      // Both are empty. Houston, we have a problem!
      assert(false);
    }
    if (IsCommaNeeded) {
      // Separated with comma and space ", "
      List << ", ";
    }
    B++;
  }

  std::string ret = List.str();
  // Remove the last comma and space, related with separator string,eg. ", "
  ret.pop_back();
  ret.pop_back();
  return ret;
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

std::string printTemplateArgument(const TemplateArgument &Arg,
                                  const PrintingPolicy &PP) {
  std::string Out;
  llvm::raw_string_ostream OS(Out);
  Arg.print(PP, OS);
  return OS.str();
}

ReplaceKernelCallExpr::ReplaceKernelCallExpr(
    std::shared_ptr<KernelCallExpr> Kernel, StmtStringMap *SSM)
    : TextModification(TMID::ReplaceKernelCallExpr, G3),
      KCall(Kernel->getCallExpr()), Kernel(Kernel), SSM(SSM) {}

std::pair<const Expr *, const Expr *>
ReplaceKernelCallExpr::getExecutionConfig() const {
  return {KCall->getConfig()->getArg(0), KCall->getConfig()->getArg(1)};
}

// Migrate some explicit and implicit constructions of dim3 objects when
// expressions are passed as kernel execution configuration. Returns the
// migration of that expression to cl::sycl::range<3>.
//
// If E is a variable reference, returns the name of the variable.
// Else assumes E is an implicit or explicit construction of dim3 and returns
// an explicit cl::sycl::range<3>-constructor call.
std::string ReplaceKernelCallExpr::getDim3Translation(const Expr *E,
                                                      const ASTContext &Context,
                                                      StmtStringMap *SSM) {
  if (auto Var = dyn_cast<DeclRefExpr>(E)) {
    // kernel<<<griddim, threaddim>>>()
    return Var->getNameInfo().getAsString();
  } else {
    // the dim3 migration rule should've inserted the necessary migration in
    // the StmtStringMap
    std::string NewStr = SSM->lookup(E);
    if (NewStr.empty()) {
      return getStmtSpelling(E, Context);
    } else {
      return NewStr;
    }
  }
}

ExtReplacement
ReplaceKernelCallExpr::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  auto NL = getNL(Kernel->getCallExpr()->getEndLoc(), SM);
  auto OrigIndent = getIndent(Kernel->getCallExpr()->getBeginLoc(), SM).str();
  std::stringstream Header;
  std::stringstream Header2;
  std::stringstream Header3;
  std::stringstream HeaderShareVarAccessor;
  std::stringstream HeaderShareVasAsArgs;
  std::stringstream HeaderConstantVarAccessor;
  std::stringstream HeaderConstantVasAsArgs;
  std::stringstream HeaderDeviceVarAccessor;
  std::stringstream HeaderDeviceVarAsArgs;

  std::vector<std::string> TemplateArgsArray;
  PrintingPolicy PP(Context.getLangOpts());

  Header << "{" << NL;
  auto Indent = OrigIndent + "  ";
  Header2 << Kernel->getAccessorDecl(Indent + "    ", NL);
  std::unordered_set<std::string> DuplicateFilter;
  for (auto *Arg : Kernel->getCallExpr()->arguments()) {
    if (Arg->getType()->isAnyPointerType()) {
      if (auto *DeclRef = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts())) {
        auto VarName = DeclRef->getNameInfo().getAsString();
        // for same VarName, only generate one (access, offset,buf)
        if (DuplicateFilter.find(VarName) == end(DuplicateFilter)) {
          DuplicateFilter.insert(VarName);
        } else {
          continue;
        }
        auto PointeeType = DeclRef->getDecl()->getType()->getPointeeType();
        // TODO check that no nested pointers in a structure
        assert(!PointeeType->isAnyPointerType());
        // auto VarType = PointeeType.getCanonicalType().getAsString();
        // remove getCanonicalType() for it will cause error while the type
        // is a template parameter type.
        Header << Indent << "std::pair<syclct::buffer_t, size_t> " << VarName
               << "_buf = syclct::get_buffer_and_offset(" << VarName + ");"
               << NL;
        Header << Indent << "size_t " << VarName
               << "_offset = " << VarName + "_buf.second;" << NL;
        Header2 << Indent << "    auto " << VarName << "_acc = " << VarName
                << "_buf.first."
                   "get_access<cl::sycl::access::mode::read_write>("
                << "cgh);" << NL;

        std::string VarType;
        if (auto *SubstedType =
                dyn_cast<SubstTemplateTypeParmType>(PointeeType)) {
          // Type is substituted by template initialization or specialization.
          VarType = SubstedType->getReplacedParameter()
                        ->getIdentifier()
                        ->getName()
                        .str();
        } else {
          VarType = PointeeType.getAsString();
          // adjust the VarType: if it is vector type ("struct int2/int3....)
          // changed it to syclsytle.
          auto Search = MapNames::TypeNamesMap.find(VarType);
          if (Search != MapNames::TypeNamesMap.end()) {
            VarType = Search->second;
          }
        }

        Header3 << Indent << "        " << VarType << " *" << VarName << " = ("
                << VarType << "*)(&" << VarName << "_acc[0] + " << VarName
                << "_offset);" << NL;
      } else {
        assert(false && "unknown argument expression");
      }
    }
  }

  const Expr *NDSize;
  const Expr *WGSize;
  std::tie(NDSize, WGSize) = getExecutionConfig();
  auto LocHash =
      getHashAsString(KCall->getBeginLoc().printToString(SM)).substr(0, 6);

  auto &KName = Kernel->getName();
  auto TemplateArgs = Kernel->getTemplateArguments();
  std::string KernelClassName, CallFunc;
  KernelClassName = "syclct_kernel_name<class " + KName + "_" + LocHash;
  CallFunc = KName;
  if (TemplateArgs.empty())
    KernelClassName += ">";
  else {
    CallFunc += TemplateArgs;
    KernelClassName += ", " + Kernel->getTemplateArguments(true).substr(1);
  }

  const std::string &ItemName = SyclctGlobalInfo::getItemName();

  // clang-format off
  std::stringstream Final;
  Final
  << Header.str()
  << Indent << "syclct::get_default_queue().submit(" << NL
  << Indent <<  "  [&](cl::sycl::handler &cgh) {" << NL
  << Header2.str()
  << HeaderShareVarAccessor.str()
  << HeaderConstantVarAccessor.str()
  << HeaderDeviceVarAccessor.str()
  << Indent <<  "    cgh.parallel_for<" << KernelClassName << ">(" << NL
  << Indent <<  "      cl::sycl::nd_range<3>(("
  << getDim3Translation(NDSize, Context, SSM) << " * "
  << getDim3Translation(WGSize, Context, SSM) << "), "
  << getDim3Translation(WGSize, Context, SSM)<<")," << NL
  << Indent <<  "      [=](cl::sycl::nd_item<3> " + ItemName + ") {" << NL
  << Header3.str()
  << Indent <<  "        " << CallFunc << "(" << Kernel->getArguments() << ");" << NL
  << Indent <<  "      });" <<  NL
  << Indent <<  "  })" << (Kernel->isSync() ? ".wait()" : "") << ";" <<  NL
  << OrigIndent << "}";
  // clang-format on

  recordTranslationInfo(Context, KCall->getBeginLoc());
  return ExtReplacement(SM, KCall->getBeginLoc(), 0, Final.str(), this);
}

ExtReplacement
ReplaceCallExpr::getReplacement(const ASTContext &Context) const {
  recordTranslationInfo(Context, C->getBeginLoc());
  return ExtReplacement(
      Context.getSourceManager(), C,
      buildCall(Name, llvm::iterator_range<decltype(begin(Args))>(Args),
                llvm::iterator_range<decltype(begin(Types))>(Types), Context),
      this);
}

bool ReplacementFilter::containsInterval(const IntervalSet &IS,
                                         const Interval &I) const {
  size_t Low = 0;
  size_t High = IS.size();

  while (High != Low) {
    size_t Mid = Low + (High - Low) / 2;

    if (IS[Mid].Offset == I.Offset && I.Length == 0)
      // I is designed to replace the deletion at IS[Mid].
      return false;
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

ExtReplacement InsertArgument::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  auto OrigIndent = getIndent(FD->getBeginLoc(), SM).str();

  auto FNameLoc = FD->getNameInfo().getEndLoc();
  // TODO: Investigate what happens in macro expansion
  auto tkn =
      Lexer::findNextToken(FNameLoc, Context.getSourceManager(), LangOptions())
          .getValue();
  // TODO: Investigate if its possible to not have l_paren as next token
  assert(tkn.is(tok::TokenKind::l_paren));
  // Emit new argument at the end of l_paren token
  std::string Arg = ArgName;
  // if (Lazy) {
  //  std::string KernelFunName = FD->getNameAsString();
  //  if (KernelTransAssist::hasKernelInfo(KernelFunName)) {
  //    KernelInfo &KI = KernelTransAssist::getKernelInfo(KernelFunName);
  //    Arg = KI.getKernelArgs();
  //  }
  //}

  auto OutStr = Arg;
  if (!FD->parameters().empty())
    OutStr = Arg + getFmtEndArg() + getFmtArgIndent(OrigIndent);

  recordTranslationInfo(Context, FD->getBeginLoc());
  return ExtReplacement(Context.getSourceManager(), tkn.getEndLoc(), 0, OutStr,
                        this);
}

ExtReplacement
InsertCallArgument::getReplacement(const ASTContext &Context) const {
  const SourceLocation &SLocBegin = CE->getBeginLoc();
  const SourceLocation &SLocEnd = CE->getEndLoc();
  const SourceManager &SM = Context.getSourceManager();
  const char *Start = SM.getCharacterData(SLocBegin);
  const char *End = SM.getCharacterData(SLocEnd);
  assert(End > Start);
  llvm::StringRef CallStr(Start, End - Start + 1);
  assert(CallStr.find_first_of("(") != llvm::StringRef::npos);
  size_t Offset = CallStr.find_first_of("(") + 1;
  const std::string InsertStr = (CE->getNumArgs() == 0) ? Arg : Arg + ", ";

  recordTranslationInfo(Context, CE->getBeginLoc());
  return ExtReplacement(Context.getSourceManager(),
                        SLocBegin.getLocWithOffset(Offset), 0, InsertStr, this);
}

SourceLocation InsertBeforeCtrInitList::getInsertLoc() const {
  auto Init = CDecl->init_begin();
  while (Init != CDecl->init_end()) {
    auto InitLoc = (*Init)->getSourceLocation();
    if (InitLoc.isValid()) {
      // Try to insert before ":"
      int i = 0;
      auto Data =
          SyclctGlobalInfo::getSourceManager().getCharacterData(InitLoc);
      while (Data[i] != ':')
        --i;
      return InitLoc.getLocWithOffset(i);
    }
    ++Init;
  }
  return CDecl->getBody()->getBeginLoc();
}

ExtReplacement
InsertBeforeCtrInitList::getReplacement(const ASTContext &Context) const {
  recordTranslationInfo(Context, CDecl->getBeginLoc());
  return ExtReplacement(Context.getSourceManager(), getInsertLoc(), 0, T, this);
}

bool ReplacementFilter::isDeletedReplacement(const ExtReplacement &R) const {
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

ReplacementFilter::ReplacementFilter(const std::vector<ExtReplacement> &RS)
    : ReplSet(RS) {
  for (const ExtReplacement &R : ReplSet)
    if (R.getReplacementText().empty())
      FileMap[R.getFilePath()].push_back({R.getOffset(), R.getLength()});
  for (auto &FMI : FileMap) {
    IntervalSet &IS = FMI.second;
    std::sort(IS.begin(), IS.end());
    // delete smaller intervals if they are overlapped by the preceeding one
    IntervalSet::iterator It = IS.begin();
    IntervalSet::iterator Prev = It++;
    while (It != IS.end()) {
      if (Prev->Offset + Prev->Length > It->Offset) {
        It = IS.erase(It);
      } else {
        Prev = It;
        It++;
      }
    }
  }
}

ExtReplacement
InsertBeforeStmt::getReplacement(const ASTContext &Context) const {
  SourceLocation Begin = S->getSourceRange().getBegin();
  recordTranslationInfo(Context, S->getBeginLoc());
  return ExtReplacement(Context.getSourceManager(),
                        CharSourceRange(SourceRange(Begin, Begin), false), T,
                        this);
}

ExtReplacement RemoveArg::getReplacement(const ASTContext &Context) const {
  SourceRange SR = CE->getArg(N)->getSourceRange();
  SourceLocation Begin = SR.getBegin();
  SourceLocation End;
  bool IsLast = (N == (CE->getNumArgs() - 1));
  if (IsLast) {
    End = SR.getEnd();
  } else {
    End = CE->getArg(N + 1)->getSourceRange().getBegin().getLocWithOffset(0);
  }
  recordTranslationInfo(Context, CE->getBeginLoc());
  return ExtReplacement(Context.getSourceManager(),
                        CharSourceRange(SourceRange(Begin, End), false), "",
                        this);
}

ExtReplacement
InsertClassName::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  auto BeginLoc = CD->getBeginLoc();
  auto DataBegin = SM.getCharacterData(BeginLoc);

  unsigned i = 0;
  auto Data = DataBegin[i];
  while ((Data != ':') && (Data != '{'))
    Data = DataBegin[++i];

  Data = DataBegin[--i];
  while ((Data == ' ') || (Data == '\t') || (Data == '\n'))
    Data = DataBegin[--i];

  recordTranslationInfo(Context, CD->getBeginLoc());
  return ExtReplacement(
      SM, BeginLoc.getLocWithOffset(i + 1), 0,
      " syclct_type_" +
          getHashAsString(BeginLoc.printToString(SM)).substr(0, 6),
      this);
}

ExtReplacement ReplaceText::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  recordTranslationInfo(Context, BeginLoc);
  return ExtReplacement(SM, BeginLoc, Len, T, this);
}

static const std::unordered_map<int, std::string> TMNameMap = {
#define TRANSFORMATION(TYPE) {static_cast<int>(TMID::TYPE), #TYPE},
#include "Transformations.inc"
#undef TRANSFORMATION
};

const std::string TextModification::getName() const {
  return TMNameMap.at(static_cast<int>(getID()));
}

constexpr char TransformStr[] = " => ";
static void printHeader(llvm::raw_ostream &OS, const TMID &ID,
                        const char *ParentRuleID) {
  OS << "[";
  if (ParentRuleID) {
    OS << ASTTraversalMetaInfo::getNameTable()[ParentRuleID] << ":";
  }
  OS << TMNameMap.at(static_cast<int>(ID));
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
  if (D) {
    printLocation(OS, D->getBeginLoc(), Context, PrintDetail);
    D->print(OS, PrintingPolicy(Context.getLangOpts()));
  } else {
    printLocation(OS, FD->getBeginLoc(), Context, PrintDetail);
    FD->print(OS, PrintingPolicy(Context.getLangOpts()));
  }
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

void InsertNameSpaceInVarDecl::print(llvm::raw_ostream &OS, ASTContext &Context,
                                     const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, D->getBeginLoc(), Context, PrintDetail);
  printInsertion(OS, T);
}

void InsertNameSpaceInCastExpr::print(llvm::raw_ostream &OS,
                                      ASTContext &Context,
                                      const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, D->getBeginLoc(), Context, PrintDetail);
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

void ReplaceKernelCallExpr::print(llvm::raw_ostream &OS, ASTContext &Context,
                                  const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, KCall->getBeginLoc(), Context, PrintDetail);
  KCall->printPretty(OS, nullptr, PrintingPolicy(Context.getLangOpts()));
  // TODO: print simple and meaningful informations
  OS << TransformStr << "[debug message unimplemented]\n";
}

void ReplaceCallExpr::print(llvm::raw_ostream &OS, ASTContext &Context,
                            const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, C->getBeginLoc(), Context, PrintDetail);
  C->printPretty(OS, nullptr, PrintingPolicy(Context.getLangOpts()));
  // TODO: print simple and meaningful informations
  OS << TransformStr << "[debug message unimplemented]\n";
}

void InsertArgument::print(llvm::raw_ostream &OS, ASTContext &Context,
                           const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, FD->getBeginLoc(), Context, PrintDetail);
  FD->print(OS, PrintingPolicy(Context.getLangOpts()));
  printInsertion(OS, ArgName);
}

void InsertCallArgument::print(llvm::raw_ostream &OS, ASTContext &Context,
                               const bool PrintDetail) const {
  printHeader(OS, getID(), PrintDetail ? getParentRuleID() : nullptr);
  printLocation(OS, CE->getBeginLoc(), Context, PrintDetail);
  CE->printPretty(OS, nullptr, PrintingPolicy(Context.getLangOpts()));
  printInsertion(OS, Arg);
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
  printLocation(OS, S->getBeginLoc(), Context, PrintDetail);
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
