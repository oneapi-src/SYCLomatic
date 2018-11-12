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
#include "AnalysisInfo.h"
#include "Utility.h"

#include "Utility.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

#include <sstream>

using namespace clang;
using namespace clang::syclct;
using namespace clang::tooling;

ExtReplacement ReplaceStmt::getReplacement(const ASTContext &Context) const {
  return ExtReplacement(Context.getSourceManager(), TheStmt, ReplacementString);
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

  return ExtReplacement(
      SM, CharSourceRange::getCharRange(ExpB, ExpB.getLocWithOffset(Len)), "");
}

ExtReplacement
ReplaceTypeInVarDecl::getReplacement(const ASTContext &Context) const {
  TypeLoc TL = D->getTypeSourceInfo()->getTypeLoc();
  return ExtReplacement(Context.getSourceManager(), &TL, T);
}

ExtReplacement RemoveVarDecl::getReplacement(const ASTContext &Context) const {
  SourceLocation slStart = D->getSourceRange().getBegin();
  SourceLocation slEnd = D->getSourceRange().getEnd();
  if (slStart.isFileID() && slEnd.isFileID()) {
    return ExtReplacement(Context.getSourceManager(), D, T);
  } else {
    auto &SM = Context.getSourceManager();
    size_t repLength;
    repLength = SM.getCharacterData(slEnd) -
                SM.getCharacterData(SM.getExpansionLoc(slStart)) + 1;

    // try to del  "    ;" in var declare
    auto DataAfter = SM.getCharacterData(slEnd.getLocWithOffset(1));
    unsigned i = 0;
    auto Data = DataAfter[i];
    while ((Data == ' ') || (Data == '\t'))
      Data = DataAfter[++i];
    if (Data == ';')
      Data = DataAfter[++i];
    repLength += i;

    return ExtReplacement(Context.getSourceManager(),
                          SM.getExpansionLoc(slStart), repLength, T);
  }
}

ExtReplacement
ReplaceReturnType::getReplacement(const ASTContext &Context) const {
  SourceRange SR = FD->getReturnTypeSourceRange();
  return ExtReplacement(Context.getSourceManager(), CharSourceRange(SR, true),
                        T);
}

ExtReplacement ReplaceToken::getReplacement(const ASTContext &Context) const {
  // Need to deal with the fact, that the type name might be a macro.
  return ExtReplacement(Context.getSourceManager(),
                        // false means [Begin, End)
                        // true means [Begin, End]
                        CharSourceRange(SourceRange(Begin, Begin), true), T);
}

ExtReplacement InsertText::getReplacement(const ASTContext &Context) const {
  // Need to deal with the fact, that the type name might be a macro.
  return ExtReplacement(Context.getSourceManager(),
                        // false means [Begin, End)
                        // true means [Begin, End]
                        CharSourceRange(SourceRange(Begin, Begin), false), T);
}

ExtReplacement
InsertNameSpaceInVarDecl::getReplacement(const ASTContext &Context) const {
  TypeLoc TL = D->getTypeSourceInfo()->getTypeLoc();
  ExtReplacement R(
      Context.getSourceManager(),
      CharSourceRange(SourceRange(TL.getBeginLoc(), TL.getBeginLoc()), false),
      T);
  R.setInsertPosition(InsertPositionRight);
  return R;
}
ExtReplacement
InsertNameSpaceInCastExpr::getReplacement(const ASTContext &Context) const {
  return ExtReplacement(
      Context.getSourceManager(),
      CharSourceRange(SourceRange(D->getLParenLoc().getLocWithOffset(1),
                                  D->getLParenLoc().getLocWithOffset(1)),
                      false),
      T);
}

ExtReplacement ReplaceCCast::getReplacement(const ASTContext &Context) const {
  auto Begin = Cast->getLParenLoc();
  auto End = Cast->getRParenLoc();
  return ExtReplacement(Context.getSourceManager(),
                        CharSourceRange(SourceRange(Begin, End), true),
                        TypeName);
}

ExtReplacement
RenameFieldInMemberExpr::getReplacement(const ASTContext &Context) const {
  SourceLocation SL = ME->getEndLoc();
  SourceLocation Begin = SL;
  if (PositionOfDot != 0) {
    // Cover dot position when translate dim3.x/y/z to
    // cl::sycl::range<3>[0]/[1]/[2].
    Begin = ME->getBeginLoc();
    Begin = Begin.getLocWithOffset(PositionOfDot);
  }

  return ExtReplacement(Context.getSourceManager(),
                        CharSourceRange(SourceRange(Begin, SL), true), T);
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
  return ExtReplacement(SM, CharSourceRange(SourceRange(End, End), false), T);
}

ExtReplacement ReplaceInclude::getReplacement(const ASTContext &Context) const {
  return ExtReplacement(Context.getSourceManager(), Range, T);
}

ExtReplacement InsertComment::getReplacement(const ASTContext &Context) const {
  auto NL = getNL(SL, Context.getSourceManager());
  return ExtReplacement(Context.getSourceManager(), SL, 0,
                        (llvm::Twine("/*") + NL + Text + NL + "*/" + NL).str());
}

template <typename ArgIterT>
std::string buildArgList(llvm::iterator_range<ArgIterT> Args,
                         const ASTContext &Context) {
  std::stringstream List;
  for (auto A = begin(Args); A != end(Args); A++) {
    std::string Elem = getStmtSpelling(*A, Context);
    if (!Elem.empty()) {
      // Fixed bug in the situation:
      // funciton declaration is "void fun(int a, int b, int c=0)",
      // and, "fun(a, b)" translated "fun(a,b,"")"
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
      std::string Elem = getStmtSpelling(*A, Context);
      if (!Elem.empty()) {
        // Fixed bug in the situation:
        // funciton declaration is "void fun(int a, int b, int c=0)",
        // and, "fun(a, b)" translated "fun(a,b,"")"
        List << *B << "(" << Elem << ")";
      } else {
        IsCommaNeeded = false;
      }
    } else if (*A != nullptr && (*B).empty()) {
      // No type, just argument
      std::string Elem = getStmtSpelling(*A, Context);
      if (!Elem.empty()) {
        // Fixed bug in the situation:
        // funciton declaration is "void fun(int a, int b, int c=0)",
        // and, "fun(a, b)" translated "fun(a,b,"")"
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

std::pair<const Expr *, const Expr *>
ReplaceKernelCallExpr::getExecutionConfig() const {
  auto NDSizeTopCtor =
      dyn_cast<CXXConstructExpr>(KCall->getConfig()->getArg(0));
  auto WGSizeTopCtor =
      dyn_cast<CXXConstructExpr>(KCall->getConfig()->getArg(1));

  auto NDTemp = dyn_cast<MaterializeTemporaryExpr>(NDSizeTopCtor->getArg(0));
  auto WGTemp = dyn_cast<MaterializeTemporaryExpr>(WGSizeTopCtor->getArg(0));

  // Return nested constructor or reference to a variable.
  return {NDTemp ? NDTemp->GetTemporaryExpr()->IgnoreCasts()
                 : NDSizeTopCtor->getArg(0)->IgnoreCasts(),
          WGTemp ? WGTemp->GetTemporaryExpr()->IgnoreCasts()
                 : WGSizeTopCtor->getArg(0)->IgnoreCasts()};
}

// Translates some explicit and implicit constructions of dim3 objects when
// expressions are passed as kernel execution configuration. Returns the
// translation of that expression to cl::sycl::range<3>.
//
// If E is a variable reference, returns the name of the variable.
// Else assumes E is an implicit or explicit construction of dim3 and returns
// an explicit cl::sycl::range<3>-constructor call.
std::string ReplaceKernelCallExpr::getDim3Translation(const Expr *E,
                                                      const ASTContext &Context,
                                                      unsigned int EffectDims) {
  if (auto Var = dyn_cast<DeclRefExpr>(E)) {
    // kernel<<<griddim, threaddim>>>()
    return Var->getNameInfo().getAsString();
  } else {
    auto Ctor = dyn_cast<CXXConstructExpr>(E);
    // kernel<<<dim3(1, 2), dim3(1, 2, 3)>>>() -- temporary objects
    // kernel<<<1, dim3(1)>>>() -- constructor expressions
    unsigned DimsN = getDimsNum(E);
    std::stringstream Stream;

    if (EffectDims == 3) {
      Stream << "cl::sycl::range<3>(";
      if (DimsN == EffectDims) {
        Stream << getStmtSpelling(Ctor->getArg(0), Context);
        Stream << ", " << getStmtSpelling(Ctor->getArg(1), Context);
        Stream << ", " << getStmtSpelling(Ctor->getArg(2), Context) << ")";
      } else if (DimsN == 2) {
        Stream << getStmtSpelling(Ctor->getArg(0), Context);
        Stream << ", " << getStmtSpelling(Ctor->getArg(1), Context);
        Stream << ", 1)";
      } else if (DimsN == 1) {
        Stream << getStmtSpelling(Ctor->getArg(0), Context);
        Stream << ", 1";
        Stream << ", 1)";
      }
    } else if (EffectDims == 2) {

      if (DimsN == EffectDims) {
        Stream << "cl::sycl::range<2>(";
        Stream << getStmtSpelling(Ctor->getArg(0), Context);
        Stream << ", " << getStmtSpelling(Ctor->getArg(1), Context) << ")";
      } else if (DimsN == 1) {
        Stream << "cl::sycl::range<2>(";
        Stream << getStmtSpelling(Ctor->getArg(0), Context);
        Stream << ", 1)";
      }
    } else if (EffectDims == 1) {
      Stream << "cl::sycl::range<1>(";
      Stream << getStmtSpelling(Ctor->getArg(0), Context);
      Stream << ")";
    }

    return Stream.str();
  }
}
unsigned int ReplaceKernelCallExpr::getDimsNum(const Expr *E) {
  unsigned EffectDims = 3;
  if (auto Ctor = dyn_cast<CXXConstructExpr>(E)) {
    assert(Ctor && "Only dim3 constructors (explicit or implicit from literal "
                   "int, variables (int and dim3)) supported.");
    if (isa<CXXDefaultArgExpr>(Ctor->getArg(1))) {
      EffectDims = 1;
    } else if (isa<CXXDefaultArgExpr>(Ctor->getArg(2))) {
      EffectDims = 2;
    }
  }
  return EffectDims;
}

ExtReplacement
ReplaceKernelCallExpr::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  auto NL = getNL(KCall->getEndLoc(), SM);
  auto OrigIndent = getIndent(KCall->getBeginLoc(), SM).str();
  std::stringstream Header;
  std::stringstream Header2;
  std::stringstream Header3;
  std::stringstream HeaderShareVarAccessor;
  std::stringstream HeaderShareVasAsArgs;
  std::stringstream HeaderConstantVarAccessor;
  std::stringstream HeaderConstantVasAsArgs;
  std::stringstream HeaderDeviceVarAccessor;
  std::stringstream HeaderDeviceVarAsArgs;

  Header << "{" << NL;
  auto Indent = OrigIndent + "  ";
  // check if sharevariable info exist for this kernel.
  // [todo] template case not support yet.
  if (KCall && KCall->getCalleeDecl()) {
    std::string KernelFunName =
        KCall->getCalleeDecl()->getAsFunction()->getNameAsString();
    if (KernelTransAssist::hasKernelInfo(KernelFunName)) {
      KernelInfo KI = KernelTransAssist::getKernelInfo(KernelFunName);
      if (KI.hasSMVDefined()) {
        HeaderShareVarAccessor << KI.declareLocalAcc(NL, Indent + "    ");
        HeaderShareVasAsArgs << KI.passSMVAsArgs() << ", ";
      }
      if (KI.hasCMVDefined()) {
        HeaderConstantVarAccessor << KI.declareConstantAcc(NL, Indent + "    ");
        HeaderConstantVasAsArgs << KI.passCMVAsArgs() << ", ";
      }
      if (KI.hasDMVDefined()) {
        HeaderDeviceVarAccessor << KI.declareDeviceAcc(NL, Indent + "    ");
        HeaderDeviceVarAsArgs << KI.passDMVAsArgs() << ", ";
      }
    }
  }
  for (auto *Arg : KCall->arguments()) {
    if (Arg->getType()->isAnyPointerType()) {
      if (auto *DeclRef = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts())) {
        auto VarName = DeclRef->getNameInfo().getAsString();
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

  std::string KName, TemplateArgs, KernelClassName, CallFunc;
  if (auto KCallee = dyn_cast<UnresolvedLookupExpr>(KCall->getCallee())) {
    KName = KCallee->getName().getAsString();
    /// Template kernel called from template function.
    /// template <class T> void run() { testKernel<T><<<64, 256>>>(); }
    // TODO: Implicit template arguments are ignored, it need to be fix in
    // future.
    TemplateArgs =
        getTemplateArgs(KCallee->getLAngleLoc(), KCallee->getRAngleLoc(), SM);
  } else {
    auto KCallDecl = KCall->getCalleeDecl()->getAsFunction();
    KName = KCallDecl->getName();
    if (KCallDecl->isFunctionTemplateSpecialization()) {
      /// Template kernel called from function.
      /// void run() { testKernel<T><<<64, 256>>>(); }
      auto KCallee = KCall->getCallee()->IgnoreParenImpCasts();
      TemplateArgs =
          getTemplateArgs(KCallee->getBeginLoc(), KCallee->getEndLoc(), SM);
    }
  }

  KernelClassName = "SyclKernelName<class " + KName + "_" + LocHash;
  CallFunc = KName;
  if (!TemplateArgs.empty()) {
    KernelClassName += ", " + TemplateArgs;
    CallFunc += "<" + TemplateArgs + ">";
  }
  KernelClassName += ">";

  // clang-format off
  std::stringstream Final;
  unsigned int DimsN = getDimsNum(NDSize);
  unsigned int DimsW = getDimsNum(WGSize);
  unsigned int EffectDims = DimsN>DimsW ? DimsN:DimsW;
  Final
  << Header.str()
  << Indent << "syclct::get_default_queue().submit(" << NL
  << Indent <<  "  [&](cl::sycl::handler &cgh) {" << NL
  << Header2.str()
  << HeaderShareVarAccessor.str()
  << HeaderConstantVarAccessor.str()
  << HeaderDeviceVarAccessor.str()
  << Indent <<  "    cgh.parallel_for<" << KernelClassName << ">(" << NL
  << Indent <<  "      cl::sycl::nd_range<" << EffectDims << ">(("
  << getDim3Translation(NDSize, Context, EffectDims) << " * "
  << getDim3Translation(WGSize, Context, EffectDims) << "), "
  << getDim3Translation(WGSize, Context, EffectDims)<<")," << NL
  << Indent <<  "      [=](cl::sycl::nd_item<"<< EffectDims << "> it) {" << NL
  << Header3.str()
  << Indent <<  "        " << CallFunc << "(it, "<< HeaderShareVasAsArgs.str()
                                                 << HeaderConstantVasAsArgs.str()
                                                 << HeaderDeviceVarAsArgs.str()
                                                 << buildArgList(KCall->arguments(), Context)
                                    << ");" <<  NL
  << Indent <<  "      });" <<  NL
  << Indent <<  "  });" <<  NL
  << OrigIndent << "}";
  // clang-format on

  return ExtReplacement(SM, KCall->getBeginLoc(), 0, Final.str());
}

ExtReplacement
ReplaceCallExpr::getReplacement(const ASTContext &Context) const {
  return ExtReplacement(
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
  auto FNameLoc = FD->getNameInfo().getEndLoc();
  // TODO: Investigate what happens in macro expansion
  auto tkn =
      Lexer::findNextToken(FNameLoc, Context.getSourceManager(), LangOptions())
          .getValue();
  // TODO: Investigate if its possible to not have l_paren as next token
  assert(tkn.is(tok::TokenKind::l_paren));
  // Emit new argument at the end of l_paren token
  std::string Arg = ArgName;
  if (Lazy) {
    std::string KernelFunName = FD->getNameAsString();
    if (KernelTransAssist::hasKernelInfo(KernelFunName)) {
      KernelInfo &KI = KernelTransAssist::getKernelInfo(KernelFunName);
      Arg = KI.getKernelArgs();
    }
  }

  auto OutStr = Arg;
  if (!FD->parameters().empty())
    OutStr = Arg + ", ";
  return ExtReplacement(Context.getSourceManager(), tkn.getEndLoc(), 0, OutStr);
}

ExtReplacement
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
    return ExtReplacement(Context.getSourceManager(), SLoc, 0, T);
  } else {
    SourceLocation Begin = CDecl->getBody()->getSourceRange().getBegin();
    return ExtReplacement(Context.getSourceManager(), Begin, 0, T);
  }
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
  // TODO: Smaller Intervals should be discarded if they are completely
  // covered by a larger Interval, so that no intervals overlap in the set.
  for (const ExtReplacement &R : ReplSet)
    if (R.getReplacementText().empty())
      FileMap[R.getFilePath()].push_back({R.getOffset(), R.getLength()});
  for (auto &FMI : FileMap)
    std::sort(FMI.second.begin(), FMI.second.end());
}

ExtReplacement
InsertBeforeStmt::getReplacement(const ASTContext &Context) const {
  SourceLocation Begin = S->getSourceRange().getBegin();
  return ExtReplacement(Context.getSourceManager(),
                        CharSourceRange(SourceRange(Begin, Begin), false), T);
}

ExtReplacement RemoveArg::getReplacement(const ASTContext &Context) const {
  SourceRange SR = CE->getArg(N)->getSourceRange();
  SourceLocation Begin = SR.getBegin();
  SourceLocation End;
  bool IsLast = (N == (CE->getNumArgs() - 1));
  if (IsLast) {
    End = SR.getEnd();
  } else {
    End = CE->getArg(N + 1)->getSourceRange().getBegin().getLocWithOffset(-1);
  }
  return ExtReplacement(Context.getSourceManager(),
                        CharSourceRange(SourceRange(Begin, End), true), "");
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

  return ExtReplacement(
      SM, BeginLoc.getLocWithOffset(i + 1), 0,
      " SYCL_TYPE_" + getHashAsString(BeginLoc.printToString(SM)).substr(0, 6));
}
