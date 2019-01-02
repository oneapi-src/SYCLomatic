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

ExtReplacement ReplaceStmt::getReplacement(const ASTContext &Context) const {
  return ExtReplacement(Context.getSourceManager(), TheStmt, ReplacementString,
                        this);
}

ExtReplacement
ReplaceCalleeName::getReplacement(const ASTContext &Context) const {
  return ExtReplacement(Context.getSourceManager(), C->getBeginLoc(),
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

  return ExtReplacement(
      SM, CharSourceRange::getCharRange(ExpB, ExpB.getLocWithOffset(Len)), "",
      this);
}

ExtReplacement
ReplaceTypeInDecl::getReplacement(const ASTContext &Context) const {
  return ExtReplacement(Context.getSourceManager(), &TL, T, this);
}

ExtReplacement ReplaceVarDecl::getReplacement(const ASTContext &Context) const {
  auto &SM = Context.getSourceManager();
  SourceLocation slStart = SM.getExpansionLoc(D->getSourceRange().getBegin());
  SourceLocation slEnd = SM.getExpansionLoc(D->getSourceRange().getEnd());
  size_t repLength;
  repLength = SM.getCharacterData(slEnd) - SM.getCharacterData(slStart) + 1;
  if (!D->getType()->isArrayType())
    repLength += D->getName().size();
  // try to del  "    ;" in var declare
  auto DataAfter = SM.getCharacterData(slStart.getLocWithOffset(repLength));
  unsigned i = 0;
  auto Data = DataAfter[i];
  while ((Data == ' ') || (Data == '\t'))
    Data = DataAfter[++i];
  if (Data == ';')
    Data = DataAfter[++i];
  repLength += i;

  return ExtReplacement(Context.getSourceManager(), SM.getExpansionLoc(slStart),
                        repLength, T, this);
}

ExtReplacement
ReplaceReturnType::getReplacement(const ASTContext &Context) const {
  SourceRange SR = FD->getReturnTypeSourceRange();
  return ExtReplacement(Context.getSourceManager(), CharSourceRange(SR, true),
                        T, this);
}

ExtReplacement ReplaceToken::getReplacement(const ASTContext &Context) const {
  // Need to deal with the fact, that the type name might be a macro.
  return ExtReplacement(Context.getSourceManager(),
                        // false means [Begin, End)
                        // true means [Begin, End]
                        CharSourceRange(SourceRange(Begin, Begin), true), T,
                        this);
}

ExtReplacement InsertText::getReplacement(const ASTContext &Context) const {
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
  return R;
}

ExtReplacement
InsertNameSpaceInCastExpr::getReplacement(const ASTContext &Context) const {
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
  return ExtReplacement(Context.getSourceManager(),
                        CharSourceRange(SourceRange(Begin, End), true),
                        TypeName, this);
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
  return ExtReplacement(SM, CharSourceRange(SourceRange(End, End), false), T,
                        this);
}

ExtReplacement ReplaceInclude::getReplacement(const ASTContext &Context) const {
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
        Params += getStmtSpellingWithTransforms(Arg, Context, SSM);
        //        Params += getStmtSpelling(Arg, Context);
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
    StmtStringPair SSP = {Ctor, S};
    SSM->insert(SSP);
    return S;
  }
}

ExtReplacement
ReplaceDim3Ctor::getReplacement(const ASTContext &Context) const {
  return ExtReplacement(Context.getSourceManager(), CSR.getBegin(), 0,
                        getReplaceString(Context), this);
}

ExtReplacement InsertComment::getReplacement(const ASTContext &Context) const {
  auto NL = getNL(SL, Context.getSourceManager());
  return ExtReplacement(Context.getSourceManager(), SL, 0,
                        (llvm::Twine("/*") + NL + Text + NL + "*/" + NL).str(),
                        this);
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

// Translates some explicit and implicit constructions of dim3 objects when
// expressions are passed as kernel execution configuration. Returns the
// translation of that expression to cl::sycl::range<3>.
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
    // the dim3 translation rule should've inserted the necessary translation in
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
  for (auto *Arg : Kernel->getCallExpr()->arguments()) {
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

  auto &KName = Kernel->getName();
  auto TemplateArgs = Kernel->getTemplateArguments();
  std::string KernelClassName, CallFunc;
  KernelClassName = "syclct_kernel_name<class " + KName + "_" + LocHash;
  CallFunc = KName;
  if (TemplateArgs.empty())
    KernelClassName += ">";
  else {
    CallFunc += TemplateArgs;
    KernelClassName += ", " + TemplateArgs.substr(1);
  }

  const std::string &ItemName = SyclctGlobalInfo::getItemName();
  std::stringstream KernelArgs;

  KernelArgs << buildArgList(KCall->arguments(), Context)
             << Kernel->getArguments();

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
  << Indent <<  "        " << CallFunc << "(" << KernelArgs.str() << ");" << NL
  << Indent <<  "      });" <<  NL
  << Indent <<  "  });" <<  NL
  << OrigIndent << "}";
  // clang-format on

  return ExtReplacement(SM, KCall->getBeginLoc(), 0, Final.str(), this);
}

ExtReplacement
ReplaceCallExpr::getReplacement(const ASTContext &Context) const {
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
  return ExtReplacement(Context.getSourceManager(),
                        SLocBegin.getLocWithOffset(Offset), 0, InsertStr, this);
}

ExtReplacement
InsertBeforeCtrInitList::getReplacement(const ASTContext &Context) const {
  const SourceManager &SM = Context.getSourceManager();
  if (CDecl->init_begin() != CDecl->init_end()) {
    // Initialization list exists, insert before ":"
    // Eg: A(int b, int c) : b(b), c(c) {}
    //                    ^
    //                Insert here

    // Compiler generated initializer has no valid poisition in source code,
    // we care only about the initializer in the source code
    //
    // Eg: class B : public A {};
    //     B(int c, int d) : c(c), d(d) {}
    //                      ^
    //       Compiler genereated a initalizer for base class A here, filter it
    //       out
    auto SLoc = CDecl->getBeginLoc();
    auto InitValidIt = CDecl->init_begin();
    while (!(*InitValidIt)->getSourceLocation().isValid()) {
      ++InitValidIt;
    }

    assert(InitValidIt != CDecl->init_end() &&
           (*InitValidIt)->getSourceLocation().isValid());

    // The location of first valid initializer in source code
    auto SLocEnd = (*InitValidIt)->getSourceLocation();

    SourceLocation SLocInsert;
    if (SLoc == SLocEnd) {
      // No initialization list in source code, but compiler generated
      // initializers for base classes
      // Eg: class C : public B, public A {};
      //     C() {}
      //        ^
      //     Compiler generate B(), A() here
      SLocInsert = CDecl->getBody()->getBeginLoc();
    } else {
      // Initialization list in source code
      // Eg: A(int b, int c) : b(b), c(c) {}
      //     ^                ^
      //   Start             End
      const char *Start = SM.getCharacterData(SLoc);
      const char *End = SM.getCharacterData(SLocEnd);
      assert(End > Start);
      // Try to insert before ":"
      llvm::StringRef Data(Start, End - Start + 1);
      size_t Offset = Data.find_last_of(":");
      if (Offset == llvm::StringRef::npos) {
        Offset = 0;
      }
      SLocInsert = SLoc.getLocWithOffset(Offset);
    }
    return ExtReplacement(Context.getSourceManager(), SLocInsert, 0, T, this);
  } else {
    // No initialization list, just insert before function body
    SourceLocation Begin = CDecl->getBody()->getSourceRange().getBegin();
    return ExtReplacement(SM, Begin, 0, T, this);
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
    End = CE->getArg(N + 1)->getSourceRange().getBegin().getLocWithOffset(-1);
  }
  return ExtReplacement(Context.getSourceManager(),
                        CharSourceRange(SourceRange(Begin, End), true), "",
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

  return ExtReplacement(
      SM, BeginLoc.getLocWithOffset(i + 1), 0,
      " syclct_type_" +
          getHashAsString(BeginLoc.printToString(SM)).substr(0, 6),
      this);
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
  printLocation(OS, D->getBeginLoc(), Context, PrintDetail);
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
