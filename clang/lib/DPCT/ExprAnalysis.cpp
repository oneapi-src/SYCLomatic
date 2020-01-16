//===--- ExprAnalysis.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "ExprAnalysis.h"

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "CallExprRewriter.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprOpenMP.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtGraphTraits.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtOpenMP.h"

namespace clang {
namespace dpct {

#define ANALYZE_EXPR(EXPR)                                                     \
  case Stmt::EXPR##Class:                                                      \
    return analyzeExpr(static_cast<const EXPR *>(Expression));

std::map<const Expr *, std::string> ArgumentAnalysis::DefaultArgMap;

void TemplateDependentReplacement::replace(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  SourceStr.replace(Offset, Length, TemplateList[TemplateIndex].getAsString());
}

TemplateDependentStringInfo::TemplateDependentStringInfo(
    const std::string &SrcStr,
    const std::map<size_t, std::shared_ptr<TemplateDependentReplacement>>
        &InTDRs)
    : SourceStr(SrcStr) {
  for (auto TDR : InTDRs)
    TDRs.emplace_back(TDR.second->alterSource(SourceStr));
}

std::string TemplateDependentStringInfo::getReplacedString(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  std::string SrcStr(SourceStr);
  for (auto Itr = TDRs.rbegin(); Itr != TDRs.rend(); ++Itr)
    (*Itr)->replace(TemplateList);
  std::swap(SrcStr, SourceStr);
  return SrcStr;
}

SourceLocation ExprAnalysis::getExprLocation(SourceLocation Loc) {
  if (SM.isMacroArgExpansion(Loc))
    return getExprLocation(SM.getImmediateSpellingLoc(Loc));
  else
    return SM.getExpansionLoc(Loc);
}

std::pair<size_t, size_t> ExprAnalysis::getOffsetAndLength(SourceLocation Loc) {
  if (Loc.isInvalid())
    return std::pair<size_t, size_t>(0, SrcLength);
  Loc = getExprLocation(Loc);
  return std::pair<size_t, size_t>(
      getOffset(Loc),
      Lexer::MeasureTokenLength(Loc, SM, Context.getLangOpts()));
}

std::pair<size_t, size_t>
ExprAnalysis::getOffsetAndLength(SourceLocation BeginLoc,
                                 SourceLocation EndLoc) {
  if (EndLoc.isValid()) {
    auto Begin = getOffset(getExprLocation(BeginLoc));
    auto End = getOffsetAndLength(EndLoc);
    return std::pair<size_t, size_t>(Begin, End.first - Begin + End.second);
  }
  return getOffsetAndLength(BeginLoc);
}

// Check if an Expr is the outer most function-like macro
// E.g. MACRO_A(MACRO_B(x,y),z)
// Where MACRO_A is outer most and MACRO_B, x, y, z are not.
bool ExprAnalysis::isOuterMostMacro(const Expr *E) {
  std::string ExpandedExpr, ExpandedParent;
  // Save the preprocessing result of E in ExpandedExpr
  llvm::raw_string_ostream StreamE(ExpandedExpr);
  E->printPretty(StreamE, nullptr, PrintingPolicy(Context.getLangOpts()));
  StreamE.flush();
  const Stmt* P = E;

  // Find a parent stmt whose preprocessing result is different from ExpandedExpr
  // Since some parent is not writable.(is not shown in the preprocessing result),
  // a while loop is required to find the first writable ancestor.
  do {
    ExpandedParent = "";
    P = getParentStmt(P);
    if (!P)
      return true;
    llvm::raw_string_ostream StreamP(ExpandedParent);
    P->printPretty(StreamP, nullptr, PrintingPolicy(Context.getLangOpts()));
    StreamP.flush();
  } while (!ExpandedParent.compare(ExpandedExpr));

  // Since SM.getExpansionLoc() will always return the range of the outer-most
  // macro. If the expanded location of the parent stmt and E are the same, E is
  // inside a function-like macro.
  // E.g. MACRO_A(MACRO_B(x,y),z) While E is the PP
  // result of MACRO_B and P will be the PP result of MACRO_A,
  // SM.getExpansionLoc(E) is at the begining of MACRO_A, same as
  // SM.getExpansionLoc(P), in the source code. E is not outer-most.
  if (P->getBeginLoc().isValid() && P->getBeginLoc().isMacroID()) {
    if (SM.getCharacterData(SM.getExpansionLoc(P->getBeginLoc())) ==
        SM.getCharacterData(SM.getExpansionLoc(E->getBeginLoc()))) {
      return false;
    }
  }
  return true;
}

std::pair<size_t, size_t> ExprAnalysis::getOffsetAndLength(const Expr *E) {
  SourceLocation BeginLoc, EndLoc;
  if (E->getBeginLoc().isMacroID() && !isOuterMostMacro(E)) {
    // If E is not OuterMostMacro, use the spelling location
    BeginLoc = SM.getExpansionLoc(SM.getImmediateSpellingLoc(E->getBeginLoc()));
    EndLoc = SM.getExpansionLoc(SM.getImmediateSpellingLoc(E->getEndLoc()));
  } else {
    // If E is the OuterMostMacro, use the expansion location
    BeginLoc = SM.getExpansionRange(E->getBeginLoc()).getBegin();
    EndLoc = SM.getExpansionRange(E->getEndLoc()).getEnd();
  }

  // Calculate offset and length from SourceLocation
  auto End = getOffset(EndLoc);
  auto LastTokenLength =
      Lexer::MeasureTokenLength(EndLoc, SM, Context.getLangOpts());

  auto DecompLoc = SM.getDecomposedLoc(BeginLoc);
  FileId = DecompLoc.first;
  // The offset of Expr used in ExprAnalysis is related to SrcBegin not
  // FileBegin
  auto Begin = DecompLoc.second - SrcBegin;
  return std::pair<size_t, size_t>(Begin, End - Begin + LastTokenLength);
}

void ExprAnalysis::initExpression(const Expr *Expression) {
  E = Expression;
  SrcBegin = 0;
  if (E && E->getBeginLoc().isValid()) {
    std::tie(SrcBegin, SrcLength) =
        getOffsetAndLength(E);
    ReplSet.init(std::string(
        SM.getBufferData(FileId).substr(SrcBegin, SrcLength)));
  } else {
    SrcLength = 0;
    ReplSet.init("");
  }
}

void StringReplacements::replaceString() {
  SourceStr.reserve(SourceStr.length() + ShiftLength);
  auto Itr = ReplMap.rbegin();
  while (Itr != ReplMap.rend()) {
    Itr->second->replaceString();
    ++Itr;
  }
  ReplMap.clear();
}

ExprAnalysis::ExprAnalysis(const Expr *Expression)
    : Context(DpctGlobalInfo::getContext()),
      SM(DpctGlobalInfo::getSourceManager()) {
  initExpression(Expression);
}

void ExprAnalysis::dispatch(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
#define STMT(CLASS, PARENT) ANALYZE_EXPR(CLASS)
#define STMT_RANGE(BASE, FIRST, LAST)
#define LAST_STMT_RANGE(BASE, FIRST, LAST)
#define ABSTRACT_STMT(STMT)
#include "clang/AST/StmtNodes.inc"
  default:
    return;
  }
}

void ExprAnalysis::analyzeExpr(const DeclRefExpr *DRE) {
  std::string CTSName;
  auto Qualifier = DRE->getQualifier();
  if (Qualifier) {
    // To handle class template specializations,
    // e.g: template<> class numeric_limits<int>.
    if (Qualifier->getKind() == clang::NestedNameSpecifier::TypeSpec) {
      auto CTSDecl = dyn_cast<ClassTemplateSpecializationDecl>(
          DRE->getDecl()->getDeclContext());
      if (CTSDecl) {
        CTSName =
            CTSDecl->getTypeForDecl()->getAsCXXRecordDecl()->getNameAsString();
        CTSName += "::" + DRE->getNameInfo().getAsString();
      }
    }
  }
  if (!CTSName.empty()) {
    RefString = CTSName;
  } else {
    RefString = DRE->getNameInfo().getAsString();
  }

  if (auto TemplateDecl = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl()))
    addReplacement(DRE, TemplateDecl->getIndex());
  else if (auto ECD = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
    auto &ReplEnum = MapNames::findReplacedName(EnumConstantRule::EnumNamesMap,
                                                ECD->getName());
    if (!ReplEnum.empty())
      addReplacement(DRE, ReplEnum);
  }
}

void ExprAnalysis::analyzeExpr(const CXXConstructExpr *Ctor) {
  if (Ctor->getConstructor()->getDeclName().getAsString() == "dim3") {
    std::string ArgsString;
    llvm::raw_string_ostream OS(ArgsString);
    DpctGlobalInfo::printCtadClass(OS, "cl::sycl::range", 3) << "(";
    ArgumentAnalysis A;
    for (auto Arg : Ctor->arguments()) {
      A.analyze(Arg);
      OS << A.getReplacedString() << ", ";
    }
    OS.flush();
    addReplacement(Ctor, ArgsString.replace(ArgsString.length() - 2, 2, ")"));
  }
}

void ExprAnalysis::analyzeExpr(const MemberExpr *ME) {
  static const std::map<std::string, std::string> MemberMap{
      {"__fetch_builtin_x", "2"},
      {"__fetch_builtin_y", "1"},
      {"__fetch_builtin_z", "0"}};
  CtTypeInfo Ty(ME->getBase()->getType());
  if (Ty.getBaseName() == "cl::sycl::range<3>") {
    addReplacement(
        ME->getOperatorLoc(), ME->getMemberLoc(),
        MapNames::findReplacedName(MapNames::Dim3MemberNamesMap,
                                   ME->getMemberNameInfo().getAsString()));
  } else if (Ty.getBaseName() == "dpct::device_info") {
    std::string ReplacementStr = MapNames::findReplacedName(
        DevicePropVarRule::PropNamesMap, ME->getMemberNameInfo().getAsString());
    if (!ReplacementStr.empty()) {
      addReplacement(ME->getMemberLoc(), "get_" + ReplacementStr + "()");
    }
  } else if (Ty.getBaseName() == "const __cuda_builtin_blockIdx_t") {
    ValueDecl *Field = ME->getMemberDecl();
    std::string FieldName = Field->getName();
    if (MapNames::replaceName(MemberMap, FieldName)) {
      std::ostringstream Repl;
      Repl << DpctGlobalInfo::getItemName() << ".get_group(" << FieldName
           << ")";
      addReplacement(ME, Repl.str());
    }
  } else if (Ty.getBaseName() == "const __cuda_builtin_blockDim_t") {
    ValueDecl *Field = ME->getMemberDecl();
    std::string FieldName = Field->getName();
    if (MapNames::replaceName(MemberMap, FieldName)) {
      std::ostringstream Repl;
      Repl << DpctGlobalInfo::getItemName() << ".get_local_range(" << FieldName
           << ")";
      addReplacement(ME, Repl.str());
    }
  } else if (Ty.getBaseName() == "const __cuda_builtin_threadIdx_t") {
    ValueDecl *Field = ME->getMemberDecl();
    std::string FieldName = Field->getName();
    if (MapNames::replaceName(MemberMap, FieldName)) {
      std::ostringstream Repl;
      Repl << DpctGlobalInfo::getItemName() << ".get_local_id(" << FieldName
           << ")";
      addReplacement(ME, Repl.str());
    }
  } else if (MapNames::SupportedVectorTypes.find(Ty.getOrginalBaseType()) !=
             MapNames::SupportedVectorTypes.end()) {
    if (*Ty.getBaseName().rbegin() == '1') {
      addReplacement(ME->getOperatorLoc(), ME->getEndLoc(), "");
      dispatch(ME->getBase());
    } else {
      ExprAnalysis EA;
      EA.analyze(ME->getBase());
      std::string MemberName = ME->getMemberNameInfo().getAsString();
      if (MapNames::replaceName(MapNames::MemberNamesMap, MemberName)) {
        std::ostringstream Repl;
        Repl << "static_cast<" << ME->getType().getAsString() << ">("
             << EA.getReplacedString() << (ME->isArrow() ? "->" : ".")
             << MemberName << ")";
        addReplacement(ME, Repl.str());
      }
    }
  }
}

void ExprAnalysis::analyzeExpr(const UnaryExprOrTypeTraitExpr *UETT) {
  if (UETT->getKind() == UnaryExprOrTypeTrait::UETT_SizeOf) {
    if (UETT->isArgumentType()) {
      analyzeType(UETT->getArgumentTypeInfo());
    } else {
      analyzeExpr(UETT->getArgumentExpr());
    }
  }
}

void ExprAnalysis::analyzeExpr(const CStyleCastExpr *Cast) {
  if (Cast->getCastKind() == CastKind::CK_BitCast)
    analyzeType(Cast->getTypeInfoAsWritten());
  dispatch(Cast->getSubExpr());
}

// Precondition: CE != nullptr
void ExprAnalysis::analyzeExpr(const CallExpr *CE) {
  dispatch(CE->getCallee());
  auto Itr = CallExprRewriterFactoryBase::RewriterMap.find(RefString);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap.end()) {
    auto Result = Itr->second->create(CE)->rewrite();
    if (Result.hasValue())
      addReplacement(CE, Result.getValue());
  } else {
    for (auto Arg : CE->arguments())
      analyzeArgument(Arg);
  }
}

void ExprAnalysis::analyzeType(const TypeLoc &TL) {
  std::string TyName;
  switch (TL.getTypeLocClass()) {
  case TypeLoc::Pointer:
    return analyzeType(static_cast<const PointerTypeLoc &>(TL).getPointeeLoc());
  case TypeLoc::Typedef:
    TyName =
        static_cast<const TypedefTypeLoc &>(TL).getTypedefNameDecl()->getName();
    break;
  case TypeLoc::Builtin:
  case TypeLoc::Record:
    TyName = TL.getType().getAsString();
    break;
  default:
    return;
  }
  if (MapNames::replaceName(MapNames::TypeNamesMap, TyName))
    addReplacement(TL.getBeginLoc(), TL.getEndLoc(), TyName);
}

const std::string &ArgumentAnalysis::getDefaultArgument(const Expr *E) {
  auto &Str = DefaultArgMap[E];
  if (Str.empty())
    Str = getStmtSpelling(E, DpctGlobalInfo::getContext());
  return Str;
}

void KernelArgumentAnalysis::dispatch(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
    ANALYZE_EXPR(DeclRefExpr)
    ANALYZE_EXPR(MemberExpr)
  default:
    return ExprAnalysis::dispatch(Expression);
  }
}

void KernelArgumentAnalysis::analyzeExpr(const DeclRefExpr *DRE) {
  if (DRE->getType()->isReferenceType()) {
    isRedeclareRequired = true;
  } else if (!DRE->getDecl()->isLexicallyWithinFunctionOrMethod()) {
    isRedeclareRequired = true;
  }
  Base::analyzeExpr(DRE);
}

void KernelArgumentAnalysis::analyzeExpr(const MemberExpr *ME) {
  if (ME->getBase()->getType()->isDependentType()) {
    isRedeclareRequired = true;
  }

  if (ME->getBase()->isImplicitCXXThis()) {
    isRedeclareRequired = true;
  } else {
    const MemberExpr *LastExpr = nullptr;
    const MemberExpr *IterExpr =
        dyn_cast_or_null<const MemberExpr>(ME->getBase());
    while (IterExpr) {
      LastExpr = IterExpr;
      IterExpr = dyn_cast_or_null<const MemberExpr>(IterExpr->getBase());
    }

    if (LastExpr && LastExpr->getBase()->isImplicitCXXThis()) {
      isRedeclareRequired = true;
    }
  }

  if (auto RD = ME->getBase()->getType()->getAsCXXRecordDecl()) {
    if (!RD->isStandardLayout()) {
      isRedeclareRequired = true;
    }
  }
  Base::analyzeExpr(ME);
}

void KernelArgumentAnalysis::analyzeExpr(const UnaryOperator *UO) {
  if (UO->getOpcode() == UO_Deref) {
    isRedeclareRequired = true;
    return;
  }
  dispatch(UO->getSubExpr());
}

void KernelConfigAnalysis::dispatch(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
    ANALYZE_EXPR(CXXConstructExpr)
    ANALYZE_EXPR(CXXTemporaryObjectExpr)
  default:
    return ArgumentAnalysis::dispatch(Expression);
  }
}

void KernelConfigAnalysis::analyzeExpr(const CXXConstructExpr *Ctor) {
  if (Ctor->getConstructor()->getDeclName().getAsString() == "dim3") {
    std::string CtorString;
    llvm::raw_string_ostream OS(CtorString);
    DpctGlobalInfo::printCtadClass(OS, "cl::sycl::range", 3) << "(";
    auto Args = getCtorArgs(Ctor);
    if (DoReverse && Ctor->getNumArgs() == 3) {
      Reversed = true;
      int Index = Args.size();
      while (Index)
        OS << Args[--Index] << ", ";
    } else {
      for (auto &A : Args)
        OS << A << ", ";
    }
    OS.flush();
    return addReplacement(Ctor,
                          CtorString.replace(CtorString.length() - 2, 2, ")"));
  }
  return ArgumentAnalysis::analyzeExpr(Ctor);
}

std::vector<std::string>
KernelConfigAnalysis::getCtorArgs(const CXXConstructExpr *Ctor) {
  std::vector<std::string> Args;
  ArgumentAnalysis A;
  for (auto Arg : Ctor->arguments())
    Args.emplace_back(getCtorArg(A, Arg));
  return Args;
}

} // namespace dpct
} // namespace clang
