//===--- ExprAnalysis.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
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
namespace syclct {

#define ANALYZE_EXPR(EXPR)                                                    \
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
    return SM.getSpellingLoc(Loc);
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

void ExprAnalysis::initExpression(const Expr *Expression) {
  E = Expression;
  SrcBegin = 0;
  if (E && E->getBeginLoc().isValid()) {
    std::tie(SrcBegin, SrcLength) =
        getOffsetAndLength(E->getBeginLoc(), E->getEndLoc());
    ReplSet.init(std::string(
        SM.getCharacterData(getExprLocation(E->getBeginLoc())), SrcLength));
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
    : Context(SyclctGlobalInfo::getContext()),
      SM(SyclctGlobalInfo::getSourceManager()) {
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

void ExprAnalysis::analyzeExpr(const CXXConstructExpr *Ctor) {
  const std::string Dim3Constructor = "dim3";
  if (Ctor->getConstructor()->getDeclName().getAsString() == Dim3Constructor) {
    std::string ArgsString = "cl::sycl::range<3>(";
    ArgumentAnalysis A;
    for (auto Arg : Ctor->arguments()) {
      A.analyze(Arg);
      ArgsString += A.getReplacedString() + ", ";
    }
    ArgsString.replace(ArgsString.length() - 2, 2, ")");
    addReplacement(Ctor, ArgsString);
  }
}

void ExprAnalysis::analyzeExpr(const MemberExpr *ME) {
  CtTypeInfo Ty(ME->getBase()->getType());
  if (Ty.getBaseName() == "cl::sycl::range<3>")
    addReplacement(
        ME->getOperatorLoc(), ME->getMemberLoc(),
        MapNames::findReplacedName(MapNames::Dim3MemberNamesMap,
                                   ME->getMemberNameInfo().getAsString()));
}

void ExprAnalysis::analyzeExpr(const UnaryExprOrTypeTraitExpr *UETT) {
  if (UETT->getKind() == UnaryExprOrTypeTrait::UETT_SizeOf)
    analyzeType(UETT->getArgumentTypeInfo());
}

void ExprAnalysis::analyzeExpr(const CStyleCastExpr *Cast) {
  if (Cast->getCastKind() == CastKind::CK_BitCast)
    analyzeType(Cast->getTypeInfoAsWritten());
  dispatch(Cast->getSubExpr());
}

void ExprAnalysis::analyzeExpr(const CallExpr *CE) {
  dispatch(CE->getCallee());
  auto Itr = CallExprRewriterFactoryBase::CallMap.find(RefString);
  if (Itr != CallExprRewriterFactoryBase::CallMap.end()) {
    auto Result = Itr->second->create(CE)->rewrite();
    if (Result.hasValue())
      addReplacement(CE, Result.getValue());
  } else if (auto FD = CE->getDirectCallee()) {
    if (!FD->hasAttr<CUDADeviceAttr>())
      return;
    auto Itr = MathFunctionsRule::SingleDoubleFunctionNamesMap.find(RefString);
    if (Itr != MathFunctionsRule::SingleDoubleFunctionNamesMap.end())
      addReplacement(
          CE,
          FuncCallExprRewriterFactory(Itr->second).create(CE)->rewrite().getValue());
    else {
      for (auto Arg : CE->arguments())
        analyzeArgument(Arg);
    }
  }
}

void ExprAnalysis::analyzeType(const TypeLoc &TL) {
  std::string TyName;
  switch (TL.getTypeLocClass()) {
  case TypeLoc::Pointer:
    return analyzeType(
        static_cast<const PointerTypeLoc &>(TL).getPointeeLoc());
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
    Str = getStmtSpelling(E, SyclctGlobalInfo::getContext());
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
  if (auto D = dyn_cast<VarDecl>(DRE->getDecl())) {
    auto LocInfo = SyclctGlobalInfo::getLocInfo(D);
    if (DRE->getType()->isPointerType()) {
      insertObject(PointerVarMap, LocInfo.second, LocInfo.first, D);
    } else if (D->getType()->isReferenceType()) {
      addReplacement(DRE,
                     insertObject(RefVarMap, LocInfo.second, LocInfo.first, D)
                         ->getDerefName());
    }
  }
  Base::analyzeExpr(DRE);
}

void KernelArgumentAnalysis::analyzeExpr(const MemberExpr *ME) {
  if (auto D = dyn_cast<FieldDecl>(ME->getMemberDecl())) {
    auto LocInfo = SyclctGlobalInfo::getLocInfo(ME);
    auto MEStr = getStmtSpelling(ME, SyclctGlobalInfo::getContext());
    if (ME->getType()->isPointerType()) {
      addReplacement(ME, insertObject(PointerVarMap, LocInfo.second,
                                      LocInfo.first, D, MEStr)
                             ->getName());
    } else if (D->getType()->isReferenceType()) {
      addReplacement(
          ME, insertObject(RefVarMap, LocInfo.second, LocInfo.first, D, MEStr)
                  ->getDerefName());
    } else if (ME->getBase()->isImplicitCXXThis()) {
      // Dereference implicit "this" pointer, for kernel functions can't capture
      // host pointer.
      auto LocInfo = SyclctGlobalInfo::getLocInfo(ME->getMemberDecl());
      addReplacement(
          ME, insertObject(RefVarMap, LocInfo.second, LocInfo.first, D, MEStr)
                  ->getDerefName());
    } else {
      // While base is still member expression, continue analyze it.
      // Like a.b.c, will continue analyze "a.b".
      if (auto Sub = dyn_cast<MemberExpr>(ME->getBase()->IgnoreImpCasts()))
        analyzeExpr(Sub);
    }
  }
  Base::analyzeExpr(ME);
}

KernelArgumentAnalysis::~KernelArgumentAnalysis() {
  mapToList(PointerVarMap, PointerVarList);
  mapToList(RefVarMap, RefVarList);
}
} // namespace syclct
} // namespace clang
