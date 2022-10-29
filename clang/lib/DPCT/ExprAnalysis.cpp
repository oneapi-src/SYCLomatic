//===--------------- ExprAnalysis.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExprAnalysis.h"

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "CallExprRewriter.h"
#include "DNNAPIMigration.h"
#include "TypeLocRewriters.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprOpenMP.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtGraphTraits.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "MemberExprRewriter.h"

extern std::string DpctInstallPath;
namespace clang {
namespace dpct {

#define ANALYZE_EXPR(EXPR)                                                     \
  case Stmt::EXPR##Class:                                                      \
    return analyzeExpr(static_cast<const EXPR *>(Expression));

std::map<const Expr *, std::string> ArgumentAnalysis::DefaultArgMap;

void TemplateDependentReplacement::replace(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  SourceStr.replace(Offset, Length,
                    getTargetArgument(TemplateList).getString());
}

TemplateDependentStringInfo::TemplateDependentStringInfo(
    const std::string &SrcStr,
    const std::map<size_t, std::shared_ptr<TemplateDependentReplacement>>
        &InTDRs)
    : SourceStr(SrcStr) {
  for (const auto &TDR : InTDRs)
    TDRs.emplace_back(TDR.second->alterSource(SourceStr));
}

std::shared_ptr<TemplateDependentStringInfo>
TemplateDependentStringInfo::applyTemplateArguments(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  std::shared_ptr<TemplateDependentStringInfo> Result =
      std::make_shared<TemplateDependentStringInfo>();
  Result->SourceStr = SourceStr;
  int OffsetShift = 0;
  auto &Repls = Result->TDRs;
  auto &Str = Result->SourceStr;
  Result->IsDependOnWrittenArgument = true;
  for (auto &R : TDRs) {
    size_t ReplsSize = Repls.size();
    size_t ApplyOffset = R->getOffset() + OffsetShift;
    auto &TargetArg = R->getTargetArgument(TemplateList);
    auto &TargetList = TargetArg.getDependentStringInfo()->TDRs;
    Repls.resize(ReplsSize + TargetList.size());

    Str.replace(ApplyOffset, R->getLength(), TargetArg.getString());
    std::transform(TargetList.begin(), TargetList.end(),
                   Repls.begin() + ReplsSize,
                   [&](std::shared_ptr<TemplateDependentReplacement> OldRepl)
                       -> std::shared_ptr<TemplateDependentReplacement> {
                     auto Repl = OldRepl->alterSource(Str);
                     Repl->shift(ApplyOffset);
                     return Repl;
                   });
    OffsetShift += TargetArg.getString().length() - R->getLength();
    Result->IsDependOnWrittenArgument &= TargetArg.isWritten();
  }
  return Result;
}
const TemplateArgumentInfo &TemplateDependentReplacement::getTargetArgument(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  if (TemplateIndex < TemplateList.size())
    return TemplateList[TemplateIndex];
  static TemplateArgumentInfo TAI;
  return TAI;
}

SourceLocation ExprAnalysis::getExprLocation(SourceLocation Loc) {
  if (SM.isMacroArgExpansion(Loc))
    return getExprLocation(SM.getImmediateSpellingLoc(Loc));
  else
    return SM.getExpansionLoc(Loc);
}

std::pair<SourceLocation, size_t>
ExprAnalysis::getSpellingOffsetAndLength(SourceLocation Loc) {
  if (Loc.isInvalid())
    return std::pair<SourceLocation, size_t>(Loc, SrcLength);
  Loc = SM.getSpellingLoc(Loc);
  auto TokenLen = Lexer::MeasureTokenLength(Loc, SM, Context.getLangOpts());
  Token Tok2;
  Lexer::getRawToken(Loc, Tok2, SM, Context.getLangOpts());
  // if the last token is ">>" or ">>>",
  // since DPCT does not support nested template type migration,
  // the last token should be treated as ">"
  if (Tok2.is(tok::greatergreater) || Tok2.is(tok::greatergreatergreater)) {
    TokenLen = 1;
  }

  return std::pair<SourceLocation, size_t>(
      Loc, TokenLen);
}

std::pair<SourceLocation, size_t>
ExprAnalysis::getSpellingOffsetAndLength(SourceLocation BeginLoc,
                                         SourceLocation EndLoc) {
  if (EndLoc.isValid()) {
    auto Begin = SM.getSpellingLoc(BeginLoc);
    auto End = getSpellingOffsetAndLength(EndLoc);
    return std::pair<SourceLocation, size_t>(
        Begin, SM.getCharacterData(End.first) - SM.getCharacterData(Begin) +
                   End.second);
  }
  return getSpellingOffsetAndLength(BeginLoc);
}

std::pair<SourceLocation, size_t>
ExprAnalysis::getSpellingOffsetAndLength(const Expr *E) {
  auto ResultRange = getDefinitionRange(E->getBeginLoc(), E->getEndLoc());
  auto LastTokenLength = Lexer::MeasureTokenLength(ResultRange.getEnd(), SM,
                                                   Context.getLangOpts());
  return std::pair<SourceLocation, size_t>(
      ResultRange.getBegin(), SM.getCharacterData(ResultRange.getEnd()) -
                                  SM.getCharacterData(ResultRange.getBegin()) +
                                  LastTokenLength);
}

std::pair<size_t, size_t> ExprAnalysis::getOffsetAndLength(SourceLocation Loc) {
  if (Loc.isInvalid())
    return std::pair<size_t, size_t>(0, SrcLength);
  Loc = getExprLocation(Loc);
  FileId = SM.getDecomposedLoc(Loc).first;
  auto TokenLen = Lexer::MeasureTokenLength(Loc, SM, Context.getLangOpts());
  Token Tok2;
  Lexer::getRawToken(Loc, Tok2, SM, Context.getLangOpts());
  // if the last token is ">>" or ">>>",
  // since DPCT does not support nested template type migration,
  // the last token should be treated as ">"
  if(Tok2.is(tok::greatergreater) || Tok2.is(tok::greatergreatergreater)){
    TokenLen = 1;
  }
  return std::pair<size_t, size_t>(
      getOffset(Loc),
      TokenLen);
}

std::pair<size_t, size_t>
ExprAnalysis::getOffsetAndLength(SourceLocation BeginLoc,
                                 SourceLocation EndLoc) {
  if (EndLoc.isValid()) {
    if (BeginLoc.isFileID() || EndLoc.isFileID()) {
      // No Macro or only one of begin/end is macro
      BeginLoc = SM.getExpansionLoc(BeginLoc);
      EndLoc = SM.getExpansionLoc(EndLoc);
    } else if (SM.isMacroArgExpansion(BeginLoc) &&
               SM.isMacroArgExpansion(EndLoc)) {
      bool IsSameFuncLikeMacro = false;
      bool IsSameMacroArgExpansion = false;
      auto BeginImmSpelling = getImmSpellingLocRecursive(BeginLoc);
      auto EndImmSpelling = getImmSpellingLocRecursive(EndLoc);
      if (isSameLocation(SM.getExpansionLoc(BeginImmSpelling),
                         SM.getExpansionLoc(EndImmSpelling))) {
        IsSameFuncLikeMacro = true;
        if (isSameLocation(SM.getImmediateExpansionRange(BeginLoc).getBegin(),
                           SM.getImmediateExpansionRange(EndLoc).getBegin())) {
          IsSameMacroArgExpansion = true;
        }
      }

      if (IsSameMacroArgExpansion) {
        // #define ALL3(X) X
        // ALL3(const int2 *)
        BeginLoc = SM.getSpellingLoc(BeginLoc);
        EndLoc = SM.getSpellingLoc(EndLoc);
      } else if (IsSameFuncLikeMacro) {
        // #define ALL2(C, T, P) C T P
        // ALL2(const, int2, *)
        BeginLoc = SM.getImmediateExpansionRange(BeginLoc).getBegin();
        EndLoc = SM.getImmediateExpansionRange(EndLoc).getBegin();
      } else {
        // ALL3(const) ALL3(int2) ALL3(*)
        BeginLoc = SM.getExpansionRange(BeginLoc).getBegin();
        EndLoc = SM.getExpansionRange(EndLoc).getBegin();
      }
    } else {
      if (SM.isMacroArgExpansion(BeginLoc)) {
        BeginLoc = getImmSpellingLocRecursive(BeginLoc);
      } else {
        EndLoc = getImmSpellingLocRecursive(EndLoc);
      }
      auto ItBegin =
          dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
              getCombinedStrFromLoc(SM.getSpellingLoc(BeginLoc)));
      auto ItEnd = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
          getCombinedStrFromLoc(SM.getSpellingLoc(EndLoc)));
      if (isSameLocation(SM.getExpansionLoc(BeginLoc),
                         SM.getExpansionLoc(EndLoc)) &&
          ItBegin !=
              dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
          ItEnd !=
              dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
          ItBegin->second->TokenIndex == 0 &&
          ItEnd->second->TokenIndex == ItEnd->second->NumTokens - 1) {
        // Begin/end contain the whole Macro def
        // ex: #define TYPE const int2*
        BeginLoc = SM.getExpansionLoc(BeginLoc);
        EndLoc = SM.getExpansionLoc(EndLoc);
      } else {
        // 1. only one of Begin/End is MacroArgExpansion and the other one is
        // Body MacroID ex: #define TYPE_PTR(T) T*
        // 2. Both Begin/End are body MacroID and Begin/end are not in the same
        // Macro def ex: #define TYPE int2
        //     #define PTR *
        //     #define TYPE_PTR TYPE PTR
        std::tie(BeginLoc, EndLoc) =
            getTheLastCompleteImmediateRange(BeginLoc, EndLoc);
      }
    }

    auto Begin = getOffset(getExprLocation(BeginLoc));
    auto End = getOffsetAndLength(EndLoc);
    if (SrcBeginLoc.isInvalid())
      SrcBeginLoc = BeginLoc;

    // Avoid illegal range which will cause SIGABRT
    if (End.first + End.second < Begin) {
      return std::pair<size_t, size_t>(Begin, 0);
    }
    return std::pair<size_t, size_t>(Begin, End.first - Begin + End.second);
  }
  return getOffsetAndLength(BeginLoc);
}

std::pair<size_t, size_t>
ExprAnalysis::getOffsetAndLength(SourceLocation BeginLoc, SourceLocation EndLoc,
                                 const Expr *Parent) {
  const std::shared_ptr<DynTypedNode> P =
      std::make_shared<DynTypedNode>(DynTypedNode::create(*Parent));
  if (BeginLoc.isMacroID() && isInsideFunctionLikeMacro(BeginLoc, EndLoc, P)) {
    BeginLoc = SM.getExpansionLoc(SM.getImmediateSpellingLoc(BeginLoc));
    EndLoc = SM.getExpansionLoc(SM.getImmediateSpellingLoc(EndLoc));
  } else {
    if (EndLoc.isValid()) {
      BeginLoc = SM.getExpansionRange(BeginLoc).getBegin();
      EndLoc = SM.getExpansionRange(EndLoc).getEnd();
    }
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

std::pair<size_t, size_t> ExprAnalysis::getOffsetAndLength(const Expr *E) {
  SourceLocation BeginLoc, EndLoc;
  size_t End;

  if (IsInMacroDefine) {
    // If the expr is in macro define, and the CallSpellingBegin/End is set,
    // we can use the CallSpellingBegin/End to get a more precise range.
    bool RangeInCall = false;
    if (CallSpellingBegin.isValid() && CallSpellingEnd.isValid()) {
      auto Range = getRangeInRange(E, CallSpellingBegin, CallSpellingEnd);
      auto DLBegin = SM.getDecomposedLoc(Range.first);
      auto DLEnd = SM.getDecomposedLoc(Range.second);
      if (DLBegin.first == DLEnd.first &&
          DLBegin.second <= DLEnd.second) {
        BeginLoc = Range.first;
        EndLoc = Range.second;
        End = getOffset(EndLoc);
        RangeInCall = true;
      }
    }
    // In cases like:
    // #define CCC <<<1,1>>>()
    // #define KERNEL foo CCC
    // The getRangeInRange cannot find the correct range,
    // fallback to use heuristics.
    if (!RangeInCall) {
      if (SM.isMacroArgExpansion(E->getBeginLoc())) {
        BeginLoc = SM.getSpellingLoc(
            SM.getImmediateExpansionRange(E->getBeginLoc()).getBegin());
        EndLoc = SM.getSpellingLoc(
            SM.getImmediateExpansionRange(E->getEndLoc()).getEnd());
      } else {
        BeginLoc =
            SM.getExpansionLoc(SM.getImmediateSpellingLoc(E->getBeginLoc()));
        EndLoc = SM.getExpansionLoc(SM.getImmediateSpellingLoc(E->getEndLoc()));
        if (isExprStraddle(E)) {
          auto Range = getTheOneBeforeLastImmediateExapansion(E->getBeginLoc(),
                                                              E->getEndLoc());
          BeginLoc = SM.getImmediateSpellingLoc(Range.first);
          EndLoc = SM.getImmediateSpellingLoc(Range.second);
        }
      }
      End = getOffset(EndLoc) +
            Lexer::MeasureTokenLength(EndLoc, SM, Context.getLangOpts());
    }
  } else {
    // If the Expr is FileID or is macro arg
    // e.g. CALL(expr)
    auto Range = getStmtExpansionSourceRange(E);
    BeginLoc = Range.getBegin();
    EndLoc = Range.getEnd();
    End = getOffset(EndLoc) + Lexer::MeasureTokenLength(EndLoc, SM, Context.getLangOpts());
  }

  // Find the begin/end location include prefix and postfix
  // Set Prefix and Postfix strings
  auto BeginLocWithoutPrefix = BeginLoc;
  BeginLoc = getBeginLocOfPreviousEmptyMacro(BeginLoc);
  auto RewritePrefixLength = SM.getCharacterData(BeginLocWithoutPrefix) -
                             SM.getCharacterData(BeginLoc);

  auto EndLocWithoutPostfix = EndLoc;
  EndLoc = SM.getExpansionLoc(getEndLocOfFollowingEmptyMacro(EndLoc));
  auto RewritePostfixLength =
      SM.getCharacterData(EndLoc) - SM.getCharacterData(EndLocWithoutPostfix);

  ExprBeginLoc = BeginLoc;
  ExprEndLoc = EndLoc;
  RewritePrefix =
      std::string(SM.getCharacterData(BeginLoc), RewritePrefixLength);

  // Get the token end of EndLocWithoutPostfix and EndLoc for correct
  // RewritePostfix
  EndLocWithoutPostfix =
      EndLocWithoutPostfix.getLocWithOffset(Lexer::MeasureTokenLength(
          EndLocWithoutPostfix, SM,
          dpct::DpctGlobalInfo::getContext().getLangOpts()));
  EndLoc = EndLoc.getLocWithOffset(Lexer::MeasureTokenLength(
      EndLoc, SM, dpct::DpctGlobalInfo::getContext().getLangOpts()));

  RewritePostfix = std::string(SM.getCharacterData(EndLocWithoutPostfix),
                               RewritePostfixLength);

  auto DecompLoc = SM.getDecomposedLoc(BeginLoc);
  FileId = DecompLoc.first;
  // The offset of Expr used in ExprAnalysis is related to SrcBegin not
  // FileBegin
  auto Begin = DecompLoc.second - SrcBegin;
  return std::pair<size_t, size_t>(Begin, End - Begin);
}

void ExprAnalysis::initExpression(const Expr *Expression) {
  E = Expression;
  SrcBegin = 0;
  if (E && E->getBeginLoc().isValid()) {
    std::tie(SrcBegin, SrcLength) = getOffsetAndLength(E);
    ReplSet.init(
        std::string(SM.getBufferData(FileId).substr(SrcBegin, SrcLength)));
  } else {
    SrcLength = 0;
    ReplSet.init("");
  }
}

void ExprAnalysis::initSourceRange(const SourceRange &Range) {
  SrcBegin = 0;
  if (Range.getBegin().isValid()) {
    std::tie(SrcBegin, SrcLength) =
        getOffsetAndLength(Range.getBegin(), Range.getEnd());
    if (auto FileBuffer = SM.getBufferOrNone(FileId)) {
      ReplSet.init(std::string(
          FileBuffer.value().getBuffer().data() + SrcBegin, SrcLength));
      return;
    }
  }
  SrcLength = 0;
  ReplSet.init("");
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
  analyze(Expression);
}

void ExprAnalysis::dispatch(const Stmt *Expression) {
  if (!Expression)
    return;
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

bool isMathFunction(std::string Name) {
  static std::set<std::string> MathFunctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_OPERATOR(APINAME, OPKIND) APINAME,
#define ENTRY_TYPECAST(APINAME) APINAME,
#define ENTRY_UNSUPPORTED(APINAME) APINAME,
#define ENTRY_REWRITE(APINAME)
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_NO_REWRITE
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
#undef ENTRY_REWRITE
  };
  return MathFunctions.count(Name);
}

bool isCGAPI(std::string Name) {
  return MapNames::CooperativeGroupsAPISet.count(Name);
}

void ExprAnalysis::analyzeExpr(const DeclRefExpr *DRE) {
  std::string CTSName;
  auto Qualifier = DRE->getQualifier();
  if (Qualifier) {
    bool IsNamespaceOrAlias =
        Qualifier->getKind() ==
            clang::NestedNameSpecifier::SpecifierKind::Namespace ||
        Qualifier->getKind() ==
            clang::NestedNameSpecifier::SpecifierKind::NamespaceAlias;
    bool IsSpecicalAPI = isMathFunction(DRE->getNameInfo().getAsString()) ||
                         isCGAPI(DRE->getNameInfo().getAsString());
    if (!IsNamespaceOrAlias || !IsSpecicalAPI) {
      CTSName = getNestedNameSpecifierString(Qualifier) +
                DRE->getNameInfo().getAsString();
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
    std::unordered_set<std::string> targetStr = {
        "thread_scope_system",  "thread_scope_device",  "thread_scope_block",
        "memory_order_relaxed", "memory_order_acq_rel", "memory_order_release",
        "memory_order_acquire", "memory_order_seq_cst"};

    if (targetStr.find(ECD->getNameAsString()) != targetStr.end())
      if (const auto *ED = dyn_cast<EnumDecl>(ECD->getDeclContext())) {
        std::string NameString = "";
        llvm::raw_string_ostream NameStringOS(NameString);
        for (const auto *NSD = dyn_cast<NamespaceDecl>(ED->getDeclContext());
             NSD; NSD = dyn_cast<NamespaceDecl>(NSD->getDeclContext())) {
          if (NSD->getName() == "__detail" || NSD->isInline() ||
              NSD->getName() == "std")
            continue;
          NameStringOS << NSD->getNameAsString() << "::";
        }
        NameStringOS << ECD->getNameAsString();
        RefString = NameString;
      }

    auto &ReplEnum =
        MapNames::findReplacedName(EnumConstantRule::EnumNamesMap, RefString);
    requestHelperFeatureForEnumNames(RefString, DRE);
    auto ItEnum = EnumConstantRule::EnumNamesMap.find(RefString);
    if (ItEnum != EnumConstantRule::EnumNamesMap.end()) {
      for (auto ItHeader = ItEnum->second->Includes.begin();
           ItHeader != ItEnum->second->Includes.end(); ItHeader++) {
        DpctGlobalInfo::getInstance().insertHeader(DRE->getBeginLoc(),
                                                   *ItHeader);
      }
    }
    if (!ReplEnum.empty())
      addReplacement(DRE, ReplEnum);
    else {
      auto &ReplBLASEnum = MapNames::findReplacedName(MapNames::BLASEnumsMap,
                                                      ECD->getName().str());
      if (!ReplBLASEnum.empty())
        addReplacement(DRE, ReplBLASEnum);
      else {
        auto &ReplFuncAttrEnum = MapNames::findReplacedName(
            MapNames::FunctionAttrMap, ECD->getName().str());
        if (!ReplFuncAttrEnum.empty())
          addReplacement(DRE, ReplFuncAttrEnum);
        else {
          auto &CuDNNEnum = MapNames::findReplacedName(
              CuDNNTypeRule::CuDNNEnumNamesMap, ECD->getName().str());
          if (!CuDNNEnum.empty())
            addReplacement(DRE, CuDNNEnum);
        }
      }
    }
  } else if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    if (RefString == "warpSize" &&
        !DpctGlobalInfo::isInAnalysisScope(VD->getLocation())) {
      addReplacement(DRE, DpctGlobalInfo::getSubGroup(DRE) +
                              ".get_local_range().get(0)");
    }
  }
}

// Get replacement str and replacement length for thrust construct type.
// If thrust construct type have mapping type in MapNames::TypeNamesMap, using
// the mapping type, else just use "std::" to replace "thrust::".
void ExprAnalysis::getThrustReplStrAndLength(const std::string &CtorClassName,
                                             std::string &Replacement,
                                             size_t &TypeLen) {
  if (CtorClassName.find("thrust::") == 0) {
    TypeLen = CtorClassName.find('<');
    if (TypeLen == std::string::npos)
      TypeLen = CtorClassName.size();

    auto RealTypeNameStr = CtorClassName.substr(0, TypeLen);
    Replacement =
        MapNames::findReplacedName(MapNames::TypeNamesMap, RealTypeNameStr);
    if (Replacement.empty()) {
      static const size_t ThrustNamespaceLength = std::strlen("thrust::");
      TypeLen = ThrustNamespaceLength;
      Replacement = "std::";
    }
  }
}

void ExprAnalysis::analyzeExpr(const ConstantExpr *CE) {
  dispatch(CE->getSubExpr());
}

void ExprAnalysis::analyzeExpr(const IntegerLiteral *IL) {
  auto DefinitionRange = getDefinitionRange(IL->getBeginLoc(), IL->getEndLoc());
  auto TokBeginLoc = DefinitionRange.getBegin();
  auto TokenLength = Lexer::MeasureTokenLength(
      TokBeginLoc, SM, Context.getLangOpts());
  std::string TokStr(SM.getCharacterData(TokBeginLoc), TokenLength);

  // TODO: cannot handle case like:
  // #define CHECK(ARG) ARG
  // #define MACRO CHECK(cufftExecC2C(plan, idata, odata, CUFFT_FORWARD))
  // void foo(cufftHandle plan, cufftComplex *idata, cufftComplex *odata) {
  //   MACRO;
  // }
  const Expr *ParentExpr = DpctGlobalInfo::findParent<Expr>(IL);
  bool IsInCudaPath = DpctGlobalInfo::isInCudaPath(
      DpctGlobalInfo::getLocInfo(SM.getSpellingLoc(IL->getBeginLoc())).first);
  if (TokStr == "CUFFT_FORWARD" && ParentExpr && IsInCudaPath) {
    addReplacement(DefinitionRange, ParentExpr,
                   MapNames::getDpctNamespace() + "fft::fft_direction::forward");
    requestFeature(HelperFeatureEnum::FftUtils_fft_direction, TokBeginLoc);
  } else if ((TokStr == "CUFFT_INVERSE") && ParentExpr && IsInCudaPath) {
    addReplacement(DefinitionRange, ParentExpr,
                   MapNames::getDpctNamespace() + "fft::fft_direction::backward");
    requestFeature(HelperFeatureEnum::FftUtils_fft_direction, TokBeginLoc);
  }
}

void ExprAnalysis::analyzeExpr(const InitListExpr *ILE) {
  if (const CXXFunctionalCastExpr *CFCE =
          DpctGlobalInfo::findParent<CXXFunctionalCastExpr>(ILE)) {
    // 'int64_t' might be alias to 'long'(LP64) or 'long long'(LLP64)
    std::string Int64CanonicalType = DpctGlobalInfo::getUnqualifiedTypeName(
        Context.getIntTypeForBitwidth(64, true)->getCanonicalTypeUnqualified());
    std::string CastType = DpctGlobalInfo::getUnqualifiedTypeName(
        CFCE->getType()->getCanonicalTypeUnqualified());
    if (CFCE->isListInitialization() && CFCE->getType()->isIntegerType() &&
        ILE->getNumInits() == 1) {
      const auto *ME = dyn_cast<MemberExpr>(ILE->getInit(0)->IgnoreImplicit());
      if (CastType == Int64CanonicalType && ME &&
          !ME->isIntegerConstantExpr(Context)) {
        auto QT = ME->getBase()->getType();
        if (QT->isPointerType()) {
          QT = QT->getPointeeType();
        }
        if (DpctGlobalInfo::getUnqualifiedTypeName(
                QT->getCanonicalTypeUnqualified()) == "dim3") {
          // Replace initializer list with explicit type conversion (e.g.,
          // 'int64_t{d3[2]}' to 'int64_t(d3[2])') to slience narrowing
          // error (e.g., 'size_t -> int64_t') for
          // non-constant-expression in int64_t initializer list.
          // E.g.,
          // dim3 d3; int64_t{d3.x};
          // will be migratd to
          // sycl::range<3> d3; int64_t(d3[2]);
          addReplacement(ILE->getLBraceLoc(), "(");
          addReplacement(ILE->getRBraceLoc(), ")");
        }
      }
    }
  }
  for (const auto &Init : ILE->inits())
    dispatch(Init);
}

void ExprAnalysis::analyzeExpr(const CXXUnresolvedConstructExpr *Ctor) {
  analyzeType(Ctor->getTypeSourceInfo()->getTypeLoc());
  for (auto It = Ctor->arg_begin(); It != Ctor->arg_end(); It++) {
    dispatch(*It);
  }
}

void ExprAnalysis::analyzeExpr(const CXXConstructExpr *Ctor) {
  std::string CtorClassName =
      Ctor->getConstructor()->getParent()->getQualifiedNameAsString();
  if (CtorClassName.find("thrust::") == 0) {
    // Distinguish CXXTemporaryObjectExpr from other copy ctor before migrating
    // the ctor. Ex. foo(thrust::minus<int>());
    auto CXXTemporaryObjectExprMatcher = clang::ast_matchers::findAll(
        clang::ast_matchers::cxxTemporaryObjectExpr().bind("CTOE"));
    auto MatchedResults =
        clang::ast_matchers::match(CXXTemporaryObjectExprMatcher, *Ctor,
                                   clang::dpct::DpctGlobalInfo::getContext());
    if (MatchedResults.size() > 0) {
      std::string Replacement;
      size_t TypeLen;
      getThrustReplStrAndLength(CtorClassName, Replacement, TypeLen);
      addReplacement(Ctor, TypeLen, Replacement);
    }
  }

  if (Ctor->getConstructor()->getDeclName().getAsString() == "dim3") {
    std::string ArgsString;
    llvm::raw_string_ostream OS(ArgsString);
    DpctGlobalInfo::printCtadClass(OS, MapNames::getClNamespace() + "range", 3)
        << "(";
    ArgumentAnalysis A;
    std::string ArgStr = "";
    for (auto Arg : Ctor->arguments()) {
      A.analyze(Arg);
      ArgStr = ", " + A.getReplacedString() + ArgStr;
    }
    ArgStr.replace(0, 2, "");
    OS << ArgStr << ")";
    OS.flush();

    // Special handling for implicit ctor.
    // #define GET_BLOCKS(a) a
    // dim3 A = GET_BLOCKS(1);
    // Result if using SM.getExpansionRange:
    //   sycl::range<3> A = sycl::range<3>(1, 1, GET_BLOCKS(1));
    // Result if using addReplacement(E):
    //   #define GET_BLOCKS(a) sycl::range<3>(1, 1, a)
    //   sycl::range<3> A = GET_BLOCKS(1);
    if (Ctor->getParenOrBraceRange().isInvalid() && isOuterMostMacro(Ctor)) {
      return addReplacement(
          SM.getExpansionRange(Ctor->getBeginLoc()).getBegin(),
          SM.getExpansionRange(Ctor->getEndLoc()).getEnd(), ArgsString);
    }
    addReplacement(Ctor, ArgsString);
    return;
  }
  for (auto It = Ctor->arg_begin(); It != Ctor->arg_end(); It++) {
    dispatch(*It);
  }
}

void ExprAnalysis::analyzeExpr(const MemberExpr *ME) {
  auto PP = DpctGlobalInfo::getContext().getPrintingPolicy();
  PP.PrintCanonicalTypes = true;
  auto QT = ME->getBase()->getType();
  auto BaseType = (QT->isPointerType() ? QT->getPointeeType() : QT)
                      .getUnqualifiedType()
                      .getAsString(PP);

  std::string FieldName = "";
  if (ME->getMemberDecl()->getIdentifier()) {
    FieldName = ME->getMemberDecl()->getName().str();
    auto MemberExprName = BaseType + "." + FieldName;
    auto ItFieldRule = MapNames::ClassFieldMap.find(MemberExprName);
    if (!MemberExprRewriterFactoryBase::MemberExprRewriterMap)
      return;
    auto Itr = MemberExprRewriterFactoryBase::MemberExprRewriterMap->find(MemberExprName);
    if (Itr != MemberExprRewriterFactoryBase::MemberExprRewriterMap->end()) {
      auto Rewriter = Itr->second->create(ME);
      auto Result = Rewriter->rewrite();
      if (Result.has_value()) {
        auto ResultStr = Result.value();
        addReplacement(ME->getBeginLoc(), ME->getEndLoc(), Result.value());
      }
      return;
    }
    if (ItFieldRule != MapNames::ClassFieldMap.end()) {
      if (ItFieldRule->second->GetterName == "") {
        addReplacement(ME->getMemberLoc(), ME->getMemberLoc(),
                       ItFieldRule->second->NewName);
        return;
      } else {
        if (auto BO = DpctGlobalInfo::findAncestor<BinaryOperator>(ME)) {
          if (BO->getOpcode() == BinaryOperatorKind::BO_Assign &&
              ME == BO->getLHS()) {
            ExprAnalysis EA;
            EA.analyze(BO->getRHS());
            std::string RHSStr = EA.getReplacedString();
            addReplacement(ME->getMemberLoc(), ME->getMemberLoc(),
                           ItFieldRule->second->SetterName + "(" + RHSStr +
                               ")");
            auto SpellingLocInfo = getSpellingOffsetAndLength(
                BO->getOperatorLoc(), BO->getOperatorLoc());
            addExtReplacement(std::make_shared<ExtReplacement>(
                SM, SpellingLocInfo.first, SpellingLocInfo.second, "",
                nullptr));
            SpellingLocInfo = getSpellingOffsetAndLength(
                BO->getRHS()->getBeginLoc(), BO->getRHS()->getEndLoc());
            addExtReplacement(std::make_shared<ExtReplacement>(
                SM, SpellingLocInfo.first, SpellingLocInfo.second, "",
                nullptr));
          }
        } else {
          addReplacement(ME->getMemberLoc(), ME->getMemberLoc(),
                         ItFieldRule->second->GetterName + "()");
        }
      }
    }
  }

  static MapNames::MapTy NdItemMemberMap{{"__fetch_builtin_x", "2"},
                                         {"__fetch_builtin_y", "1"},
                                         {"__fetch_builtin_z", "0"}};
  static const MapNames::MapTy NdItemMap{
      {"__cuda_builtin_blockIdx_t", "get_group"},
      {"__cuda_builtin_gridDim_t", "get_group_range"},
      {"__cuda_builtin_blockDim_t", "get_local_range"},
      {"__cuda_builtin_threadIdx_t", "get_local_id"},
  };

  auto ItemItr = NdItemMap.find(BaseType);
  if (ItemItr != NdItemMap.end()) {
    if (MapNames::replaceName(NdItemMemberMap, FieldName)) {
      if (DpctGlobalInfo::getAssumedNDRangeDim() == 1) {
        auto TargetExpr = getTargetExpr();
        auto FD = getImmediateOuterFuncDecl(TargetExpr);
        auto DFI = DeviceFunctionDecl::LinkRedecls(FD);
        if (ME->getMemberDecl()->getName().str() == "__fetch_builtin_x") {
          auto Index = DpctGlobalInfo::getCudaKernelDimDFIIndexThenInc();
          DpctGlobalInfo::insertCudaKernelDimDFIMap(Index, DFI);
          addReplacement(ME, buildString(DpctGlobalInfo::getItem(ME), ".",
                                         ItemItr->second, "({{NEEDREPLACER",
                                         std::to_string(Index), "}})"));
          DpctGlobalInfo::updateSpellingLocDFIMaps(ME->getBeginLoc(), DFI);
        } else {
          DFI->getVarMap().Dim = 3;
          addReplacement(ME, buildString(DpctGlobalInfo::getItem(ME), ".",
                                         ItemItr->second, "(", FieldName, ")"));
        }
      } else {
        addReplacement(ME, buildString(DpctGlobalInfo::getItem(ME), ".",
                                       ItemItr->second, "(", FieldName, ")"));
      }
    }
  } else if (BaseType == "dim3") {
    if (ME->isArrow()) {
      addReplacement(ME->getBase(), "(" + getDrefName(ME->getBase()) + ")");
    }
    addReplacement(
        ME->getOperatorLoc(), ME->getMemberLoc(),
        MapNames::findReplacedName(MapNames::Dim3MemberNamesMap,
                                   ME->getMemberNameInfo().getAsString()));
  } else if (BaseType == "cudaDeviceProp") {
    auto MemberName = ME->getMemberNameInfo().getAsString();

    std::string ReplacementStr = MapNames::findReplacedName(DeviceInfoVarRule::PropNamesMap, MemberName);
    if (!ReplacementStr.empty()) {
      std::string TmplArg = "";
      if (MemberName == "maxGridSize" ||
          MemberName == "maxThreadsDim") {
        // Similar code in ASTTraversal.cpp
        TmplArg = "<int *>";
      }
      addReplacement(ME->getMemberLoc(), "get_" + ReplacementStr + TmplArg + "()");
      requestFeature(
          PropToGetFeatureMap.at(ME->getMemberNameInfo().getAsString()), ME);
    }
  } else if (BaseType == "textureReference") {
    std::string FieldName = ME->getMemberDecl()->getName().str();
    if (MapNames::replaceName(TextureRule::TextureMemberNames, FieldName)) {
      addReplacement(ME->getMemberLoc(), buildString("get_", FieldName, "()"));
      requestFeature(ImageWrapperBaseToGetFeatureMap.at(FieldName), ME);
    }
  } else if (MapNames::SupportedVectorTypes.find(BaseType) !=
             MapNames::SupportedVectorTypes.end()) {

    // Skip user-defined type.
    if (isTypeInAnalysisScope(ME->getBase()->getType().getTypePtr()))
      return;

    if (*BaseType.rbegin() == '1') {
      addReplacement(ME->getOperatorLoc(), ME->getEndLoc(), "");
    } else {
      std::string MemberName = ME->getMemberNameInfo().getAsString();
      if (MapNames::replaceName(MapNames::MemberNamesMap, MemberName)) {
        // Retrieve the correct location before addReplacement
        auto Loc =
            getLocInRange(ME->getMemberLoc(), getStmtExpansionSourceRange(ME));
        addReplacement(Loc, MemberName);
      }
    }
  }
  dispatch(ME->getBase());
  RefString.clear();
  RefString +=
      DpctGlobalInfo::getTypeName(ME->getBase()->getType().getCanonicalType()) +
      "." + ME->getMemberDecl()->getDeclName().getAsString();
}

void ExprAnalysis::analyzeExpr(const UnaryExprOrTypeTraitExpr *UETT) {
  if (UETT->getKind() == UnaryExprOrTypeTrait::UETT_SizeOf) {
    if (UETT->isArgumentType()) {
      analyzeType(UETT->getArgumentTypeInfo(), UETT);
    } else {
      analyzeExpr(UETT->getArgumentExpr());
    }
  }
}

inline void ExprAnalysis::analyzeExpr(const UnresolvedLookupExpr *ULE) {
  RefString.clear();
  llvm::raw_string_ostream OS(RefString);
  if (auto NNS = ULE->getQualifier()) {
    if (NNS->getKind() != clang::NestedNameSpecifier::SpecifierKind::Global) {
      NNS->print(OS, dpct::DpctGlobalInfo::getContext().getPrintingPolicy());
    }
  }
  ULE->getName().print(OS,
                       dpct::DpctGlobalInfo::getContext().getPrintingPolicy());
}

void ExprAnalysis::analyzeExpr(const ExplicitCastExpr *Cast) {
  if (Cast->getCastKind() == CastKind::CK_ConstructorConversion) {
    if (DpctGlobalInfo::getUnqualifiedTypeName(Cast->getTypeAsWritten()) ==
        "dim3")
      return dispatch(Cast->getSubExpr());
  }
  analyzeType(Cast->getTypeInfoAsWritten(), Cast);
  dispatch(Cast->getSubExprAsWritten());
}

// Precondition: CE != nullptr
void ExprAnalysis::analyzeExpr(const CallExpr *CE) {
  // To set the RefString
  dispatch(CE->getCallee());
  // If the callee requires rewrite, get the rewriter
  if (!CallExprRewriterFactoryBase::RewriterMap)
    return;
  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(RefString);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    auto Rewriter = Itr->second->create(CE);
    auto Result = Rewriter->rewrite();
    BlockLevelFormatFlag = Rewriter->getBlockLevelFormatFlag();

    if (Rewriter->isNoRewrite()) {
      // if the function is NoRewrite
      // Only change the function name in the spelling loc and
      // applyAllSubExprRepl
      if (Result.has_value()) {
        auto ResultStr = Result.value();
        addExtReplacement(std::make_shared<ExtReplacement>(
            SM, SM.getSpellingLoc(CE->getBeginLoc()), getCalleeName(CE).size(),
            ResultStr, nullptr));
        Rewriter->Analyzer.applyAllSubExprRepl();
      }
    } else {
      if (Result.has_value()) {
        auto ResultStr = Result.value();
        auto LocStr =
            getCombinedStrFromLoc(SM.getSpellingLoc(CE->getBeginLoc()));
        auto &FCIMMR =
            dpct::DpctGlobalInfo::getFunctionCallInMacroMigrateRecord();
        if (FCIMMR.find(LocStr) != FCIMMR.end() &&
            FCIMMR.find(LocStr)->second.compare(ResultStr) &&
            !isExprStraddle(CE)) {
          Rewriter->report(Diagnostics::CANNOT_UNIFY_FUNCTION_CALL_IN_MACOR,
                           false, RefString);
        }
        FCIMMR[LocStr] = ResultStr;
        // When migrating thrust API with usmnone and raw-ptr,
        // the CallExpr will be rewritten into an if-else stmt,
        // DPCT needs to remove the following semicolon.
        std::string EndBracket =
            "}" + std::string(
                      getNL(getStmtExpansionSourceRange(CE).getBegin(), SM));
        if (ResultStr.length() > EndBracket.length() &&
            ResultStr.substr(ResultStr.length() - EndBracket.length(),
                             EndBracket.length()) == EndBracket) {
          auto EndLoc = Lexer::getLocForEndOfToken(
              getStmtExpansionSourceRange(CE).getEnd(), 0, SM,
              DpctGlobalInfo::getContext().getLangOpts());
          Token Tok;
          Lexer::getRawToken(EndLoc, Tok, SM,
                             DpctGlobalInfo::getContext().getLangOpts(),
                             true);
          if (Tok.getKind() == tok::semi) {
            DpctGlobalInfo::getInstance().addReplacement(
                std::make_shared<ExtReplacement>(SM, EndLoc, 1, "", nullptr));
          }
        }

        addReplacement(CE, ResultStr);
        Rewriter->Analyzer.applyAllSubExprRepl();
        return;
      }
    }
  }
  // If the callee does not need rewrite, analyze the args
  for (auto Arg : CE->arguments())
    analyzeArgument(Arg);

  if (auto FD = DpctGlobalInfo::getParentFunction(CE)) {
    if (auto F = DpctGlobalInfo::getInstance().findDeviceFunctionDecl(FD)) {
      if (auto C = F->getFuncInfo()->findCallee(CE)) {
        auto Extra = C->getExtraArguments();
        if (Extra.empty())
          return;
        addReplacement(CE->getRParenLoc(), Extra);
      }
    }
  }
}

void ExprAnalysis::analyzeExpr(const CXXMemberCallExpr *CMCE) {
  auto PP = DpctGlobalInfo::getContext().getPrintingPolicy();
  PP.PrintCanonicalTypes = true;
  auto BaseType = CMCE->getObjectType().getUnqualifiedType().getAsString(PP);

  if (CMCE->getMethodDecl()->getIdentifier()) {
    auto MethodName = CMCE->getMethodDecl()->getNameAsString();

    if (CallExprRewriterFactoryBase::MethodRewriterMap) {
      auto Itr = CallExprRewriterFactoryBase::MethodRewriterMap->find(
          BaseType + "." + MethodName);
      if (Itr != CallExprRewriterFactoryBase::MethodRewriterMap->end()) {
        auto Rewriter = Itr->second->create(CMCE);
        auto Result = Rewriter->rewrite();
        if (Result.has_value()) {
          auto ResultStr = Result.value();
          addReplacement(CMCE, ResultStr);
          Rewriter->Analyzer.applyAllSubExprRepl();
          return;
        }
      }
    }
  }
  dispatch(CMCE->getCallee());
  for (auto Arg : CMCE->arguments())
    analyzeArgument(Arg);
}

void ExprAnalysis::analyzeExpr(const CXXBindTemporaryExpr *CBTE) {
  dispatch(CBTE->getSubExpr());
}

void ExprAnalysis::analyzeExpr(const CompoundStmt *CS) {
  for (auto It = CS->body_begin(); It != CS->body_end(); It++) {
    dispatch(*It);
  }
}

void ExprAnalysis::analyzeExpr(const ReturnStmt *RS) {
  dispatch(RS->getRetValue());
}


void ExprAnalysis::removeCUDADeviceAttr(const LambdaExpr *LE) {
  // E.g.,
  // my_kernel<<<1, 1>>>([=] __device__(int idx) { idx++; });
  // The "__device__" attribute need to be removed.
  if (const CXXRecordDecl *CRD = LE->getLambdaClass()) {
    if (const CXXMethodDecl *CMD = CRD->getLambdaCallOperator()) {
      if (CMD->hasAttr<CUDADeviceAttr>()) {
        addReplacement(CMD->getAttr<CUDADeviceAttr>()->getRange(), LE, "");
      }
    }
  }
}

void ExprAnalysis::analyzeExpr(const LambdaExpr *LE) {
  removeCUDADeviceAttr(LE);
  // TODO: Need to handle capture ([=] in lambda) if required in the future
  for (const auto &Param : LE->getCallOperator()->parameters()) {
    analyzeType(Param->getTypeSourceInfo()->getTypeLoc(), LE);
  }
  dispatch(LE->getBody());
}

void ExprAnalysis::analyzeExpr(const IfStmt *IS) {
  dispatch(IS->getCond());
  dispatch(IS->getThen());
  // "else if" will also be handled here as another ifstmt
  dispatch(IS->getElse());
}

void ExprAnalysis::analyzeExpr(const DeclStmt *DS) {
  for (const auto *Child : DS->children()) {
    dispatch(Child);
  }
}

void ExprAnalysis::analyzeType(TypeLoc TL, const Expr *CSCE) {
  SourceRange SR = TL.getSourceRange();
  std::string TyName;

  auto ETL = TL.getAs<ElaboratedTypeLoc>();
  if (ETL) {
    auto QualifierLoc = ETL.getQualifierLoc();
    TL = ETL.getNamedTypeLoc();
    TyName = getNestedNameSpecifierString(QualifierLoc);
    if (ETL.getTypePtr()->getKeyword() == ETK_Typename) {
      if (QualifierLoc)
        SR.setBegin(QualifierLoc.getBeginLoc());
      else
        SR = TL.getSourceRange();
    }
  }

#define TYPELOC_CAST(Target) static_cast<const Target &>(TL)
  switch (TL.getTypeLocClass()) {
  case TypeLoc::Qualified:
    return analyzeType(TL.getUnqualifiedLoc(), CSCE);
  case TypeLoc::LValueReference:
  case TypeLoc::RValueReference:
    return analyzeType(TYPELOC_CAST(ReferenceTypeLoc).getPointeeLoc(), CSCE);
  case TypeLoc::Pointer:
    return analyzeType(TYPELOC_CAST(PointerTypeLoc).getPointeeLoc(), CSCE);
  case TypeLoc::Typedef:
    TyName +=
        TYPELOC_CAST(TypedefTypeLoc).getTypedefNameDecl()->getName().str();
    break;
  case TypeLoc::Builtin:
  case TypeLoc::Record:
    TyName += DpctGlobalInfo::getTypeName(TL.getType());
    break;
  case TypeLoc::TemplateTypeParm:
    if (auto D = TYPELOC_CAST(TemplateTypeParmTypeLoc).getDecl()) {
      return addReplacement(TL.getBeginLoc(), TL.getEndLoc(), CSCE,
                            D->getIndex());
    } else {
      return;
    }
  case TypeLoc::TemplateSpecialization: {
    llvm::raw_string_ostream OS(TyName);
    auto &TSTL = TYPELOC_CAST(TemplateSpecializationTypeLoc);
    TSTL.getTypePtr()->getTemplateName().print(OS, Context.getPrintingPolicy());
    if (!TypeLocRewriterFactoryBase::TypeLocRewriterMap)
      return;
    auto Itr = TypeLocRewriterFactoryBase::TypeLocRewriterMap->find(OS.str());
    if (Itr != TypeLocRewriterFactoryBase::TypeLocRewriterMap->end()) {
      auto Rewriter = Itr->second->create(TSTL);
      auto Result = Rewriter->rewrite();
      if (Result.has_value()) {
        auto ResultStr = Result.value();
        // Since Parser splits ">>" or ">>>" to ">" when parse template
        // the SR.getEnd location might be a "scratch space" location.
        // Therfore, need to apply SM.getExpansionLoc before call addReplacement.
        addReplacement(SM.getExpansionLoc(SR.getBegin()),
                       SM.getExpansionLoc(SR.getEnd()), CSCE, ResultStr);
        return;
      }
    }
    if (OS.str() != "cub::WarpScan" && OS.str() != "cub::WarpReduce" &&
        OS.str() != "cub::BlockReduce" && OS.str() != "cub::BlockScan") {
      SR.setEnd(TSTL.getTemplateNameLoc());
    }
    analyzeTemplateSpecializationType(TSTL);
    break;
  }
  case TypeLoc::DependentTemplateSpecialization:
    analyzeTemplateSpecializationType(
        TYPELOC_CAST(DependentTemplateSpecializationTypeLoc));
    break;
  default:
    return;
  }

  auto Iter = MapNames::TypeNamesMap.find(TyName);
  if (Iter != MapNames::TypeNamesMap.end()) {
    HelperFeatureSet.insert(Iter->second->RequestFeature);
    requestHelperFeatureForTypeNames(TyName, SR.getBegin());
  } else {
    Iter = MapNames::CuDNNTypeNamesMap.find(TyName);
    if (Iter != MapNames::CuDNNTypeNamesMap.end()) {
      HelperFeatureSet.insert(Iter->second->RequestFeature);
      requestHelperFeatureForTypeNames(TyName, SR.getBegin());
    }
  }

  auto Range = getDefinitionRange(SR.getBegin(), SR.getEnd());
  if (MapNames::replaceName(MapNames::TypeNamesMap, TyName)) {
    addReplacement(Range.getBegin(), Range.getEnd(), CSCE, TyName);
  } else if (MapNames::replaceName(MapNames::CuDNNTypeNamesMap, TyName)) {
    addReplacement(Range.getBegin(), Range.getEnd(), CSCE, TyName);
  } else if (getFinalCastTypeNameStr(TyName) != TyName) {
    addReplacement(Range.getBegin(), Range.getEnd(), CSCE,
                   getFinalCastTypeNameStr(TyName));
  }
}

void ExprAnalysis::analyzeTemplateArgument(const TemplateArgumentLoc &TAL) {
  switch (TAL.getArgument().getKind()) {
  case TemplateArgument::Type:
    return analyzeType(TAL.getTypeSourceInfo());
  case TemplateArgument::Expression:
    return dispatch(TAL.getSourceExpression());
  case TemplateArgument::Integral:
    return dispatch(TAL.getSourceIntegralExpression());
  case TemplateArgument::Declaration:
    return dispatch(TAL.getSourceDeclExpression());
  default:
    break;
  }
}

void ExprAnalysis::applyAllSubExprRepl() {
  for (std::shared_ptr<ExtReplacement> Repl : SubExprRepl) {
    if (BlockLevelFormatFlag)
      Repl->setBlockLevelFormatFlag();

    DpctGlobalInfo::getInstance().addReplacement(Repl);
  }
}

const std::string &ArgumentAnalysis::getDefaultArgument(const Expr *E) {
  auto &Str = DefaultArgMap[E];
  if (Str.empty())
    Str = getStmtSpelling(E);
  return Str;
}

ManagedPointerAnalysis::ManagedPointerAnalysis(const CallExpr *C,
                                               bool IsAssigned) {
  Call = C;
  Assigned = IsAssigned;
  FirstArg = Call->getArg(0);
  SecondArg = Call->getArg(1);
  Pointer = nullptr;
  CPointer = nullptr;
  PointerScope = nullptr;
  CPointerScope = nullptr;
  UK = NoUse;
}
void ManagedPointerAnalysis::initAnalysisScope() {

  FirstArg = FirstArg->IgnoreParens()->IgnoreCasts()->IgnoreImplicitAsWritten();
  const clang::Expr *P = nullptr;
  if (auto UO = dyn_cast<UnaryOperator>(FirstArg)) {
    if (UO->getOpcode() == clang::UO_AddrOf) {
      P = UO->getSubExpr()->IgnoreImplicitAsWritten();
      FirstArg = P;
      NeedDerefOp = false;
    }
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(FirstArg)) {
    if (COCE->getOperator() == clang::OO_Amp && COCE->getNumArgs() == 1) {
      P = COCE->getArg(0)->IgnoreImplicitAsWritten();
      FirstArg = P;
      NeedDerefOp = false;
    }
  }
  // find managed pointer and pointer scope
  if (P) {
    if (auto PointerRef = dyn_cast<MemberExpr>(P)) {
      bool isInMethod = false;
      // ensure this pointer is class member
      for (auto C : PointerRef->children()) {
        if (C->getStmtClass() == Stmt::CXXThisExprClass)
          isInMethod = true;
      }
      if (isInMethod) {
        if (CPointer = dyn_cast<FieldDecl>(PointerRef->getMemberDecl())) {
          QualType QT = CPointer->getType();
          if (QT->isPointerType()) {
            QT = QT->getPointeeType();
            if (!QT->isPointerType() && !QT->isArrayType() &&
                !QT->isStructureOrClassType()) {
              if (CPointerScope =
                      dyn_cast<CXXRecordDecl>(CPointer->getParent()))
                Trackable = true;
            }
          }
        }
      }
    } else if (auto PointerRef = dyn_cast<DeclRefExpr>(P)) {
      if (Pointer = dyn_cast<VarDecl>(PointerRef->getDecl())) {
        QualType QT = Pointer->getType();
        if (QT->isPointerType()) {
          QT = QT->getPointeeType();
          if (!QT->isPointerType() && !QT->isArrayType() &&
              !QT->isStructureOrClassType()) {
            if (PointerScope = findImmediateBlock(Pointer))
              Trackable = true;
          }
        }
      }
    }
  }
}
void ManagedPointerAnalysis::RecursiveAnalyze() {
  initAnalysisScope();
  buildCallExprRepl();
  auto LocInfo = DpctGlobalInfo::getLocInfo(Call);
  if (Trackable) {
    if (PointerScope)
      dispatch(PointerScope);
    else if (CPointerScope)
      dispatch(CPointerScope);
    else {
      return;
    }
    if (ReAssigned) {
      DiagnosticsUtils::report(LocInfo.first, LocInfo.second,
                               Diagnostics::VIRTUAL_POINTER_HOST_ACCESS, true,
                               false, PointerName, PointerTempType);
    } else if (Transfered) {
      DiagnosticsUtils::report(LocInfo.first, LocInfo.second,
                               Diagnostics::VIRTUAL_POINTER_HOST_ACCESS, true,
                               false, PointerName, PointerTempType);
      addRepl();
    } else {
      addRepl();
    }
  } else {
    DiagnosticsUtils::report(LocInfo.first, LocInfo.second,
                             Diagnostics::VIRTUAL_POINTER_HOST_ACCESS, true,
                             false, PointerName, PointerTempType);
  }
}
void ManagedPointerAnalysis::buildCallExprRepl() {
  std::ostringstream OS;
  if (Assigned)
    OS << "(";
  auto E = FirstArg;
  bool NeedParen = false;
  if (NeedDerefOp) {
    OS << "*";
    Stmt::StmtClass SC = E->getStmtClass();
    if (!(SC == Stmt::DeclRefExprClass || SC == Stmt::MemberExprClass ||
          SC == Stmt::ParenExprClass || SC == Stmt::CallExprClass ||
          SC == Stmt::IntegerLiteralClass)) {
      NeedParen = true;
    }
  }
  QualType DerefQT = E->getType();
  if (NeedDerefOp)
    DerefQT = DerefQT->getPointeeType();

  ExprAnalysis EA(E);
  EA.analyze();
  if (NeedParen)
    OS << "(" << EA.getReplacedString() << ")";
  else
    OS << EA.getReplacedString();

  PointerName = OS.str();
  PointerCastType = DpctGlobalInfo::getReplacedTypeName(DerefQT);
  PointerTempType =
      DpctGlobalInfo::getReplacedTypeName(DerefQT->getPointeeType());
  PointerCastType = getFinalCastTypeNameStr(PointerCastType);
  PointerTempType = getFinalCastTypeNameStr(PointerTempType);
  if (PointerCastType != "NULL TYPE" && PointerCastType != "void *") {
    OS << " = (" << PointerCastType << ")";
  } else {
    OS << " = ";
  }
  requestFeature(HelperFeatureEnum::Memory_dpct_malloc, Call);
  requestFeature(HelperFeatureEnum::Memory_dpct_malloc_2d, Call);
  requestFeature(HelperFeatureEnum::Memory_dpct_malloc_3d, Call);
  OS << MapNames::getDpctNamespace() << "dpct_malloc(";
  ExprAnalysis ArgEA(SecondArg);
  ArgEA.analyze();
  OS << ArgEA.getReplacedString() << ")";
  if (Assigned) {
    OS << ", 0)";
    auto LocInfo = DpctGlobalInfo::getLocInfo(Call);
    DiagnosticsUtils::report(LocInfo.first, LocInfo.second,
                             Diagnostics::NOERROR_RETURN_COMMA_OP, false,
                             false);
  }
  addReplacement(Call->getBeginLoc(), Call->getEndLoc(), OS.str());
}
void ManagedPointerAnalysis::dispatch(const Decl *Decleration) {
  if (auto CXXRD = dyn_cast_or_null<CXXRecordDecl>(Decleration)) {
    analyzeExpr(CXXRD);
  }
}
void ManagedPointerAnalysis::dispatch(const Stmt *Expression) {
  if (!Expression)
    return;
  switch (Expression->getStmtClass()) {
    ANALYZE_EXPR(DeclRefExpr)
    ANALYZE_EXPR(MemberExpr)
    ANALYZE_EXPR(CallExpr)
    ANALYZE_EXPR(ArraySubscriptExpr)
    ANALYZE_EXPR(UnaryOperator)
    ANALYZE_EXPR(BinaryOperator)
    ANALYZE_EXPR(DeclStmt)
  default:
    return analyzeExpr(Expression);
  }
}
bool ManagedPointerAnalysis::isInCudaPath(const Decl *Decleration) {
  bool Result = false;
  std::string InFile = dpct::DpctGlobalInfo::getSourceManager()
                           .getFilename(Decleration->getLocation())
                           .str();
  bool InInstallPath = isChildOrSamePath(DpctInstallPath, InFile);
  bool InCudaPath = DpctGlobalInfo::isInCudaPath(Decleration->getLocation());
  if (InInstallPath || InCudaPath) {
    Result = true;
  }
  return Result;
}
void ManagedPointerAnalysis::addRepl() {
  if (!Repl.empty()) {
    for (const auto &R : Repl) {
      addReplacement(R.first.first, R.first.second, R.second);
    }
  }
}
void ManagedPointerAnalysis::analyzeExpr(const Stmt *S) {
  UseKind UK_TEMP = NoUse;
  for (auto SubS : S->children()) {
    if (SubS && dyn_cast<Stmt>(SubS)) {
      UK = NoUse;
      dispatch(SubS);
      UK_TEMP = UK > UK_TEMP ? UK : UK_TEMP;
    }
  }
  UK = UK_TEMP;
}
void ManagedPointerAnalysis::analyzeExpr(const CXXRecordDecl *CRD) {
  if (auto CXXDecl = dyn_cast<CXXRecordDecl>(CRD)) {
    for (auto *MT : CXXDecl->methods()) {
      if (MT->hasBody()) {
        UK = NoUse;
        dispatch(MT->getBody());
      }
    }
  }
}
void ManagedPointerAnalysis::analyzeExpr(const DeclRefExpr *DRE) {
  if (DRE->getDecl() != Pointer)
    return;
  UK = Literal;
}
void ManagedPointerAnalysis::analyzeExpr(const MemberExpr *ME) {
  if (ME->getMemberDecl() != CPointer)
    return;
  UK = Literal;
}
void ManagedPointerAnalysis::analyzeExpr(const DeclStmt *DS) {
  UseKind UK_TEMP = NoUse;
  for (auto CH : DS->decls()) {
    UK = NoUse;
    if (auto Var = dyn_cast<VarDecl>(CH)) {
      if (Var->hasInit()) {
        dispatch(Var->getInit());
        UK_TEMP = UK > UK_TEMP ? UK : UK_TEMP;
      }
    }
  }
  UK = UK_TEMP;
}
void ManagedPointerAnalysis::analyzeExpr(const CallExpr *CE) {
  UseKind UKCALL = NoUse;
  if (CE == Call)
    return;
  bool InCuda = true;
  std::string APIName;
  const FunctionDecl *CalleeDecl = CE->getDirectCallee();
  if (CalleeDecl) {
    InCuda = isInCudaPath(CalleeDecl);
    APIName = CalleeDecl->getNameAsString();
  } else {
    if (auto ULExpr = dyn_cast<UnresolvedLookupExpr>(*CE->child_begin())) {
      for (auto D = ULExpr->decls_begin(); D != ULExpr->decls_end(); D++) {
        auto Decl = D.getDecl();
        InCuda = InCuda & isInCudaPath(Decl);
        APIName = Decl->getNameAsString();
      }
    } else {
      return;
    }
  };
  for (auto Arg : CE->arguments()) {
    UK = NoUse;
    dispatch(Arg);
    UKCALL = UK > UKCALL ? UK : UKCALL;
  }
  UK = UKCALL;
  if (InCuda) {
    if (UK == Address) {
      if (APIName == "cudaMalloc" || APIName == "cudaMallocManaged" ||
          APIName == "cudaMallocHost") {
        ReAssigned = true;
      }
    } else {
      UK = NoUse;
    }
  } else {
    if (UK == Literal || UK == Address) {
      Transfered = true;
    }
  }
}
void ManagedPointerAnalysis::analyzeExpr(const UnaryOperator *UO) {
  UK = NoUse;
  if (UO->getOpcode() == UnaryOperatorKind::UO_Deref) {
    auto SubE = UO->getSubExpr();
    dispatch(SubE);
    if (UK == Literal) {
      UK = Reference;
      ExprAnalysis EA(SubE);
      EA.analyze();
      std::string Rep = EA.getReplacedString();
      requestFeature(HelperFeatureEnum::Memory_get_host_ptr, Call);
      if (SubE->IgnoreImplicitAsWritten()->getStmtClass() ==
          Stmt::ParenExprClass) {
        Repl.push_back(
            {{SubE->getBeginLoc(), SubE->getEndLoc()},
             std::string(MapNames::getDpctNamespace() + "get_host_ptr<" +
                         PointerTempType + ">" + Rep)});
      } else {
        Repl.push_back(
            {{SubE->getBeginLoc(), SubE->getEndLoc()},
             std::string(MapNames::getDpctNamespace() + "get_host_ptr<" +
                         PointerTempType + ">(" + Rep + ")")});
      }
    }
  } else if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf) {
    dispatch(UO->getSubExpr());
    if (UK == Literal)
      UK = Address;
  } else {
    dispatch(UO->getSubExpr());
  };
}
void ManagedPointerAnalysis::analyzeExpr(const BinaryOperator *BO) {

  UK = NoUse;
  dispatch(BO->getLHS());
  UseKind UKLHS = UK;

  UK = NoUse;
  dispatch(BO->getRHS());
  UseKind UKRHS = UK;

  UK = UKLHS > UKRHS ? UKLHS : UKRHS;
  if (BO->getOpcode() == BinaryOperatorKind::BO_Assign) {
    if (UKRHS == Literal) {
      Transfered = true;
    }
    if (UKLHS == Literal || UKRHS == Address) {
      ReAssigned = true;
    }
  }
}
void ManagedPointerAnalysis::analyzeExpr(const ArraySubscriptExpr *ASE) {
  UK = NoUse;
  auto Base = ASE->getBase();
  dispatch(Base);
  if (UK == Literal) {
    UK = Reference;
    Repl.push_back({{Base->getBeginLoc(), Base->getEndLoc()},
                    std::string(MapNames::getDpctNamespace() + "get_host_ptr<" +
                                PointerTempType + ">(" + PointerName + ")")});
    requestFeature(HelperFeatureEnum::Memory_get_host_ptr, Call);
  }
}
void KernelArgumentAnalysis::dispatch(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
    ANALYZE_EXPR(DeclRefExpr)
    ANALYZE_EXPR(MemberExpr)
    ANALYZE_EXPR(CXXMemberCallExpr)
    ANALYZE_EXPR(CallExpr)
    ANALYZE_EXPR(ArraySubscriptExpr)
    ANALYZE_EXPR(UnaryOperator)
    ANALYZE_EXPR(CXXDependentScopeMemberExpr)
    ANALYZE_EXPR(MaterializeTemporaryExpr)
    ANALYZE_EXPR(LambdaExpr)
  default:
    return ExprAnalysis::dispatch(Expression);
  }
}

int64_t
KernelConfigAnalysis::calculateWorkgroupSize(const CXXConstructExpr *Ctor) {
  int64_t Size = 1;
  auto Num = Ctor->getNumArgs();
  for (size_t i = 0; i < Num; ++i) {
    if (Ctor->getArg(i)->isDefaultArgument()) {
      if (i == 0) {
        SizeOfHighestDimension = 1;
      }
      return Size;
    }

    Expr::EvalResult ER;
    if (!Ctor->getArg(i)->isValueDependent() &&
        Ctor->getArg(i)->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (i == 0) {
        SizeOfHighestDimension = Value;
      }
      Size = Size * Value;
    } else {
      // Not all args can be evaluated, so return a value larger than 256 to
      // emit warning.
      return 265 + 1;
    }
  }
  return Size;
}

void KernelArgumentAnalysis::analyzeExpr(const MaterializeTemporaryExpr *MTE) {
  KernelArgumentAnalysis::dispatch(MTE->getSubExpr());
}

void KernelArgumentAnalysis::analyzeExpr(
    const CXXDependentScopeMemberExpr *Arg) {
  if (Arg->isImplicitAccess()) {
    IsRedeclareRequired = true;
  } else {
    if (Arg->isArrow())
      IsRedeclareRequired = true;
    KernelArgumentAnalysis::dispatch(Arg->getBase());
  }
}

void KernelArgumentAnalysis::analyzeExpr(const DeclRefExpr *DRE) {
  if (DRE->getType()->isReferenceType()) {
    IsRedeclareRequired = true;
  } else if (!isLexicallyInLocalScope(DRE->getDecl())) {
    IsRedeclareRequired = true;
  } else if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    bool PreviousFlag = IsRedeclareRequired;
    if (VD->getStorageClass() == SC_Static) {
      IsRedeclareRequired = true;
      // exclude const variable with zero-init and const-init
      if (VD->getType().isConstQualified()) {
        if (VD->hasInit() && VD->hasICEInitializer(Context)) {
          IsRedeclareRequired = PreviousFlag;
        }
      }
    }
  }

  // The VarDecl in MemVarInfo are matched in MemVarRule, which only matches
  // variables on device. They are migrated to objects, so need add get_ptr() by
  // setting IsDefinedOnDevice flag.
  if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    if (auto Var = DpctGlobalInfo::getInstance().findMemVarInfo(VD)) {
      IsDefinedOnDevice = true;
      IsRedeclareRequired = true;
      if (!IsAddrOf && !VD->getType()->isArrayType()) {
        addReplacement(Lexer::getLocForEndOfToken(
                           DRE->getEndLoc(), 0, SM,
                           DpctGlobalInfo::getContext().getLangOpts()),
                       DRE->getEndLoc(), "[0]");
      } else {
        requestFeature(HelperFeatureEnum::Memory_device_memory_get_ptr,
                       DRE->getEndLoc());
        addReplacement(Lexer::getLocForEndOfToken(
                           DRE->getEndLoc(), 0, SM,
                           DpctGlobalInfo::getContext().getLangOpts()),
                       DRE->getEndLoc(), ".get_ptr()");
      }
    }
  }

  if (checkPointerInStructRecursively(DRE))
    IsDoublePointer = true;

  Base::analyzeExpr(DRE);
}

void KernelArgumentAnalysis::analyzeExpr(const MemberExpr *ME) {
  if (ME->getBase()->getType()->isDependentType()) {
    IsRedeclareRequired = true;
  }

  if (ME->getBase()->isImplicitCXXThis()) {
    IsRedeclareRequired = true;
  }

  if (ME->isArrow()) {
    IsRedeclareRequired = true;
  }

  if (auto RD = ME->getBase()->getType()->getAsCXXRecordDecl()) {
    if (!RD->isStandardLayout()) {
      IsRedeclareRequired = true;
    }
  }
  Base::analyzeExpr(ME);
}

void KernelArgumentAnalysis::analyzeExpr(const LambdaExpr *LE) {
  Base::analyzeExpr(LE);
  // Lambda function can be passed to kernel function directly.
  // So, not need to redeclare a variable for lambda function passed to kernel
  // function
  IsRedeclareRequired = false;
}


void KernelArgumentAnalysis::analyzeExpr(const UnaryOperator *UO) {
  if (UO->getOpcode() == UO_Deref) {
    IsRedeclareRequired = true;
    return;
  }
  if (UO->getOpcode() == UO_AddrOf) {
    IsAddrOf = true;
  }
  dispatch(UO->getSubExpr());
  /// If subexpr is variable defined on device, remove operator '&'.
  if (IsAddrOf && IsDefinedOnDevice) {
    addReplacement(UO->getOperatorLoc(), "");
  }
  /// Clear flag 'IsDefinedOnDevice' and 'IsAddrOf'
  IsDefinedOnDevice = false;
  IsAddrOf = false;
}

void KernelArgumentAnalysis::analyze(const Expr *Expression) {
  IsPointer = Expression->getType()->isPointerType();
  if (IsPointer) {
    IsDoublePointer = Expression->getType()->getPointeeType()->isPointerType();
  }
  TryGetBuffer = IsPointer &&
                 DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
                 !isNullPtr(Expression);
  IsRedeclareRequired = false;
  ArgumentAnalysis::analyze(Expression);
}

bool KernelArgumentAnalysis::isNullPtr(const Expr *E) {
  E = E->IgnoreCasts();
  if (isa<GNUNullExpr>(E))
    return true;
  if (isa<CXXNullPtrLiteralExpr>(E))
    return true;
  if (auto IL = dyn_cast<IntegerLiteral>(E)) {
    if (!IL->getValue().getZExtValue())
      return true;
  }
  return false;
}

void FunctorAnalysis::dispatch(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
    ANALYZE_EXPR(CXXTemporaryObjectExpr)
    ANALYZE_EXPR(DeclRefExpr)
    ANALYZE_EXPR(CXXConstructExpr)
    ANALYZE_EXPR(CXXFunctionalCastExpr)
  default:
    return ArgumentAnalysis::dispatch(Expression);
  }
}

void FunctorAnalysis::addConstQuailfier(const CXXRecordDecl *CRD) {
  for (const auto &D : CRD->decls()) {
    const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D);
    if (!Method) {
      if (const FunctionTemplateDecl *FTD =
              dyn_cast<FunctionTemplateDecl>(D)) {
        if (const CXXMethodDecl *CMD =
                dyn_cast_or_null<CXXMethodDecl>(FTD->getAsFunction())) {
          Method = CMD;
        }
      }
    }
    if (!Method) {
      continue;
    }
    if (Method->getOverloadedOperator() == OverloadedOperatorKind::OO_Call &&
        !Method->isConst()) {
      for (const auto &FD : Method->redecls()) {
        SourceLocation InsertLoc = FD->getFunctionTypeLoc().getRParenLoc();
        // Get the location after ')'
        if (InsertLoc.isMacroID()) {
          InsertLoc = SM.getSpellingLoc(InsertLoc).getLocWithOffset(1);
        } else {
          InsertLoc = InsertLoc.getLocWithOffset(1);
        }
        DpctGlobalInfo::getInstance().addReplacement(
            std::make_shared<ExtReplacement>(SM, InsertLoc, 0, " const",
                                             nullptr));
      }
      break;
    }
  }
}

void FunctorAnalysis::analyzeExpr(const CXXFunctionalCastExpr *CFCE) {
  dispatch(CFCE->getSubExpr());
}

void FunctorAnalysis::analyzeExpr(const CXXTemporaryObjectExpr *CTOE) {
  const CXXConstructorDecl *CCD = CTOE->getConstructor();
  if (!CCD)
    return;
  const CXXRecordDecl *CRD = CCD->getParent();
  if (!CRD)
    return;
  if (DpctGlobalInfo::isInAnalysisScope(CRD->getBeginLoc())) {
    addConstQuailfier(CRD);
  }
  Base::analyzeExpr(CTOE);
}

void FunctorAnalysis::analyzeExpr(const DeclRefExpr *DRE) {
  // Process thrust placeholder
  auto TypeStr = DRE->getType().getAsString();
  static const std::string PlaceHolderTypeStr =
      "const thrust::detail::functional::placeholder<";
  if (TypeStr.find(PlaceHolderTypeStr) == 0) {
    unsigned PlaceholderNum = (TypeStr[PlaceHolderTypeStr.length()] - '0') + 1;
    if (PlaceholderNum > PlaceholderCount)
      PlaceholderCount = PlaceholderNum;
    addReplacement(DRE, std::string("_") + std::to_string(PlaceholderNum));
  }

  // Process functor's quailfier
  const Type *Tp = DRE->getType().getTypePtr();
  if (!Tp)
    return;
  const CXXRecordDecl *CRD = Tp->getAsCXXRecordDecl();
  if (!CRD)
    return;
  if (DpctGlobalInfo::isInAnalysisScope(CRD->getBeginLoc())) {
    addConstQuailfier(CRD);
  }
  ArgumentAnalysis::analyzeExpr(DRE);
}

void FunctorAnalysis::analyzeExpr(const CXXConstructExpr *CCE) {
  const CXXConstructorDecl *CCD = CCE->getConstructor();
  if (!CCD)
    return;
  const CXXRecordDecl *CRD = CCD->getParent();
  if (!CRD)
    return;
  if (DpctGlobalInfo::isInAnalysisScope(CRD->getBeginLoc())) {
    addConstQuailfier(CRD);
  }
  Base::analyzeExpr(CCE);
}

void FunctorAnalysis::analyze(const Expr *Expression) {
  ArgumentAnalysis::initArgumentExpr(Expression);
  ArgumentAnalysis::analyze();
  std::string LambdaPrefix;
  std::string LambdaPostfix;
  if (PlaceholderCount) {
    LambdaPrefix = "[=](";
    for (unsigned i = 1; i <= PlaceholderCount; ++i) {
      if (i > 1)
        LambdaPrefix += ", ";
      LambdaPrefix += "auto _" + std::to_string(i);
    }
    LambdaPrefix += "){ return ";
    LambdaPostfix = "; }";
  }
  std::string R = LambdaPrefix + getReplacedString() + LambdaPostfix;
  addReplacement(Expression, R);
}

void KernelConfigAnalysis::dispatch(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
    ANALYZE_EXPR(CXXConstructExpr)
    ANALYZE_EXPR(CXXTemporaryObjectExpr)
    ANALYZE_EXPR(CXXDependentScopeMemberExpr)
  default:
    return ArgumentAnalysis::dispatch(Expression);
  }
}

void KernelConfigAnalysis::analyzeExpr(
    const CXXDependentScopeMemberExpr *CDSME) {
  if (ArgIndex == 1) {
    NeedEmitWGSizeWarning = true;
  }

  if (ArgIndex < 2) {
    std::string CDSMEString;
    llvm::raw_string_ostream OS(CDSMEString);
    DpctGlobalInfo::printCtadClass(OS, MapNames::getClNamespace() + "range", 3);
    OS << "(1, 1, " << ExprAnalysis::ref(CDSME) << ")";
    OS.flush();
    addReplacement(CDSME, CDSMEString);
  }
}

bool KernelConfigAnalysis::isOneDimensionConfigArg(
    const CXXConstructExpr *Ctor) {
  if (Ctor->getNumArgs() == 1) {
    // E.g., copy constructor: dim3 a(1,2,3); k<<<dim3(a), 1>>>();
    return false;
  }

  if (Ctor->getNumArgs() == 3) {
    Expr::EvalResult ER1, ER2;
    if (!Ctor->getArg(1)->isValueDependent() &&
        Ctor->getArg(1)->EvaluateAsInt(ER1, DpctGlobalInfo::getContext()) &&
        !Ctor->getArg(2)->isValueDependent() &&
        Ctor->getArg(2)->EvaluateAsInt(ER2, DpctGlobalInfo::getContext())) {
      if (ER1.Val.getInt().getZExtValue() == 1 &&
          ER2.Val.getInt().getZExtValue() == 1)
        return true;
    }
    return false;
  }
  return false;
}

void KernelConfigAnalysis::analyzeExpr(const CXXConstructExpr *Ctor) {
  if (Ctor->getConstructor()->getDeclName().getAsString() == "dim3") {
    if (ArgIndex == 1) {
      if (calculateWorkgroupSize(Ctor) <= 256)
        NeedEmitWGSizeWarning = false;
      if (IsTryToUseOneDimension)
        Dim = isOneDimensionConfigArg(Ctor) ? 1 : 3;
    } else if (ArgIndex == 0) {
      if (IsTryToUseOneDimension)
        Dim = isOneDimensionConfigArg(Ctor) ? 1 : 3;
    }

    std::string CtorString;
    llvm::raw_string_ostream OS(CtorString);
    if (IsTryToUseOneDimension && Dim == 1) {
      DpctGlobalInfo::printCtadClass(OS, MapNames::getClNamespace() + "range",
                                     1)
          << "(";
      auto Args = getCtorArgs(Ctor);
      if (Ctor->getNumArgs() > 0) {
        OS << Args[0] << ", ";
      } else {
        llvm_unreachable("Ctor of the kernel config hasn't any argument!");
      }
    } else {
      DpctGlobalInfo::printCtadClass(OS, MapNames::getClNamespace() + "range",
                                     3)
          << "(";
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
    }

    OS.flush();
    // Special handling for implicit ctor.
    // #define GET_BLOCKS(a) a
    // foo_kernel<<<GET_BLOCKS(1), 2, 0>>>();
    // Result if using SM.getExpansionRange:
    //   sycl::range<3>(1, 1, GET_BLOCKS(1)) in kernel
    // Result if using addReplacement(E):
    //   #define GET_BLOCKS(a) sycl::range<3>(1, 1, a)
    //   GET_BLOCKS(1) in kernel
    if (Ctor->getParenOrBraceRange().isInvalid() && isOuterMostMacro(Ctor)) {
      return addReplacement(
          SM.getExpansionRange(Ctor->getBeginLoc()).getBegin(),
          SM.getExpansionRange(Ctor->getEndLoc()).getEnd(),
          CtorString.replace(CtorString.length() - 2, 2, ")"));
    }
    return addReplacement(Ctor,
                          CtorString.replace(CtorString.length() - 2, 2, ")"));
  }
  return ArgumentAnalysis::analyzeExpr(Ctor);
}

std::vector<std::string>
KernelConfigAnalysis::getCtorArgs(const CXXConstructExpr *Ctor) {
  std::vector<std::string> Args;
  ArgumentAnalysis A(IsInMacroDefine);
  A.setCallSpelling(CallSpellingBegin, CallSpellingEnd);
  for (auto Arg : Ctor->arguments())
    Args.emplace_back(getCtorArg(A, Arg));
  return Args;
}

void KernelConfigAnalysis::analyze(const Expr *E, unsigned int Idx,
                                   bool ReverseIfNeed) {
  ArgIndex = Idx;
  MustDim3 = ArgIndex < 2;

  if (IsInMacroDefine && SM.isMacroArgExpansion(E->getBeginLoc())) {
    Reversed = false;
    DirectRef = true;
    if (ArgIndex == 3 && isPredefinedStreamHandle(E)) {
      addReplacement("0");
      return;
    }
  }

  DoReverse = ReverseIfNeed;
  if (ArgIndex == 3 && isPredefinedStreamHandle(E)) {
    addReplacement("0");
    return;
  }
  ArgumentAnalysis::analyze(E);

  if (getTargetExpr()->IgnoreImplicit()->getStmtClass() ==
          Stmt::DeclRefExprClass ||
      getTargetExpr()->IgnoreImpCasts()->getStmtClass() ==
          Stmt::MemberExprClass ||
      getTargetExpr()->IgnoreImpCasts()->getStmtClass() ==
          Stmt::IntegerLiteralClass) {
    if (MustDim3 && getTargetExpr()->getType()->isIntegralType(
                        DpctGlobalInfo::getContext())) {
      if (IsTryToUseOneDimension) {
        Dim = 1;
        addReplacement(buildString(DpctGlobalInfo::getCtadClass(
                                       MapNames::getClNamespace() + "range", 1),
                                   "(", getReplacedString(), ")"));
      } else {
        addReplacement(buildString(DpctGlobalInfo::getCtadClass(
                                       MapNames::getClNamespace() + "range", 3),
                                   "(1, 1, ", getReplacedString(), ")"));
      }

      Reversed = true;
      return;
    }

    DirectRef = true;
  }
}

std::string ArgumentAnalysis::getRewriteString() {
  // Find rewrite range
  auto RewriteRange = getLocInCallSpelling(getTargetExpr());
  auto RewriteRangeBegin = RewriteRange.first;
  auto RewriteRangeEnd = RewriteRange.second;
  size_t RewriteLength = SM.getCharacterData(RewriteRangeEnd) -
                         SM.getCharacterData(RewriteRangeBegin);
  // Get original string
  auto DL = SM.getDecomposedLoc(RewriteRangeBegin);
  std::string OriginalStr =
      std::string(SM.getBufferData(DL.first).substr(DL.second, RewriteLength));

  StringReplacements SRs;
  SRs.init(std::move(OriginalStr));

  for (std::shared_ptr<ExtReplacement> SubRepl : SubExprRepl) {
    if (isInRange(RewriteRangeBegin, RewriteRangeEnd, SubRepl->getFilePath(),
                  SubRepl->getOffset()) &&
        isInRange(RewriteRangeBegin, RewriteRangeEnd, SubRepl->getFilePath(),
                  SubRepl->getOffset() + SubRepl->getLength())) {
      SRs.addStringReplacement(
          SubRepl->getOffset() - SM.getDecomposedLoc(RewriteRangeBegin).second,
          SubRepl->getLength(), SubRepl->getReplacementText().str());
    }
  }

  return SRs.getReplacedString();
}

std::pair<SourceLocation, SourceLocation>
ArgumentAnalysis::getLocInCallSpelling(const Expr *E) {
  return getRangeInRange(E, CallSpellingBegin, CallSpellingEnd);
}

void SideEffectsAnalysis::dispatch(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
    ANALYZE_EXPR(BinaryOperator)
  // The following types of expressions don't have side effects
  case Stmt::IntegerLiteralClass:
  case Stmt::FloatingLiteralClass:
  case Stmt::StringLiteralClass:
  case Stmt::DeclRefExprClass:
  case Stmt::ArraySubscriptExprClass:
  case Stmt::CStyleCastExprClass:
  case Stmt::CXXStaticCastExprClass:
  case Stmt::ImplicitCastExprClass:
  case Stmt::ParenExprClass:
  case Stmt::ConditionalOperatorClass:
  case Stmt::MemberExprClass:
    break;
  default:
    HasSideEffects = true;
  }
  return ExprAnalysis::dispatch(Expression);
}

} // namespace dpct
} // namespace clang
