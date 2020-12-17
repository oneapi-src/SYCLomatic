//===--- LibraryAPIMigration.cpp -------------------------*- C++ -*---===//
//
// Copyright (C) 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "LibraryAPIMigration.h"
#include "AnalysisInfo.h"
#include "MapNames.h"
#include "Utility.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/LangOptions.h"

namespace clang {
namespace dpct {

/// Set the prec and domain in the FFTDescriptorTypeInfo of the declaration of
/// \p DescIdx
void FFTFunctionCallBuilder::addDescriptorTypeInfo(
    std::string PrecAndDomainStr) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  const DeclaratorDecl *HandleVar = getHandleVar(TheCallExpr->getArg(0));
  if (!HandleVar)
    return;

  SourceLocation TypeBeginLoc =
      HandleVar->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
  if (TypeBeginLoc.isMacroID()) {
    auto SpellingLocation = SM.getSpellingLoc(TypeBeginLoc);
    if (DpctGlobalInfo::replaceMacroName(SpellingLocation)) {
      TypeBeginLoc = SM.getExpansionLoc(TypeBeginLoc);
    } else {
      TypeBeginLoc = SpellingLocation;
    }
  }
  unsigned int TypeLength = Lexer::MeasureTokenLength(
      TypeBeginLoc, SM, DpctGlobalInfo::getContext().getLangOpts());

  auto LocInfo = DpctGlobalInfo::getLocInfo(TypeBeginLoc);
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(LocInfo.first);
  auto &M = FileInfo->getFFTDescriptorTypeMap();
  auto Iter = M.find(LocInfo.second);
  if (Iter == M.end()) {
    FFTDescriptorTypeInfo FDTI(TypeLength);
    FDTI.PrecAndDom = PrecAndDomainStr;
    M.insert(std::make_pair(LocInfo.second, FDTI));
  } else if (Iter->second.IsValid) {
    if (Iter->second.PrecAndDom.empty()) {
      Iter->second.PrecAndDom = PrecAndDomainStr;
    } else if (Iter->second.PrecAndDom != PrecAndDomainStr) {
      Iter->second.IsValid = false;
    }
  }
}

/// Evaluate the value of \p PrecDomainIdx then return the string.
std::string
FFTFunctionCallBuilder::getPrecAndDomainStr(unsigned int PrecDomainIdx) {
  std::string PrecAndDomain;
  Expr::EvalResult ER;
  if (!TheCallExpr->getArg(PrecDomainIdx)->isValueDependent() &&
      TheCallExpr->getArg(PrecDomainIdx)
          ->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
    int64_t Value = ER.Val.getInt().getExtValue();
    PrecAndDomain = getPrecAndDomainStrFromValue(Value);
  }
  return PrecAndDomain;
}

FFTTypeEnum FFTFunctionCallBuilder::getFFTType(unsigned int PrecDomainIdx) {
  Expr::EvalResult ER;
  if (!TheCallExpr->getArg(PrecDomainIdx)->isValueDependent() &&
      TheCallExpr->getArg(PrecDomainIdx)
          ->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
    int64_t Value = ER.Val.getInt().getExtValue();
    return getFFTTypeFromValue(Value);
  }
  return FFTTypeEnum::Unknown;
}

/// Add buffer decl.
void FFTFunctionCallBuilder::updateBufferArgs(unsigned int Idx,
                                              const std::string &TypeStr,
                                              std::string PointerName) {
  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
    std::string BufferDecl;
    if (PointerName.empty()) {
      PointerName = dpct::ExprAnalysis::ref(TheCallExpr->getArg(Idx));
      ArgsList[Idx] =
          getTempNameForExpr(TheCallExpr->getArg(Idx), true, true) + "buf_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
    } else {
      ArgsList[Idx] =
          PointerName + "_buf_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
    }

    BufferDecl = "auto " + ArgsList[Idx] + " = dpct::get_buffer<" + TypeStr +
                 ">(" + PointerName + ");";
    PrefixStmts.emplace_back(BufferDecl);
  } else {
    if ((Idx == 1 && (FuncName[9] == 'C' || FuncName[9] == 'Z')) ||
        (Idx == 2 && (FuncName[11] == 'C' || FuncName[11] == 'Z')))
      ArgsList[Idx] = "(" + TypeStr + "*)" + ArgsList[Idx];
  }
}

/// build exec API info
void FFTFunctionCallBuilder::assembleExecCallExpr(int64_t Dir) {
  auto getRealDomainType = [=](const char C) -> std::string {
    if (C == 'Z' || C == 'D')
      return "double";
    else
      return "float";
  };

  auto getType = [=](const char C) -> std::string {
    if (C == 'Z')
      return MapNames::getClNamespace() + "::double2";
    else if (C == 'C')
      return MapNames::getClNamespace() + "::float2";
    else if (C == 'D')
      return "double";
    else
      return "float";
  };

  std::string ComputeAPI;
  if (Dir == -1) {
    ComputeAPI = "oneapi::mkl::dft::compute_forward";
  } else if (Dir == 1) {
    ComputeAPI = "oneapi::mkl::dft::compute_backward";
  } else {
    ComputeAPI = "oneapi::mkl::dft::dpct_placeholder/*fix the computational "
                 "function manaully*/";
  }

  std::string OriginalInputPtr = ArgsList[1];
  std::string OriginalOutputPtr = ArgsList[2];

  if (Flags.IsFunctionPointer) {
    auto LocInfo =
        DpctGlobalInfo::getLocInfo(Locations.FuncPtrDeclHandleTypeBegin);
    auto FileInfo = DpctGlobalInfo::getInstance().insertFile(LocInfo.first);
    auto &M = FileInfo->getFFTDescriptorTypeMap();
    auto Iter = M.find(LocInfo.second);
    if (Iter == M.end()) {
      FFTDescriptorTypeInfo FDTI(0);
      FDTI.SkipGeneration = true;
      M.insert(std::make_pair(LocInfo.second, FDTI));
    } else {
      Iter->second.SkipGeneration = true;
    }

    std::string WrapperBegin =
        "auto " + FuncPtrName +
        " = [](std::shared_ptr<oneapi::mkl::dft::descriptor<" +
        getPrecAndDomainStrFromExecFuncName(FuncName) + ">> desc, " +
        getType(FuncName[9]) + " *in_data, " + getType(FuncName[11]) +
        " *out_data";
    if (FuncName[9] == FuncName[11])
      WrapperBegin = WrapperBegin + ", int dir";
    WrapperBegin = WrapperBegin + "){";
    PrefixStmts.emplace_back(std::move(WrapperBegin));

    updateBufferArgs(1, getRealDomainType(FuncName[9]), OriginalInputPtr);
  } else {
    updateBufferArgs(1, getRealDomainType(FuncName[9]));
  }

  PrefixStmts.emplace_back("if ((void *)" + OriginalInputPtr + " == (void *)" +
                           OriginalOutputPtr + ") {");
  if (Flags.IsFunctionPointer) {
    PrefixStmts.emplace_back(ComputeAPI + "(*" + ArgsList[0] + ", " +
                             ArgsList[1] + ");");
  } else {
    PrefixStmts.emplace_back(ComputeAPI + "(" +
                             getDrefName(TheCallExpr->getArg(0)) + ", " +
                             ArgsList[1] + ");");
  }

  PrefixStmts.emplace_back("} else {");

  if (Flags.IsFunctionPointer) {
    updateBufferArgs(2, getRealDomainType(FuncName[11]), OriginalOutputPtr);
    CallExprRepl = ComputeAPI + "(*" + ArgsList[0] + ", " + ArgsList[1] + ", " +
                   ArgsList[2] + ")";
  } else {
    updateBufferArgs(2, getRealDomainType(FuncName[11]));
    CallExprRepl = ComputeAPI + "(" + getDrefName(TheCallExpr->getArg(0)) +
                   ", " + ArgsList[1] + ", " + ArgsList[2] + ")";
  }
  SuffixStmts.emplace_back("}");
  if (Flags.IsFunctionPointer) {
    SuffixStmts.emplace_back("}");
  }
}

std::string FFTFunctionCallBuilder::getPrePrefixString() {
  if (!PrePrefixStmt.empty())
    return PrePrefixStmt + getNL() + IndentStr;
  else
    return "";
}

std::string FFTFunctionCallBuilder::getPrefixString() {
  std::string Res;
  for (const auto &Stmt : PrefixStmts)
    Res = Res + Stmt + getNL() + IndentStr;

  return Res;
}

std::string FFTFunctionCallBuilder::getSuffixString() {
  std::string Res;
  for (const auto &Stmt : SuffixStmts)
    Res = Res + getNL() + IndentStr + Stmt;

  return Res;
}

std::string FFTFunctionCallBuilder::getCallExprReplString() {
  return CallExprRepl;
}

/// if the cufftResult variable is inited by exec API, tool need move the
/// declaration out of braces
bool FFTFunctionCallBuilder::moveDeclOutOfBracesIfNeeds(
    const LibraryMigrationFlags Flags, SourceLocation &TypeBegin,
    int &TypeLength) {
  if (Flags.IsFunctionPointer)
    return false;

  auto &SM = DpctGlobalInfo::getSourceManager();
  // Now this function only covers this pattern:
  // cufftResult R = cufftAPI();

  std::shared_ptr<ast_type_traits::DynTypedNode> P =
      std::make_shared<ast_type_traits::DynTypedNode>(
          ast_type_traits::DynTypedNode::create(*TheCallExpr));
  const VarDecl *VD = getNonImplicitCastParentNode(P)
                          ? getNonImplicitCastParentNode(P)
                              ->get<VarDecl>() : NULL;
  if (!VD)
    return false;
  if (VD->getInitStyle() != VarDecl::InitializationStyle::CInit)
    return false;

  auto NeedMove = [&]() -> bool {
    const Stmt *S = getParentStmt(VD);
    if (!S)
      return false;
    const DeclStmt *DS = dyn_cast<DeclStmt>(S);
    if (!DS)
      return false;

    S = getParentStmt(S);
    if (!S)
      return false;
    const CompoundStmt *CS = dyn_cast<const CompoundStmt>(S);
    if (!CS)
      return false;
    return true;
  };

  if (!NeedMove())
    return false;

  // get type location
  TypeBegin = VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
  SourceLocation TypeEnd = VD->getTypeSourceInfo()->getTypeLoc().getEndLoc();
  TypeEnd = TypeEnd.getLocWithOffset(
      Lexer::MeasureTokenLength(SM.getExpansionLoc(TypeEnd), SM,
                                DpctGlobalInfo::getContext().getLangOpts()));

  auto C = SM.getCharacterData(TypeEnd);
  int Offset = 0;
  while (C && isblank(*C)) {
    C++;
    Offset++;
  }
  TypeEnd = TypeEnd.getLocWithOffset(Offset);
  TypeLength = SM.getDecomposedLoc(TypeEnd).second -
               SM.getDecomposedLoc(TypeBegin).second;

  std::string TypeRepl = DpctGlobalInfo::getReplacedTypeName(VD->getType());
  std::string VarName = VD->getNameAsString();
  PrePrefixStmt = TypeRepl + " " + VarName + " = 0;";
  return true;
}

void initVars(const CallExpr *CE, const VarDecl *VD,
              LibraryMigrationFlags &Flags,
              LibraryMigrationStrings &ReplaceStrs,
              LibraryMigrationLocations &Locations) {
  auto &SM = DpctGlobalInfo::getSourceManager();

  if (Flags.IsFunctionPointer) {
    Locations.FuncPtrDeclBegin = SM.getExpansionLoc(VD->getBeginLoc());
    SourceLocation FuncPtrDeclEnd = SM.getExpansionLoc(VD->getEndLoc());
    FuncPtrDeclEnd = FuncPtrDeclEnd.getLocWithOffset(Lexer::MeasureTokenLength(
        FuncPtrDeclEnd, SM, DpctGlobalInfo::getContext().getLangOpts()));
    Locations.FuncPtrDeclLen =
        SM.getDecomposedLoc(FuncPtrDeclEnd).second -
        SM.getDecomposedLoc(Locations.FuncPtrDeclBegin).second;
    ReplaceStrs.IndentStr = getIndent(Locations.FuncPtrDeclBegin, SM).str();

    TypeLoc TL = VD->getTypeSourceInfo()->getTypeLoc();
    QualType QT = VD->getType();
    if (QT->isPointerType()) {
      QT = QT->getPointeeType();
      TL = TL.getAs<PointerTypeLoc>().getPointeeLoc();
    } else {
      Locations.FuncPtrDeclHandleTypeBegin = Locations.FuncPtrDeclBegin;
      return;
    }

    if (QT->getAs<ParenType>()) {
      QT = QT->getAs<ParenType>()->getInnerType();
      TL = TL.getAs<ParenTypeLoc>().getInnerLoc();
    } else {
      Locations.FuncPtrDeclHandleTypeBegin = Locations.FuncPtrDeclBegin;
      return;
    }

    if (QT->getAs<FunctionProtoType>()) {
      TL = TL.getAs<FunctionProtoTypeLoc>()
               .getParam(0)
               ->getTypeSourceInfo()
               ->getTypeLoc();
      Locations.FuncPtrDeclHandleTypeBegin =
          SM.getExpansionLoc(TL.getBeginLoc());
    } else {
      Locations.FuncPtrDeclHandleTypeBegin = Locations.FuncPtrDeclBegin;
      return;
    }
    return;
  }

  Locations.FuncNameBegin = CE->getBeginLoc();
  Locations.FuncCallEnd = CE->getEndLoc();

  Locations.OutOfMacroInsertLoc = SM.getExpansionLoc(CE->getBeginLoc());

  // TODO: For case like:
  //  #define CHECK_STATUS(x) fun(c)
  //  CHECK_STATUS(anAPICall());
  // Below code can distinguish this kind of function like macro, need refine to
  // cover more cases.
  Flags.IsMacroArg = SM.isMacroArgExpansion(CE->getBeginLoc()) &&
                     SM.isMacroArgExpansion(CE->getEndLoc());

  // Offset 1 is the length of the last token ")"
  Locations.FuncCallEnd =
      SM.getExpansionLoc(Locations.FuncCallEnd).getLocWithOffset(1);
  auto SR =
      getScopeInsertRange(CE, Locations.FuncNameBegin, Locations.FuncCallEnd);
  Locations.PrefixInsertLoc = SR.getBegin();
  Locations.SuffixInsertLoc = SR.getEnd();

  Flags.CanAvoidUsingLambda = false;
  Flags.NeedUseLambda = isConditionOfFlowControl(CE, Flags.OriginStmtType,
                                                 Flags.CanAvoidUsingLambda,
                                                 Locations.OuterInsertLoc);
  bool IsInReturnStmt = isInReturnStmt(CE, Locations.OuterInsertLoc);
  Flags.CanAvoidBrace = false;
  const CompoundStmt *CS = findImmediateBlock(CE);
  if (CS && (CS->size() == 1)) {
    const Stmt *S = *(CS->child_begin());
    if (CE == S || dyn_cast<ReturnStmt>(S))
      Flags.CanAvoidBrace = true;
  }

  if (Flags.NeedUseLambda || Flags.IsMacroArg || IsInReturnStmt) {
    Flags.NeedUseLambda = true;
    SourceRange SR = getFunctionRange(CE);
    Locations.PrefixInsertLoc = SR.getBegin();
    Locations.SuffixInsertLoc = SR.getEnd();
    if (IsInReturnStmt) {
      Flags.OriginStmtType = "return";
      Flags.CanAvoidUsingLambda = true;
    }
  }

  ReplaceStrs.IndentStr = getIndent(Locations.PrefixInsertLoc, SM).str();

  // This length should be used only when NeedUseLambda is true.
  // If NeedUseLambda is false, Len may longer than the function call length,
  // because in this case, PrefixInsertLoc and SuffixInsertLoc are the begin
  // location of the whole statement and the location after the semi of the
  // statement.
  Locations.Len = SM.getDecomposedLoc(Locations.SuffixInsertLoc).second -
                  SM.getDecomposedLoc(Locations.PrefixInsertLoc).second;
}

void FFTFunctionCallBuilder::updateFFTPlanAPIInfo(
    FFTPlanAPIInfo &FPAInfo, LibraryMigrationFlags &Flags, int Index) {
  std::string PrecAndDomainStr;
  FFTTypeEnum FFTType;
  std::int64_t Rank = -1;
  StringRef FuncNameRef(FuncName);

  if (FuncNameRef.endswith("1d")) {
    PrecAndDomainStr = getPrecAndDomainStr(2);
    FFTType = getFFTType(2);
    Rank = 1;

    Expr::EvalResult ER;
    if (!TheCallExpr->getArg(3)->isValueDependent() &&
        TheCallExpr->getArg(3)->EvaluateAsInt(ER,
                                              DpctGlobalInfo::getContext())) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 1) {
        FPAInfo.NeedBatchFor1D = false;
      }
    }
  } else if (FuncNameRef.endswith("2d")) {
    PrecAndDomainStr = getPrecAndDomainStr(3);
    FFTType = getFFTType(3);
    Rank = 2;
  } else if (FuncNameRef.endswith("3d")) {
    PrecAndDomainStr = getPrecAndDomainStr(4);
    FFTType = getFFTType(4);
    Rank = 3;
  } else {
    // cufftPlanMany/cufftMakePlanMany/cufftMakePlanMany64
    PrecAndDomainStr = getPrecAndDomainStr(9);
    FFTType = getFFTType(9);
    Expr::EvalResult ER;
    if (!TheCallExpr->getArg(1)->isValueDependent() &&
        TheCallExpr->getArg(1)->EvaluateAsInt(ER,
                                              DpctGlobalInfo::getContext())) {
      Rank = ER.Val.getInt().getExtValue();
    }
  }

  Flags.CanAvoidBrace = true;
  Flags.MoveOutOfMacro = true;
  addDescriptorTypeInfo(PrecAndDomainStr);
  if (!PrecAndDomainStr.empty())
    DpctGlobalInfo::getPrecAndDomPairSet().insert(PrecAndDomainStr);

  FPAInfo.addInfo(PrecAndDomainStr, FFTType, Index, ArgsList,
                  ArgsListAddRequiredParen, IndentStr, FuncName, Flags, Rank,
                  getDescrMemberCallPrefix(), getDescr());
}

void FFTFunctionCallBuilder::updateFFTExecAPIInfo(
    std::string FFTExecAPIInfoKey) {
  StringRef FuncNameRef(FuncName);
  FFTPlacementType Placement;
  if (isInplace(TheCallExpr->getArg(1), TheCallExpr->getArg(2))) {
    Placement = FFTPlacementType::inplace;
  } else {
    Placement = FFTPlacementType::outofplace;
  }

  if (FuncNameRef.endswith("C2C") || FuncNameRef.endswith("Z2Z")) {
    Expr::EvalResult ER;
    if (!TheCallExpr->getArg(3)->isValueDependent() &&
        TheCallExpr->getArg(3)->EvaluateAsInt(ER,
                                              DpctGlobalInfo::getContext())) {
      int64_t Dir = ER.Val.getInt().getExtValue();
      assembleExecCallExpr(Dir);
      if (Dir == -1) {
        DpctGlobalInfo::insertOrUpdateFFTExecAPIInfo(
            FFTExecAPIInfoKey, FFTDirectionType::forward, Placement);
      } else {
        DpctGlobalInfo::insertOrUpdateFFTExecAPIInfo(
            FFTExecAPIInfoKey, FFTDirectionType::backward, Placement);
      }
    } else {
      assembleExecCallExpr(int64_t(0));
      DpctGlobalInfo::insertOrUpdateFFTExecAPIInfo(
          FFTExecAPIInfoKey, FFTDirectionType::unknown, Placement);
    }
  } else if (FuncNameRef.endswith("R2C") || FuncNameRef.endswith("D2Z")) {
    assembleExecCallExpr(-1);
    DpctGlobalInfo::insertOrUpdateFFTExecAPIInfo(
        FFTExecAPIInfoKey, FFTDirectionType::forward, Placement);
  } else {
    assembleExecCallExpr(1);
    DpctGlobalInfo::insertOrUpdateFFTExecAPIInfo(
        FFTExecAPIInfoKey, FFTDirectionType::backward, Placement);
  }
}

void FFTFunctionCallBuilder::updateFFTExecAPIInfo() {
  StringRef FuncNameRef(FuncName);
  if (FuncNameRef.endswith("C2C") || FuncNameRef.endswith("Z2Z")) {
    assembleExecCallExpr(int64_t(0));
  } else if (FuncNameRef.endswith("R2C") || FuncNameRef.endswith("D2Z")) {
    assembleExecCallExpr(-1);
  } else {
    assembleExecCallExpr(1);
  }
}

// return true means in-place, return false means cannot deduce.
bool FFTFunctionCallBuilder::isInplace(const Expr *Ptr1, const Expr *Ptr2) {
  auto RemoveCStyleCast = [](const Expr *E) -> const Expr * {
    if (const CStyleCastExpr *CSCE = dyn_cast<CStyleCastExpr>(E)) {
      return CSCE->getSubExpr();
    } else {
      return E;
    }
  };

  const Expr *CleanPtr1 = RemoveCStyleCast(Ptr1);
  const Expr *CleanPtr2 = RemoveCStyleCast(Ptr2);
  if (ExprAnalysis::ref(CleanPtr1) == ExprAnalysis::ref(CleanPtr2))
    return true;
  else
    return false;
}

// Prefix has included the array since all Descr are migrated to shared_ptr
std::string FFTFunctionCallBuilder::getDescrMemberCallPrefix() {
  std::string MemberCallPrefix;
  std::string Descr = getDescr();
  if ('*' == *Descr.begin()) {
    MemberCallPrefix = "(" + Descr + ")->";
  } else {
    MemberCallPrefix = Descr + "->";
  }
  return MemberCallPrefix;
}

std::string FFTFunctionCallBuilder::getDescr() {
  StringRef FuncNameRef(FuncName);
  std::string Descr;
  if (FuncNameRef.startswith("cufftMake") ||
      FuncNameRef.startswith("cufftExec"))
    Descr = ArgsList[0];
  else
    Descr = getDrefName(TheCallExpr->getArg(0));
  return Descr;
}

} // namespace dpct
} // namespace clang
