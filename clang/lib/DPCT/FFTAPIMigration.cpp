//===--- FFTAPIMigration.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "FFTAPIMigration.h"
#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "MapNames.h"
#include "Utility.h"

#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
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

  if (HandleVar->getTypeSourceInfo()->getTypeLoc().getBeginLoc().isInvalid()) {
    return;
  }

  auto TypeBeginLoc =
      getDefinitionRange(
          HandleVar->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
          HandleVar->getTypeSourceInfo()->getTypeLoc().getEndLoc())
          .getBegin();
  // WA for concatinated macro token
  if (SM.isWrittenInScratchSpace(SM.getSpellingLoc(
          HandleVar->getTypeSourceInfo()->getTypeLoc().getBeginLoc()))) {
    TypeBeginLoc = SM.getExpansionLoc(
        HandleVar->getTypeSourceInfo()->getTypeLoc().getBeginLoc());
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
  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
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
    if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment) {
      requestFeature(HelperFeatureEnum::Memory_get_buffer_T,
                     Locations.FuncPtrDeclBegin);
    } else {
      requestFeature(HelperFeatureEnum::Memory_get_buffer_T,
                     Locations.PrefixInsertLoc);
    }

    BufferDecl = "auto " + ArgsList[Idx] + " = " +
                 MapNames::getDpctNamespace() + "get_buffer<" + TypeStr + ">(" +
                 PointerName + ");";
    PrefixStmts.emplace_back(BufferDecl);
  } else {
    if ((Idx == 1 && (FuncName[9] == 'C' || FuncName[9] == 'Z')) ||
        (Idx == 2 && (FuncName[11] == 'C' || FuncName[11] == 'Z')))
      ArgsList[Idx] = "(" + TypeStr + "*)" + ArgsList[Idx];
  }
}

/// build exec API info
void FFTFunctionCallBuilder::assembleExecCallExpr() {
  auto getRealDomainType = [=](const char C) -> std::string {
    if (C == 'Z' || C == 'D')
      return "double";
    else
      return "float";
  };

  auto getType = [=](const char C) -> std::string {
    if (C == 'Z')
      return MapNames::getClNamespace() + "double2";
    else if (C == 'C')
      return MapNames::getClNamespace() + "float2";
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

  if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment) {
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

    std::string WrapperBegin;
    if (Flags.IsFunctionPointer)
      WrapperBegin = "auto " + FuncPtrName +
                     " = [](std::shared_ptr<oneapi::mkl::dft::descriptor<" +
                     getPrecAndDomainStrFromExecFuncName(FuncName) +
                     ">> desc, " + getType(FuncName[9]) + " *in_data, " +
                     getType(FuncName[11]) + " *out_data";
    else if (Flags.IsFunctionPointerAssignment)
      WrapperBegin = "[](std::shared_ptr<oneapi::mkl::dft::descriptor<" +
                     getPrecAndDomainStrFromExecFuncName(FuncName) +
                     ">> desc, " + getType(FuncName[9]) + " *in_data, " +
                     getType(FuncName[11]) + " *out_data";

    if (FuncName[9] == FuncName[11])
      WrapperBegin = WrapperBegin + ", int dir";
    WrapperBegin = WrapperBegin + "){";
    PrefixStmts.emplace_back(std::move(WrapperBegin));

    updateBufferArgs(1, getRealDomainType(FuncName[9]), OriginalInputPtr);
  } else {
    updateBufferArgs(1, getRealDomainType(FuncName[9]));
  }

  if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(
        DpctGlobalInfo::getSourceManager().getExpansionLoc(
            Locations.FuncPtrDeclBegin));
    if (DiagnosticsUtils::report(LocInfo.first, LocInfo.second,
                                 Diagnostics::CHECK_RELATED_QUEUE, false,
                                 false)) {
      PrefixStmts.push_back("/*");
      PrefixStmts.push_back(DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
          Diagnostics::CHECK_RELATED_QUEUE));
      PrefixStmts.push_back("*/");
      PrefixStmts.push_back("desc->commit(" + MapNames::getDpctNamespace() +
                            "get_default_queue());");
      requestFeature(HelperFeatureEnum::Device_get_default_queue,
                     LocInfo.first);
    }
  }
  PrefixStmts.emplace_back("if ((void *)" + OriginalInputPtr + " == (void *)" +
                           OriginalOutputPtr + ") {");
  if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment) {
    PrefixStmts.emplace_back(ComputeAPI + "(*" + ArgsList[0] + ", " +
                             ArgsList[1] + ");");
  } else {
    PrefixStmts.emplace_back(ComputeAPI + "(" +
                             getDrefName(TheCallExpr->getArg(0)) + ", " +
                             ArgsList[1] + ");");
  }

  PrefixStmts.emplace_back("} else {");

  if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment) {
    updateBufferArgs(2, getRealDomainType(FuncName[11]), OriginalOutputPtr);
    CallExprRepl = ComputeAPI + "(*" + ArgsList[0] + ", " + ArgsList[1] + ", " +
                   ArgsList[2] + ")";
  } else {
    updateBufferArgs(2, getRealDomainType(FuncName[11]));
    CallExprRepl = ComputeAPI + "(" + getDrefName(TheCallExpr->getArg(0)) +
                   ", " + ArgsList[1] + ", " + ArgsList[2] + ")";
  }
  SuffixStmts.emplace_back("}");
  if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment) {
    SuffixStmts.emplace_back("return 0;");
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
  if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment)
    return false;

  auto &SM = DpctGlobalInfo::getSourceManager();
  // Now this function only covers this pattern:
  // cufftResult R = cufftAPI();

  std::shared_ptr<DynTypedNode> P =
      std::make_shared<DynTypedNode>(DynTypedNode::create(*TheCallExpr));
  auto PN = getNonImplicitCastParentNode(P);
  if(!PN)
    return false;

  const VarDecl *VD = getNonImplicitCastParentNode(P)->get<VarDecl>();
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

void FFTFunctionCallBuilder::updateFFTPlanAPIInfo(
    FFTPlanAPIInfo &FPAInfo, LibraryMigrationFlags &Flags) {
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

  FPAInfo.addInfo(PrecAndDomainStr, FFTType, ArgsList, ArgsListAddRequiredParen,
                  IndentStr, FuncName, Flags, Rank, getDescrMemberCallPrefix(),
                  getDescr());
}

void FFTFunctionCallBuilder::updateExecCallExpr(std::string FFTHandleInfoKey) {
  StringRef FuncNameRef(FuncName);
  FFTPlacementType Placement;
  if (isInplace(TheCallExpr->getArg(1), TheCallExpr->getArg(2))) {
    Placement = FFTPlacementType::Inplace;
  } else {
    Placement = FFTPlacementType::Outofplace;
  }

  if (FuncNameRef.endswith("C2C") || FuncNameRef.endswith("Z2Z")) {
    Expr::EvalResult ER;
    if (!TheCallExpr->getArg(3)->isValueDependent() &&
        TheCallExpr->getArg(3)->EvaluateAsInt(ER,
                                              DpctGlobalInfo::getContext())) {
      Dir = ER.Val.getInt().getExtValue();
      assembleExecCallExpr();
      if (Dir == -1) {
        DpctGlobalInfo::insertOrUpdateFFTHandleInfo(
            FFTHandleInfoKey, FFTDirectionType::Forward, Placement);
      } else {
        DpctGlobalInfo::insertOrUpdateFFTHandleInfo(
            FFTHandleInfoKey, FFTDirectionType::Backward, Placement);
      }
    } else {
      Dir = 0;
      assembleExecCallExpr();
      DpctGlobalInfo::insertOrUpdateFFTHandleInfo(
          FFTHandleInfoKey, FFTDirectionType::Unknown, Placement);
    }
  } else if (FuncNameRef.endswith("R2C") || FuncNameRef.endswith("D2Z")) {
    Dir = -1;
    assembleExecCallExpr();
    DpctGlobalInfo::insertOrUpdateFFTHandleInfo(
        FFTHandleInfoKey, FFTDirectionType::Forward, Placement);
  } else {
    Dir = 1;
    assembleExecCallExpr();
    DpctGlobalInfo::insertOrUpdateFFTHandleInfo(
        FFTHandleInfoKey, FFTDirectionType::Backward, Placement);
  }
}

void FFTFunctionCallBuilder::updateExecCallExpr() {
  StringRef FuncNameRef(FuncName);
  if (FuncNameRef.endswith("C2C") || FuncNameRef.endswith("Z2Z")) {
    Dir = 0;
    assembleExecCallExpr();
  } else if (FuncNameRef.endswith("R2C") || FuncNameRef.endswith("D2Z")) {
    Dir = -1;
    assembleExecCallExpr();
  } else {
    Dir = 1;
    assembleExecCallExpr();
  }
}

void FFTFunctionCallBuilder::updateFFTHandleInfoFromPlan(
    std::string FFTHandleInfoKey) {
  if (FuncName == "cufftPlanMany" || FuncName == "cufftMakePlanMany" ||
      FuncName == "cufftMakePlanMany64") {
    DpctGlobalInfo::insertOrUpdateFFTHandleInfo(FFTHandleInfoKey, true,
                                                ArgsList[5], ArgsList[8],
                                                ArgsList[3], ArgsList[6]);
  }
}

void FFTFunctionCallBuilder::updateFFTExecAPIInfo(FFTExecAPIInfo &FEAInfo) {
  bool IsComplexDomainInput =
      (FuncName == "cufftExecC2C") || (FuncName == "cufftExecZ2Z");
  FEAInfo.addInfo(IndentStr, Flags, PrePrefixStmt, PrefixStmts, SuffixStmts,
                  CallExprRepl, IsComplexDomainInput, getDescr(), Dir);
  replacementLocation(Locations, Flags, FEAInfo.ReplaceOffset,
                      FEAInfo.ReplaceLen, FEAInfo.InsertOffsets,
                      FEAInfo.FilePath);
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

void FFTPlanAPIInfo::buildInfo() {
  linkInfo();
  replacementText(Flags, PrePrefixStmt, PrefixStmts, SuffixStmts, CallExprRepl,
                  IndentStr, FilePath, ReplaceOffset, ReplaceLen,
                  InsertOffsets);
}

void FFTPlanAPIInfo::linkInfo() {
  auto &Map = DpctGlobalInfo::getFFTHandleInfoMap();
  auto I = Map.find(HandleDeclFileAndOffset);
  if (I != Map.end()) {
    DirectionFromExec = I->second.Direction;
    PlacementFromExec = I->second.Placement;
  }

  if (FFTType == FFTTypeEnum::Unknown &&
      DpctGlobalInfo::getFFTTypeSet().size() == 1) {
    FFTType = *(DpctGlobalInfo::getFFTTypeSet().begin());
  }

  StringRef FuncNameRef(FuncName);

  if (FuncNameRef.endswith("1d")) {
    update1D2D3DCommitCallExpr({1});
    setValueFor1DBatched();
  } else if (FuncNameRef.endswith("2d")) {
    update1D2D3DCommitCallExpr({1, 2});
  } else if (FuncNameRef.endswith("3d")) {
    update1D2D3DCommitCallExpr({1, 2, 3});
  } else {
    updateManyCommitCallExpr();
  }
}

void FFTPlanAPIInfo::addInfo(
    std::string PrecAndDomainStrInput, FFTTypeEnum FFTTypeInput,
    std::vector<std::string> ArgsListInput,
    std::vector<std::string> ArgsListAddRequiredParenInput,
    std::string IndentStrInput, std::string FuncNameInput,
    LibraryMigrationFlags FlagsInput, std::int64_t RankInput,
    std::string DescrMemberCallPrefixInput, std::string DescStrInput) {
  PrecAndDomainStr = PrecAndDomainStrInput;
  FFTType = FFTTypeInput;
  ArgsList = ArgsListInput;
  ArgsListAddRequiredParen = ArgsListAddRequiredParenInput;
  IndentStr = IndentStrInput;
  FuncName = FuncNameInput;
  Flags = FlagsInput;
  Rank = RankInput;
  DescrMemberCallPrefix = DescrMemberCallPrefixInput;
  DescStr = DescStrInput;
}

void FFTPlanAPIInfo::setValueFor1DBatched() {
  if (!NeedBatchFor1D)
    return;

  // in-place:
  // C2R,R2C,D2Z,Z2D: real-distance: (n/2+1)*2, complex-distance: n/2+1
  // C2C,Z2Z: distance: n
  // out-of-place:
  // C2R,R2C,D2Z,Z2D: real-distance: n, complex-distance: n/2+1
  // C2C,Z2Z: distance: n

  std::string SetStr = DescrMemberCallPrefix + "set_value";
  if (FFTType == FFTTypeEnum::R2C || FFTType == FFTTypeEnum::D2Z ||
      FFTType == FFTTypeEnum::C2R || FFTType == FFTTypeEnum::Z2D) {
    if (PlacementFromExec != FFTPlacementType::Inplace) {
      // out-of-place
      SuffixStmts.emplace_back(
          SetStr + "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
          ArgsList[1] + ");");
    } else {
      // in-place
      SuffixStmts.emplace_back(
          SetStr + "(oneapi::mkl::dft::config_param::FWD_DISTANCE, (" +
          ArgsListAddRequiredParen[1] + "/2+1)*2);");
    }
    SuffixStmts.emplace_back(SetStr +
                             "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                             ArgsListAddRequiredParen[1] + "/2+1);");
  } else if (FFTType == FFTTypeEnum::C2C || FFTType == FFTTypeEnum::Z2Z) {
    SuffixStmts.emplace_back(SetStr +
                             "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                             ArgsList[1] + ");");
    SuffixStmts.emplace_back(SetStr +
                             "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                             ArgsList[1] + ");");
  } else if (FFTType == FFTTypeEnum::Unknown) {
    DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                             Diagnostics::UNDEDUCED_PARAM, true, false,
                             "FFT type", "'FWD_DISTANCE' and 'BWD_DISTANCE'");
  }

  SuffixStmts.emplace_back(
      SetStr + "(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, " +
      ArgsList[3] + ");");
}

std::vector<std::string> FFTPlanAPIInfo::setValueForBasicManyBatched(
    std::vector<std::string> Dims, std::vector<std::string> DimsWithoutParen) {
  // Dims[rank-1] is innermost, Dims[0] is outermost

  // Below "n" is the innermost dim, and then all distance need multiply the
  // rest dims value. in-place: C2R,R2C,D2Z,Z2D: real-distance: (n/2+1)*2,
  // complex-distance: n/2+1 C2C,Z2Z: distance: n out-of-place: C2R,R2C,D2Z,Z2D:
  // real-distance: n, complex-distance: n/2+1 C2C,Z2Z: distance: n

  std::vector<std::string> Res;
  std::string SetStr = DescrMemberCallPrefix + "set_value";

  if (Dims.size() == 1) {
    if (FFTType == FFTTypeEnum::R2C || FFTType == FFTTypeEnum::D2Z ||
        FFTType == FFTTypeEnum::C2R || FFTType == FFTTypeEnum::Z2D) {
      if (PlacementFromExec != FFTPlacementType::Inplace) {
        // out-of-place
        Res.emplace_back(SetStr +
                         "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                         DimsWithoutParen[0] + ");");
      } else {
        // in-place
        Res.emplace_back(SetStr +
                         "(oneapi::mkl::dft::config_param::FWD_DISTANCE, (" +
                         Dims[0] + "/2+1)*2);");
      }
      Res.emplace_back(SetStr +
                       "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                       Dims[0] + "/2+1);");
    } else if (FFTType == FFTTypeEnum::C2C || FFTType == FFTTypeEnum::Z2Z) {
      Res.emplace_back(SetStr +
                       "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                       DimsWithoutParen[0] + ");");
      Res.emplace_back(SetStr +
                       "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                       DimsWithoutParen[0] + ");");
    }
  } else if (Dims.size() == 2) {
    if (FFTType == FFTTypeEnum::R2C || FFTType == FFTTypeEnum::D2Z ||
        FFTType == FFTTypeEnum::C2R || FFTType == FFTTypeEnum::Z2D) {
      if (PlacementFromExec != FFTPlacementType::Inplace) {
        // out-of-place
        Res.emplace_back(SetStr +
                         "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                         Dims[1] + "*" + Dims[0] + ");");
      } else {
        // in-place
        Res.emplace_back(SetStr +
                         "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                         Dims[1] + "*(" + Dims[0] + "/2+1)*2);");
      }
      Res.emplace_back(SetStr +
                       "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                       Dims[1] + "*(" + Dims[0] + "/2+1));");
    } else if (FFTType == FFTTypeEnum::C2C || FFTType == FFTTypeEnum::Z2Z) {
      Res.emplace_back(SetStr +
                       "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                       Dims[1] + "*" + Dims[0] + ");");
      Res.emplace_back(SetStr +
                       "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                       Dims[1] + "*" + Dims[0] + ");");
    }
  } else {
    if (FFTType == FFTTypeEnum::R2C || FFTType == FFTTypeEnum::D2Z ||
        FFTType == FFTTypeEnum::C2R || FFTType == FFTTypeEnum::Z2D) {
      if (PlacementFromExec != FFTPlacementType::Inplace) {
        // out-of-place
        Res.emplace_back(SetStr +
                         "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                         Dims[2] + "*" + Dims[1] + "*" + Dims[0] + ");");
      } else {
        // in-place
        Res.emplace_back(
            SetStr + "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
            Dims[2] + "*" + Dims[1] + "*(" + Dims[0] + "/2+1)*2);");
      }
      Res.emplace_back(SetStr +
                       "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                       Dims[2] + "*" + Dims[1] + "*(" + Dims[0] + "/2+1));");
    } else if (FFTType == FFTTypeEnum::C2C || FFTType == FFTTypeEnum::Z2Z) {
      Res.emplace_back(SetStr +
                       "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                       Dims[2] + "*" + Dims[1] + "*" + Dims[0] + ");");
      Res.emplace_back(SetStr +
                       "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                       Dims[2] + "*" + Dims[1] + "*" + Dims[0] + ");");
    }
  }
  return Res;
}

void FFTPlanAPIInfo::updateManyCommitCallExpr() {
  Expr::EvalResult ER;
  std::vector<std::string>
      Dims; // Dims[rank-1] is innermost, Dims[0] is outermost
  std::string InStridesStr;
  std::string OutStridesStr;

  std::string SetStr = DescrMemberCallPrefix + "set_value";

  std::string InputStrideName =
      "input_stride_ct" +
      std::to_string(DpctGlobalInfo::getSuffixIndexGlobalThenInc());
  std::string OutputStrideName =
      "output_stride_ct" +
      std::to_string(DpctGlobalInfo::getSuffixIndexGlobalThenInc());

  if (Rank != -1) {
    // dim = 3:
    // s[3]=stride
    // s[2]=stride*nembed[2]
    // s[1]=stride*nembed[2]*nembed[1]
    // s[0]=0
    //
    // dim = 2:
    // s[2]=stride
    // s[1]=stride*nembed[1]
    // s[0]=0
    //
    // dim = 1:
    // s[1]=stride
    // s[0]=0
    std::vector<std::string> InStrides;
    std::vector<std::string> OutStrides;
    InStrides.emplace_back("0");
    OutStrides.emplace_back("0");

    if (Rank == 1) {
      InStrides.emplace_back(ArgsList[4]);
      OutStrides.emplace_back(ArgsList[7]);
    } else if (Rank == 2) {
      InStrides.emplace_back(ArgsListAddRequiredParen[3] + "[1] * " +
                             ArgsListAddRequiredParen[4]);
      OutStrides.emplace_back(ArgsListAddRequiredParen[6] + "[1] * " +
                              ArgsListAddRequiredParen[7]);
      InStrides.emplace_back(ArgsList[4]);
      OutStrides.emplace_back(ArgsList[7]);
    } else if (Rank == 3) {
      InStrides.emplace_back(ArgsListAddRequiredParen[3] + "[2] * " +
                             ArgsListAddRequiredParen[3] + "[1] * " +
                             ArgsListAddRequiredParen[4]);
      OutStrides.emplace_back(ArgsListAddRequiredParen[6] + "[2] * " +
                              ArgsListAddRequiredParen[6] + "[1] * " +
                              ArgsListAddRequiredParen[7]);
      InStrides.emplace_back(ArgsListAddRequiredParen[3] + "[2] * " +
                             ArgsListAddRequiredParen[4]);
      OutStrides.emplace_back(ArgsListAddRequiredParen[6] + "[2] * " +
                              ArgsListAddRequiredParen[7]);
      InStrides.emplace_back(ArgsList[4]);
      OutStrides.emplace_back(ArgsList[7]);
    }
    for (int64_t i = 0; i < Rank; ++i)
      Dims.emplace_back(ArgsListAddRequiredParen[2] + "[" + std::to_string(i) +
                        "]");
    for (size_t i = 0; i < InStrides.size(); ++i) {
      InStridesStr = InStridesStr + InStrides[i] + ", ";
      OutStridesStr = OutStridesStr + OutStrides[i] + ", ";
    }
    InStridesStr = InStridesStr.substr(0, InStridesStr.size() - 2);
    OutStridesStr = OutStridesStr.substr(0, OutStridesStr.size() - 2);
    InStridesStr = "std::int64_t " + InputStrideName + "[" +
                   std::to_string(Rank + 1) + "] = {" + InStridesStr + "};";
    OutStridesStr = "std::int64_t " + OutputStrideName + "[" +
                    std::to_string(Rank + 1) + "] = {" + OutStridesStr + "};";
  } else {
    DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                             Diagnostics::UNDEDUCED_PARAM, true, false,
                             "dimensions and strides", "'dpct_placeholder'");
    Dims.emplace_back("dpct_placeholder/*Fix the dimensions manually*/");
    InStridesStr = "std::int64_t " + InputStrideName +
                   "[dpct_placeholder/*Fix the "
                   "dimensions manually*/] = {dpct_placeholder/*Fix the stride "
                   "manually*/};";
    OutStridesStr = "std::int64_t " + OutputStrideName +
                    "[dpct_placeholder/*Fix the "
                    "dimensions manually*/] = {dpct_placeholder/*Fix the "
                    "stride manually*/};";
  }

  updateCommitCallExpr(Dims);

  SuffixStmts.emplace_back(
      SetStr + "(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, " +
      ArgsList[10] + ");");

  SuffixStmts.emplace_back("if (" + ArgsListAddRequiredParen[3] +
                           " != nullptr && " + ArgsListAddRequiredParen[6] +
                           " != nullptr) {");

  SuffixStmts.emplace_back(InStridesStr);
  SuffixStmts.emplace_back(OutStridesStr);

  if (FFTType == FFTTypeEnum::R2C || FFTType == FFTTypeEnum::D2Z ||
      ((FFTType == FFTTypeEnum::C2C || FFTType == FFTTypeEnum::Z2Z) &&
       DirectionFromExec == FFTDirectionType::Forward)) {
    SuffixStmts.emplace_back(SetStr +
                             "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                             ArgsList[5] + ");");
    SuffixStmts.emplace_back(SetStr +
                             "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                             ArgsList[8] + ");");
  } else if (FFTType == FFTTypeEnum::C2R || FFTType == FFTTypeEnum::Z2D ||
             ((FFTType == FFTTypeEnum::C2C || FFTType == FFTTypeEnum::Z2Z) &&
              DirectionFromExec == FFTDirectionType::Backward)) {
    SuffixStmts.emplace_back(SetStr +
                             "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                             ArgsList[8] + ");");
    SuffixStmts.emplace_back(SetStr +
                             "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                             ArgsList[5] + ");");
  } else if (FFTType == FFTTypeEnum::C2C || FFTType == FFTTypeEnum::Z2Z) {
    // DirectionFromExec is "unknown" or "uninitialized"
    if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment) {
      DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                               Diagnostics::ONLY_SUPPORT_SAME_DISTANCE, true,
                               false, "the FWD_DISTANCE and the BWD_DISTANCE");
      SuffixStmts.emplace_back(
          SetStr + "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
          ArgsList[5] + ");");
      SuffixStmts.emplace_back(
          SetStr + "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
          ArgsList[8] + ");");
    }
  } else {
    DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                             Diagnostics::UNDEDUCED_PARAM, true, false,
                             "FFT type", "'FWD_DISTANCE' and 'BWD_DISTANCE'");
  }

  SuffixStmts.emplace_back(SetStr +
                           "(oneapi::mkl::dft::config_param::INPUT_STRIDES, " +
                           InputStrideName + ");");
  SuffixStmts.emplace_back(SetStr +
                           "(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, " +
                           OutputStrideName + ");");

  std::vector<std::string> Res;
  if (Rank != -1) {
    std::vector<std::string> DimsWithoutParen;
    for (int64_t i = 0; i < Rank; ++i)
      DimsWithoutParen.emplace_back(ArgsList[2] + "[" + std::to_string(i) +
                                    "]");

    Res = update1D2D3DCommitPrefix(Dims);
    std::vector<std::string> BasicManyBatchedRes =
        setValueForBasicManyBatched(Dims, DimsWithoutParen);
    Res.insert(Res.end(), BasicManyBatchedRes.begin(),
               BasicManyBatchedRes.end());
  }

  if (Res.empty()) {
    SuffixStmts.emplace_back("}");
  } else {
    SuffixStmts.emplace_back("} else {");
    SuffixStmts.insert(SuffixStmts.end(), Res.begin(), Res.end());
    SuffixStmts.emplace_back("}");
  }
}

std::vector<std::string>
FFTPlanAPIInfo::update1D2D3DCommitPrefix(std::vector<std::string> Dims) {
  // Dims[rank-1] is innermost, Dims[0] is outermost
  std::vector<std::string> ResultStmts;
  std::string SetStr = DescrMemberCallPrefix + "set_value";

  std::string InputStrideName =
      "input_stride_ct" +
      std::to_string(DpctGlobalInfo::getSuffixIndexGlobalThenInc());
  std::string OutputStrideName =
      "output_stride_ct" +
      std::to_string(DpctGlobalInfo::getSuffixIndexGlobalThenInc());

  if (FFTType == FFTTypeEnum::R2C || FFTType == FFTTypeEnum::D2Z) {
    if (Dims.size() == 1) {
      if (PlacementFromExec == FFTPlacementType::Inplace) {
        ResultStmts.emplace_back("std::int64_t " + InputStrideName +
                                 "[2] = {0, 1};");
      }
      ResultStmts.emplace_back("std::int64_t " + OutputStrideName +
                               "[2] = {0, 1};");
    } else if (Dims.size() == 2) {
      if (PlacementFromExec == FFTPlacementType::Inplace) {
        ResultStmts.emplace_back("std::int64_t " + InputStrideName +
                                 "[3] = {0, (" + Dims[1] + "/2+1)*2, 1};");
      }
      ResultStmts.emplace_back("std::int64_t " + OutputStrideName +
                               "[3] = {0, (" + Dims[1] + "/2+1), 1};");
    } else {
      if (PlacementFromExec == FFTPlacementType::Inplace) {
        ResultStmts.emplace_back("std::int64_t " + InputStrideName +
                                 "[4] = {0, " + Dims[1] + "*(" + Dims[2] +
                                 "/2+1)*2, (" + Dims[2] + "/2+1)*2, 1};");
      }
      ResultStmts.emplace_back("std::int64_t " + OutputStrideName +
                               "[4] = {0, " + Dims[1] + "*(" + Dims[2] +
                               "/2+1), (" + Dims[2] + "/2+1), 1};");
    }

    if (PlacementFromExec == FFTPlacementType::Inplace) {
      ResultStmts.emplace_back(
          SetStr + "(oneapi::mkl::dft::config_param::INPUT_STRIDES, " +
          InputStrideName + ");");
    }
    ResultStmts.emplace_back(
        SetStr + "(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, " +
        OutputStrideName + ");");
  } else if (FFTType == FFTTypeEnum::C2R || FFTType == FFTTypeEnum::Z2D) {
    if (Dims.size() == 1) {
      if (PlacementFromExec == FFTPlacementType::Inplace) {
        ResultStmts.emplace_back("std::int64_t " + OutputStrideName +
                                 "[2] = {0, 1};");
      }
      ResultStmts.emplace_back("std::int64_t " + InputStrideName +
                               "[2] = {0, 1};");
    } else if (Dims.size() == 2) {
      if (PlacementFromExec == FFTPlacementType::Inplace) {
        ResultStmts.emplace_back("std::int64_t " + OutputStrideName +
                                 "[3] = {0, (" + Dims[1] + "/2+1)*2, 1};");
      }
      ResultStmts.emplace_back("std::int64_t " + InputStrideName +
                               "[3] = {0, (" + Dims[1] + "/2+1), 1};");
    } else {
      if (PlacementFromExec == FFTPlacementType::Inplace) {
        ResultStmts.emplace_back("std::int64_t " + OutputStrideName +
                                 "[4] = {0, " + Dims[1] + "*(" + Dims[2] +
                                 "/2+1)*2, (" + Dims[2] + "/2+1)*2, 1};");
      }
      ResultStmts.emplace_back("std::int64_t " + InputStrideName +
                               "[4] = {0, " + Dims[1] + "*(" + Dims[2] +
                               "/2+1), (" + Dims[2] + "/2+1), 1};");
    }

    if (PlacementFromExec == FFTPlacementType::Inplace) {
      ResultStmts.emplace_back(
          SetStr + "(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, " +
          OutputStrideName + ");");
    }
    ResultStmts.emplace_back(
        SetStr + "(oneapi::mkl::dft::config_param::INPUT_STRIDES, " +
        InputStrideName + ");");
  } else if (FFTType == FFTTypeEnum::Unknown) {
    DiagnosticsUtils::report(
        FilePath, InsertOffsets.first, Diagnostics::UNDEDUCED_PARAM, true,
        false, "FFT type", "'INPUT_STRIDES' and 'OUTPUT_STRIDES'");
  }
  return ResultStmts;
}

void FFTPlanAPIInfo::update1D2D3DCommitCallExpr(std::vector<int> DimIdxs) {
  std::vector<std::string> Dims;
  for (const auto &Idx : DimIdxs)
    Dims.emplace_back(ArgsList[Idx]);

  updateCommitCallExpr(Dims);

  Dims.clear();
  for (const auto &Idx : DimIdxs)
    Dims.emplace_back(ArgsList[Idx]);

  std::vector<std::string> Res = update1D2D3DCommitPrefix(Dims);
  SuffixStmts.insert(SuffixStmts.end(), Res.begin(), Res.end());
}

void FFTPlanAPIInfo::updateCommitCallExpr(std::vector<std::string> Dims) {
  if (FuncName.substr(0, 9) == "cufftMake") {
    DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                             Diagnostics::UNSUPPORTED_PARAM, true, false,
                             UnsupportedArg);
  }

  if (PrecAndDomainStr.empty()) {
    if (DpctGlobalInfo::getPrecAndDomPairSet().size() == 1) {
      PrecAndDomainStr = *(DpctGlobalInfo::getPrecAndDomPairSet().begin());
    } else {
      DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                               Diagnostics::UNDEDUCED_TYPE, true, false,
                               "FFT precision and domain type");
      PrecAndDomainStr =
          "dpct_placeholder/*Fix the precision and domain type manually*/";
    }
  }

  std::string DescCtor = DescStr + " = ";
  DescCtor = DescCtor + "std::make_shared<oneapi::mkl::dft::descriptor<" +
             PrecAndDomainStr + ">>(";
  if (Dims.size() == 1) {
    DescCtor = DescCtor + Dims[0];
  } else {
    std::string DimStr;
    for (const auto &Dim : Dims) {
      DimStr = DimStr + Dim + ", ";
    }
    DimStr = DimStr.substr(0, DimStr.size() - 2);
    DescCtor = DescCtor + "std::vector<std::int64_t>{" + DimStr + "}";
  }

  DescCtor = DescCtor + ")";
  CallExprRepl = DescCtor;

  std::string SetStr = DescrMemberCallPrefix + "set_value";
  if (PlacementFromExec != FFTPlacementType::Inplace) {
    DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                             Diagnostics::OUT_OF_PLACE_FFT_EXEC, true, false);
    SuffixStmts.emplace_back(SetStr +
                             "(oneapi::mkl::dft::config_param::PLACEMENT, "
                             "DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);");
  }
}

void FFTDescriptorTypeInfo::buildInfo(std::string FilePath,
                                      unsigned int Offset) {
  if (SkipGeneration)
    return;

  if (!PrecAndDom.empty() && IsValid) {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            FilePath, Offset, Length,
            "std::shared_ptr<oneapi::mkl::dft::descriptor<" + PrecAndDom + ">>",
            nullptr));
    return;
  }
  if (DpctGlobalInfo::getPrecAndDomPairSet().size() == 1) {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            FilePath, Offset, Length,
            "std::shared_ptr<oneapi::mkl::dft::descriptor<" +
                *DpctGlobalInfo::getPrecAndDomPairSet().begin() + ">>",
            nullptr));
  } else {
    DiagnosticsUtils::report(FilePath, Offset, Diagnostics::UNDEDUCED_TYPE,
                             true, false, "FFT precision and domain type");
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            FilePath, Offset, Length,
            "std::shared_ptr<oneapi::mkl::dft::descriptor<dpct_placeholder/"
            "*Fix the precision and domain type manually*/>>",
            nullptr));
  }
}

void FFTExecAPIInfo::addInfo(std::string IndentStrInput,
                             LibraryMigrationFlags FlagsInput,
                             std::string PrePrefixStmtInput,
                             std::vector<std::string> PrefixStmtsInput,
                             std::vector<std::string> SuffixStmtsInput,
                             std::string CallExprReplInput,
                             bool IsComplexDomainInput,
                             std::string DescStrInput, std::int64_t DirInput) {
  Flags = FlagsInput;
  IndentStr = IndentStrInput;
  PrePrefixStmt = PrePrefixStmtInput;
  PrefixStmts = PrefixStmtsInput;
  SuffixStmts = SuffixStmtsInput;
  CallExprRepl = CallExprReplInput;
  IsComplexDomain = IsComplexDomainInput;
  DescStr = DescStrInput;
  Dir = DirInput;
}

void FFTExecAPIInfo::updateResetAndCommitStmts() {
  if (NeedReset) {
    DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                             Diagnostics::ONLY_SUPPORT_SAME_DISTANCE, true,
                             false, "the FWD_DISTANCE and the BWD_DISTANCE");
    ResetAndCommitStmts.emplace_back("if (" + InembedStr + " != nullptr && " +
                                     OnembedStr + " != nullptr) {");
    if (Dir == -1) {
      ResetAndCommitStmts.emplace_back(
          DescStr +
          "->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
          InputDistance + ");");
      ResetAndCommitStmts.emplace_back(
          DescStr +
          "->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
          OutputDistance + ");");
    } else {
      ResetAndCommitStmts.emplace_back(
          DescStr +
          "->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
          InputDistance + ");");
      ResetAndCommitStmts.emplace_back(
          DescStr +
          "->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
          OutputDistance + ");");
    }

    ResetAndCommitStmts.emplace_back("}");
  }

  if (!Flags.IsFunctionPointer && !Flags.IsFunctionPointerAssignment) {
    std::string Stream;

    if (!DefiniteStream.empty()) {
      Stream = DefiniteStream;
    } else {
      if (!StreamStr.empty())
        Stream = StreamStr;
      else {
        if (QueueIndex == -1) {
          Stream = MapNames::getDpctNamespace() + "get_default_queue()";
          requestFeature(HelperFeatureEnum::Device_get_default_queue, FilePath);
        } else
          Stream = "{{NEEDREPLACEQ" + std::to_string(QueueIndex) + "}}";
      }

      if (DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                                   Diagnostics::CHECK_RELATED_QUEUE, false,
                                   false, Stream)) {
        ResetAndCommitStmts.push_back("/*");
        ResetAndCommitStmts.push_back(
            DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
                Diagnostics::CHECK_RELATED_QUEUE));
        ResetAndCommitStmts.push_back("*/");
      }
    }

    ResetAndCommitStmts.emplace_back(DescStr + "->commit(" + Stream + ");");
  }
  PrefixStmts.insert(PrefixStmts.begin(), ResetAndCommitStmts.begin(),
                     ResetAndCommitStmts.end());
}

void FFTExecAPIInfo::linkInfo() {
  if (!Flags.IsFunctionPointer && !Flags.IsFunctionPointerAssignment) {
    auto &Map = DpctGlobalInfo::getFFTHandleInfoMap();
    auto I = Map.find(HandleDeclFileAndOffset);
    if (I != Map.end()) {
      InputDistance = I->second.InputDistance;
      OutputDistance = I->second.OutputDistance;
      InembedStr = I->second.InembedStr;
      OnembedStr = I->second.OnembedStr;
      NeedReset = I->second.MayNeedReset && IsComplexDomain &&
                  (I->second.Direction == FFTDirectionType::Uninitialized ||
                   I->second.Direction == FFTDirectionType::Unknown);
    }

    if (CompoundStmtBeginOffset && PlanHandleDeclBeginOffset &&
        ExecAPIBeginOffset)
      StreamStr = DpctGlobalInfo::getInstance().getRelatedFFTStream(
          FilePath, CompoundStmtBeginOffset, PlanHandleDeclBeginOffset,
          ExecAPIBeginOffset);
  } else {
    NeedReset = false;
  }
}

void FFTExecAPIInfo::buildInfo() {
  linkInfo();
  updateResetAndCommitStmts();
  replacementText(Flags, PrePrefixStmt, PrefixStmts, SuffixStmts, CallExprRepl,
                  IndentStr, FilePath, ReplaceOffset, ReplaceLen,
                  InsertOffsets);
}

/// This function will check the statemates before \p ExecCall.
/// If there is no flow control statements between previous cufftSetStream()
/// call and current cufftExec() call, this funcion will return true and
/// \p StreamStr will be set as the value in cufftSetStream().
/// Else, this funcion will return false.
bool isPreviousStmtRelatedSetStream(const CallExpr *ExecCall, int Index,
                                    std::string &StreamStr) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  const CompoundStmt *CS =
      dyn_cast_or_null<CompoundStmt>(getParentStmt(ExecCall));
  if (!CS)
    return false;

  // Step 1: Find all cufftSetStream() call in current CompoundStmt
  auto SetStreamCallMatcher = ast_matchers::findAll(
      ast_matchers::callExpr(ast_matchers::callee(ast_matchers::functionDecl(
                                 ast_matchers::hasName("cufftSetStream"))))
          .bind("FunctionCall"));
  auto MatchedResults = ast_matchers::match(SetStreamCallMatcher, *CS,
                                            DpctGlobalInfo::getContext());
  std::vector<const CallExpr *> Calls;
  for (auto &Result : MatchedResults) {
    const CallExpr *Call = Result.getNodeAs<CallExpr>("FunctionCall");
    if (!Call)
      continue;
    Calls.push_back(Call);
  }

  // Step 2: Find all assignment of the handle var of ExecCall in current
  // CompoundStmt
  const auto HandleDecl = getHandleVar(ExecCall->getArg(0));
  std::vector<const DeclRefExpr *> Refs;
  findAssignments(HandleDecl, CS, Refs);

  if (Refs.empty())
    return false;

  // Step 3: Check the cufftSetStream() just before current cufftExec() call
  unsigned int ExecCallOffset =
      SM.getExpansionLoc(ExecCall->getBeginLoc()).getRawEncoding();
  const CallExpr *LastSetStreamCall = nullptr;
  bool IsLastSetStreamCallDetermined = false;

  for (std::vector<const CallExpr *>::reverse_iterator Iter = Calls.rbegin();
       Iter != Calls.rend(); ++Iter) {
    const auto Call = *Iter;

    if (SM.getExpansionLoc(Call->getBeginLoc()).getRawEncoding() >
        ExecCallOffset)
      continue;

    // The handle of these two APIs must be same
    if (getHandleVar(Call->getArg(0)) &&
        (getHandleVar(Call->getArg(0)) == HandleDecl)) {
      LastSetStreamCall = Call;
      // If cufftSetStream() is in control flow statements, then need emit
      // warning
      if (isInCtrlFlowStmt(Call, CS, DpctGlobalInfo::getContext())) {
        IsLastSetStreamCallDetermined = false;
      } else {
        bool NewHandleVarAssigned = false;
        for (const auto &Ref : Refs) {
          unsigned int RefOffset =
              SM.getExpansionLoc(Ref->getBeginLoc()).getRawEncoding();
          unsigned int SetStreamCallOffset =
              SM.getExpansionLoc(Call->getEndLoc()).getRawEncoding();
          if (RefOffset < ExecCallOffset && RefOffset > SetStreamCallOffset) {
            NewHandleVarAssigned = true;
            break;
          }
        }
        if (NewHandleVarAssigned)
          IsLastSetStreamCallDetermined = false;
        else
          IsLastSetStreamCallDetermined = true;
      }
    } else {
      IsLastSetStreamCallDetermined = false;
    }
    break;
  }

  if (!IsLastSetStreamCallDetermined || !LastSetStreamCall)
    return false;

  // Step4: using the stream which is set in the last cufftSetStream()
  if (isDefaultStream(LastSetStreamCall->getArg(1))) {
    if (Index == -1) {
      StreamStr = MapNames::getDpctNamespace() + "get_default_queue()";
      requestFeature(HelperFeatureEnum::Device_get_default_queue, ExecCall);
    } else {
      StreamStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
    }
  } else {
    StreamStr = "*" + ExprAnalysis::ref(LastSetStreamCall->getArg(1));
  }

  return true;
}

} // namespace dpct
} // namespace clang
