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
    unsigned int DescIdx, std::string PrecAndDomainStr) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  const DeclaratorDecl *HandleVar = getHandleVar(TheCallExpr->getArg(DescIdx));
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
  unsigned int TypeLength = Lexer::MeasureTokenLength(TypeBeginLoc, SM,
                                    DpctGlobalInfo::getContext().getLangOpts());

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
    std::string Prec, Domain;
    if (Value == 0x2a || Value == 0x2c) {
      Prec = "oneapi::mkl::dft::precision::SINGLE";
      Domain = "oneapi::mkl::dft::domain::REAL";
    } else if (Value == 0x29) {
      Prec = "oneapi::mkl::dft::precision::SINGLE";
      Domain = "oneapi::mkl::dft::domain::COMPLEX";
    } else if (Value == 0x6a || Value == 0x6c) {
      Prec = "oneapi::mkl::dft::precision::DOUBLE";
      Domain = "oneapi::mkl::dft::domain::REAL";
    } else {
      Prec = "oneapi::mkl::dft::precision::DOUBLE";
      Domain = "oneapi::mkl::dft::domain::COMPLEX";
    }
    PrecAndDomain = Prec + ", " + Domain;
  }
  return PrecAndDomain;
}

/// Add the set_value call for 1d plan
void FFTFunctionCallBuilder::setValueFor1DBatched(unsigned int DescIdx,
                                                  unsigned int SizeIdx,
                                                  unsigned int BatchIdx) {
  Expr::EvalResult ER;
  if (!TheCallExpr->getArg(BatchIdx)->isValueDependent() &&
      TheCallExpr->getArg(BatchIdx)
          ->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
    int64_t Value = ER.Val.getInt().getExtValue();
    if (Value == 1) {
      return;
    }
  }

  std::string SetStr = getDescrMemberCallPrefix() + "set_value";
  PrefixStmts.emplace_back(SetStr +
                           "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                           ArgsList[SizeIdx] + ");");
  PrefixStmts.emplace_back(SetStr +
                           "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                           ArgsList[SizeIdx] + ");");
  PrefixStmts.emplace_back(
      SetStr + "(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, " +
      ArgsList[BatchIdx] + ");");
}

/// Add set_value call for many plan, collect dim info and then call
/// updateCommitCallExpr
void FFTFunctionCallBuilder::updateManyCommitCallExpr(int QueueIndex) {
  // TODO: Add if-stmt to check whether inembed and onembed are NULL.
  // TODO: It will be submitted in another patch.
  Expr::EvalResult ER;
  std::vector<std::string> Dims;
  std::string InStridesStr;
  std::string OutStridesStr;
  int64_t Value = -1;
  if (!TheCallExpr->getArg(1)->isValueDependent() &&
      TheCallExpr->getArg(1)->EvaluateAsInt(
          ER, DpctGlobalInfo::getContext())) {
    Value = ER.Val.getInt().getExtValue();
    // dim = 3:
    // ios[3]=istride*inembed[2]*inembed[1]
    // ios[2]=istride*inembed[2]
    // ios[1]=istride
    // ios[0]=0
    //
    // dim = 2:
    // ios[2]=istride*inembed[1]
    // ios[1]=istride
    // ios[0]=0
    //
    // dim = 1:
    // ios[1]=istride
    // ios[0]=0
    std::vector<std::string> InStrides;
    std::vector<std::string> OutStrides;
    InStrides.emplace_back("0");
    OutStrides.emplace_back("0");
    InStrides.emplace_back(ArgsList[4]);
    OutStrides.emplace_back(ArgsList[7]);
    if (Value == 2) {
      InStrides.emplace_back(ArgsList[3] + "[1] * " + ArgsList[4]);
      OutStrides.emplace_back(ArgsList[6] + "[1] * " + ArgsList[7]);
    } else if (Value == 3) {
      InStrides.emplace_back(ArgsList[3] + "[2] * " + ArgsList[4]);
      OutStrides.emplace_back(ArgsList[6] + "[2] * " + ArgsList[7]);
      InStrides.emplace_back(ArgsList[3] + "[2] * " + ArgsList[3] + "[1] * " +
                             ArgsList[4]);
      OutStrides.emplace_back(ArgsList[6] + "[2] * " + ArgsList[6] + "[1] * " +
                              ArgsList[7]);
    }
    for (int64_t i = 0; i < Value; ++i)
      Dims.emplace_back(ArgsList[2] + "[" + std::to_string(i) + "]");
    for (size_t i = 0; i < InStrides.size(); ++i) {
      InStridesStr = InStridesStr + InStrides[i] + ", ";
      OutStridesStr = OutStridesStr + OutStrides[i] + ", ";
    }
    InStridesStr = InStridesStr.substr(0, InStridesStr.size() - 2);
    OutStridesStr = OutStridesStr.substr(0, OutStridesStr.size() - 2);
    InStridesStr = "std::array<std::int64_t, " + std::to_string(Value + 1) +
                   ">{" + InStridesStr + "}";
    OutStridesStr = "std::array<std::int64_t, " + std::to_string(Value + 1) +
                    ">{" + OutStridesStr + "}";
  } else {
    report(Locations.PrefixInsertLoc, Diagnostics::UNDEDUCED_PARAM, false,
           "dimensions and strides");
    Dims.emplace_back("dpct_placeholder/*Fix the dimensions manually*/");
    InStridesStr = "dpct_placeholder/*Fix the stride manually*/";
    OutStridesStr = "dpct_placeholder/*Fix the stride manually*/";
  }

  updateCommitCallExpr(0, Dims, 9, QueueIndex);

  std::string SetStr = getDescrMemberCallPrefix() + "set_value";

  report(Locations.PrefixInsertLoc, Diagnostics::ONLY_SUPPORT_SAME_DISTANCE, false);
  PrefixStmts.emplace_back(SetStr +
                           "(oneapi::mkl::dft::config_param::FWD_DISTANCE, " +
                           ArgsList[5] + ");");
  PrefixStmts.emplace_back(SetStr +
                           "(oneapi::mkl::dft::config_param::BWD_DISTANCE, " +
                           ArgsList[5] + ");");
  PrefixStmts.emplace_back(
      SetStr + "(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, " +
      ArgsList[10] + ");");

  PrefixStmts.emplace_back(SetStr +
                           "(oneapi::mkl::dft::config_param::INPUT_STRIDES, " +
                           InStridesStr + ");");
  PrefixStmts.emplace_back(SetStr +
                           "(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, " +
                           OutStridesStr + ");");

}

/// collect dim info then call updateCommitCallExpr
void FFTFunctionCallBuilder::update1D2D3DCommitCallExpr(
    unsigned int DescIdx, std::vector<int> DimIdxs, unsigned int PrecDomainIdx,
    int QueueIndex) {
  std::vector<std::string> Dims;
  for (const auto &Idx : DimIdxs)
    Dims.emplace_back(ArgsList[Idx]);
  updateCommitCallExpr(DescIdx, Dims, PrecDomainIdx, QueueIndex);
}

/// generates  stmts: construct a descriptor and commit method
void FFTFunctionCallBuilder::updateCommitCallExpr(unsigned int DescIdx,
                                                  std::vector<std::string> Dims,
                                                  unsigned int PrecDomainIdx,
                                                  int QueueIndex) {
  if (FuncName.startswith("cufftMake")) {
    report(Locations.PrefixInsertLoc, Diagnostics::UNSUPPORTED_PARAM, false,
        ExprAnalysis::ref(TheCallExpr->getArg(TheCallExpr->getNumArgs() - 1)));
  }

  std::string PrecAndDomainStr = getPrecAndDomainStr(PrecDomainIdx);
  addDescriptorTypeInfo(DescIdx, PrecAndDomainStr);
  if (PrecAndDomainStr.empty()) {
    report(Locations.PrefixInsertLoc, Diagnostics::UNDEDUCED_TYPE, false,
           FuncName, "FFT precision and domain type");
    PrecAndDomainStr =
        "dpct_placeholder/*Fix the precision and domain type manually*/";
  } else {
    DpctGlobalInfo::getPrecAndDomPairSet().insert(PrecAndDomainStr);
  }

  std::string DescStr = getDescr();
  std::string DescMethod = getDescrMemberCallPrefix();

  CallExprRepl = CallExprRepl + DescMethod + "commit({{NEEDREPLACEQ" +
                 std::to_string(QueueIndex) + "}})";

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

  DescCtor = DescCtor + ");";
  PrefixStmts.emplace_back(DescCtor);
}

/// Add buffer decl
void FFTFunctionCallBuilder::updateBufferArgs(unsigned int Idx,
                                              const std::string &TypeStr,
                                              std::string ExtraIndent) {
  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
    std::string BufferDecl;
    std::string PointerName = dpct::ExprAnalysis::ref(TheCallExpr->getArg(Idx));
    ArgsList[Idx] =
        getTempNameForExpr(TheCallExpr->getArg(Idx), true, true) + "buf_ct" +
        std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
    BufferDecl = ExtraIndent + "auto " + ArgsList[Idx] +
                 " = dpct::get_buffer<" + TypeStr + ">(" + PointerName + ");";
    PrefixStmts.emplace_back(BufferDecl);
  }
}

/// build C2C, Z2Z exec API info
void FFTFunctionCallBuilder::assembleExecCallExpr(const Expr *DirExpr,
                                                  int Index) {
  Expr::EvalResult ER;
  if (!DirExpr->isValueDependent() &&
      DirExpr->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
    int64_t Dir = ER.Val.getInt().getExtValue();
    assembleExecCallExpr(Dir, Index);
  } else {
    // TODO: handle this case. It will be submitted in another patch.
  }
}

/// build CRC, R2C,D2Z, Z2D exec API info
void FFTFunctionCallBuilder::assembleExecCallExpr(int64_t Dir, int Index) {
  // For USM-none, the size of the data need copy is known (from virutal ptr
  // info) For USM, the size maybe unknown. If info can be linked with
  // descriptor, it is known.
  auto getType = [=](const char C) -> std::string {
    if (C == 'Z' || C == 'D')
      return "double";
    else
      return "float";
  };

  std::string ComputeAPI;
  if (Dir == -1) {
    ComputeAPI = "oneapi::mkl::dft::compute_forward";
  } else if (Dir == 1) {
    ComputeAPI = "oneapi::mkl::dft::compute_backward";
  }

  std::string DescMethod = getDescrMemberCallPrefix();
  std::string OriginalInputPtr = ArgsList[1];
  std::string OriginalOutputPtr = ArgsList[2];
  updateBufferArgs(1, getType(FuncName[9]));

  PrefixStmts.emplace_back("if ((void *)" + OriginalInputPtr + " == (void *)" +
                           OriginalOutputPtr + ") {");
  PrefixStmts.emplace_back(ComputeAPI + "(" +
                           getDrefName(TheCallExpr->getArg(0)) + ", " +
                           ArgsList[1] + ");");
  PrefixStmts.emplace_back("} else {");
  PrefixStmts.emplace_back(DescMethod +
                           "set_value(oneapi::mkl::dft::config_param::"
                           "PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);");
  PrefixStmts.emplace_back(DescMethod + "commit({{NEEDREPLACEQ" +
                           std::to_string(Index) + "}});");
  updateBufferArgs(2, getType(FuncName[11]), "  ");
  CallExprRepl = ComputeAPI + "(" + getDrefName(TheCallExpr->getArg(0)) +
                 ", " + ArgsList[1] + ", " + ArgsList[2] + ")";
  SuffixStmts.emplace_back("}");
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
  auto &SM = DpctGlobalInfo::getSourceManager();
  // Now this function only covers this pattern:
  // cufftResult R = cufftAPI();

  std::shared_ptr<ast_type_traits::DynTypedNode> P =
      std::make_shared<ast_type_traits::DynTypedNode>(
          ast_type_traits::DynTypedNode::create(*TheCallExpr));
  const VarDecl *VD = getNonImplicitCastParentNode(P)->get<VarDecl>();
  if (!VD)
    return false;
  if (VD->getInitStyle() != VarDecl::InitializationStyle::CInit)
    return false;

  auto NeedMove = [&]() -> bool { const Stmt *S = getParentStmt(VD);
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
  TypeBegin =
      VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
  SourceLocation TypeEnd =
      VD->getTypeSourceInfo()->getTypeLoc().getEndLoc();
  TypeEnd = TypeEnd.getLocWithOffset(
      Lexer::MeasureTokenLength(SM.getExpansionLoc(TypeEnd), SM,
                                DpctGlobalInfo::getContext().getLangOpts()));

  auto C = SM.getCharacterData(TypeEnd);
  int Offset = 0;
  while (C && isblank (*C)) {
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

void initVars(const CallExpr *CE, LibraryMigrationFlags &Flags,
              LibraryMigrationStrings &ReplaceStrs,
              LibraryMigrationLocations &Locations) {
  auto &SM = DpctGlobalInfo::getSourceManager();

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

} // namespace dpct
} // namespace clang
