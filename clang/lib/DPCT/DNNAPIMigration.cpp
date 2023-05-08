//===---DNNAPIMigration.cpp -----------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "DNNAPIMigration.h"
#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "Statics.h"
#include "MapNames.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"

namespace clang {
namespace dpct {

using namespace clang::ast_matchers;

void CuDNNTypeRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      typeLoc(
          loc(qualType(hasDeclaration(namedDecl(hasAnyName(
              "cudnnHandle_t", "cudnnTensorDescriptor_t", "cudnnTensorFormat_t",
              "cudnnDataType_t", "cudnnActivationDescriptor_t",
              "cudnnActivationMode_t", "cudnnLRNDescriptor_t", "cudnnLRNMode_t",
              "cudnnPoolingDescriptor_t", "cudnnPoolingMode_t",
              "cudnnSoftmaxAlgorithm_t", "cudnnSoftmaxMode_t", "cudnnStatus_t",
              "cudnnReduceTensorDescriptor_t", "cudnnReduceTensorOp_t",
              "cudnnOpTensorDescriptor_t", "cudnnOpTensorOp_t",
              "cudnnBatchNormOps_t", "cudnnBatchNormMode_t", "cudnnNormMode_t",
              "cudnnNormOps_t", "cudnnConvolutionDescriptor_t",
              "cudnnConvolutionFwdAlgo_t", "cudnnConvolutionBwdDataAlgo_t",
              "cudnnConvolutionBwdFilterAlgo_t", "cudnnFilterDescriptor_t",
              "cudnnRNNMode_t", "cudnnRNNBiasMode_t", "cudnnDirectionMode_t",
              "cudnnRNNDescriptor_t", "cudnnForwardMode_t", "cudnnRNNDataDescriptor_t",
              "cudnnRNNDataLayout_t", "cudnnDropoutDescriptor_t"))))))
          .bind("CuDNNType"),
      this);
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName("CUDNN_.*"))))
                    .bind("CuDNNEnumConstant"),
                this);
}

void CuDNNTypeRule::runRule(const MatchFinder::MatchResult &Result) {
  SourceManager *SM = Result.SourceManager;
  auto LOpts = Result.Context->getLangOpts();
  if (auto TL = getNodeAsType<TypeLoc>(Result, "CuDNNType")) {

    auto TypeStr =
        DpctGlobalInfo::getTypeName(TL->getType().getUnqualifiedType());

    // typedef void* cudnnHandle_t;
    // cudnnHandle_t handle;
    // for this case, cudnnHandle_t should not be migrated.
    if (const clang::ElaboratedType *ET =
            llvm::dyn_cast<clang::ElaboratedType>(TL->getType())) {
      if (const clang::TypedefType *TDT =
              llvm::dyn_cast<clang::TypedefType>(ET->getNamedType().getTypePtr())) {
        if (DpctGlobalInfo::isInRoot(TDT->getDecl()->getBeginLoc())) {
          return;
        }
      }
    }

    if (!DpctGlobalInfo::isInAnalysisScope(SM->getSpellingLoc(TL->getBeginLoc()))) {
      return;
    }

    auto Range = getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc());
    auto BeginLoc = Range.getBegin();
    auto EndLoc = Range.getEnd();

    if (SM->isWrittenInScratchSpace(SM->getSpellingLoc(TL->getBeginLoc()))) {
      BeginLoc = SM->getExpansionRange(TL->getBeginLoc()).getBegin();
      EndLoc = SM->getExpansionRange(TL->getBeginLoc()).getEnd();
    }

    std::string Str =
        MapNames::findReplacedName(MapNames::CuDNNTypeNamesMap, TypeStr);
    if (!Str.empty()) {
      requestHelperFeatureForTypeNames(TypeStr, BeginLoc);
      SrcAPIStaticsMap[TypeStr]++;

      auto Len = Lexer::MeasureTokenLength(
          EndLoc, *SM, DpctGlobalInfo::getContext().getLangOpts());
      Len += SM->getDecomposedLoc(EndLoc).second -
             SM->getDecomposedLoc(BeginLoc).second;
      emplaceTransformation(new ReplaceText(BeginLoc, Len, std::move(Str)));
      return;
    }
  } else if (auto *E =
                 getNodeAsType<DeclRefExpr>(Result, "CuDNNEnumConstant")) {
    std::string EnumName = E->getNameInfo().getName().getAsString();

    if (EnumName.find("CUDNN_STATUS_") != std::string::npos) {
      if (auto EC = dyn_cast<EnumConstantDecl>(E->getDecl())) {
        std::string Repl = toString(EC->getInitVal(), 10);
        emplaceTransformation(new ReplaceStmt(E, Repl));
        return;
      }
    } else if(EnumName == "CUDNN_BATCHNORM_SPATIAL_PERSISTENT") {
      report(E->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, EnumName);
    }

    auto Search = CuDNNEnumNamesMap.find(EnumName);
    if (Search == CuDNNEnumNamesMap.end()) {
      report(E->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, EnumName);
      return;
    }

    emplaceTransformation(new ReplaceStmt(E, Search->second));
    requestHelperFeatureForEnumNames(EnumName, E);
  }
}

auto AssignedStmt = []() {
  return anyOf(hasParent(compoundStmt()), hasParent(forStmt()),
               hasParent(whileStmt()), hasParent(doStmt()),
               hasParent(ifStmt()));
};

void CuDNNAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto CuDNNAPI = [&]() {
    return hasAnyName(
        "cudnnCreate", "cudnnDestroy", "cudnnSetStream", "cudnnGetStream",
        "cudnnCreateTensorDescriptor", "cudnnSetTensor4dDescriptor",
        "cudnnSetTensor4dDescriptorEx", "cudnnSetTensorNdDescriptor",
        "cudnnSetTensorNdDescriptorEx", "cudnnDestroyTensorDescriptor",
        "cudnnGetTensor4dDescriptor", "cudnnGetTensorNdDescriptor",
        "cudnnGetTensorSizeInBytes", "cudnnTransformTensor", "cudnnScaleTensor",
        "cudnnAddTensor", "cudnnCreateActivationDescriptor",
        "cudnnDestroyActivationDescriptor", "cudnnSetActivationDescriptor",
        "cudnnSetActivationDescriptorSwishBeta",
        "cudnnGetActivationDescriptorSwishBeta", "cudnnGetActivationDescriptor",
        "cudnnActivationForward", "cudnnActivationBackward",
        "cudnnCreateLRNDescriptor", "cudnnDestroyLRNDescriptor",
        "cudnnSetLRNDescriptor", "cudnnGetLRNDescriptor",
        "cudnnLRNCrossChannelForward", "cudnnLRNCrossChannelBackward",
        "cudnnCreatePoolingDescriptor", "cudnnDestroyPoolingDescriptor",
        "cudnnSetPooling2dDescriptor", "cudnnSetPoolingNdDescriptor",
        "cudnnGetPooling2dDescriptor", "cudnnGetPooling2dForwardOutputDim",
        "cudnnGetPoolingNdDescriptor", "cudnnGetPoolingNdForwardOutputDim",
        "cudnnPoolingForward", "cudnnPoolingBackward", "cudnnSoftmaxForward",
        "cudnnSoftmaxBackward", "cudnnSetTensor",
        "cudnnCreateReduceTensorDescriptor",
        "cudnnDestroyReduceTensorDescriptor", "cudnnSetReduceTensorDescriptor",
        "cudnnSetReduceTensorDescriptor", "cudnnGetReduceTensorDescriptor",
        "cudnnGetReductionWorkspaceSize", "cudnnReduceTensor",
        "cudnnCreateOpTensorDescriptor", "cudnnDestroyOpTensorDescriptor",
        "cudnnGetOpTensorDescriptor", "cudnnSetOpTensorDescriptor",
        "cudnnOpTensor", "cudnnBatchNormalizationForwardInference",
        "cudnnBatchNormalizationForwardTraining",
        "cudnnBatchNormalizationForwardTrainingEx",
        "cudnnBatchNormalizationBackward", "cudnnBatchNormalizationBackwardEx",
        "cudnnDeriveBNTensorDescriptor",
        "cudnnGetBatchNormalizationBackwardExWorkspaceSize",
        "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize",
        "cudnnGetBatchNormalizationTrainingExReserveSpaceSize",
        "cudnnNormalizationForwardInference",
        "cudnnNormalizationForwardTraining", "cudnnNormalizationBackward",
        "cudnnDeriveNormTensorDescriptor",
        "cudnnGetNormalizationForwardTrainingWorkspaceSize",
        "cudnnGetNormalizationTrainingReserveSpaceSize",
        "cudnnCreateFilterDescriptor", "cudnnDestroyFilterDescriptor",
        "cudnnGetFilter4dDescriptor", "cudnnGetFilterNdDescriptor",
        "cudnnGetFilterSizeInBytes", "cudnnSetFilter4dDescriptor",
        "cudnnSetFilterNdDescriptor", "cudnnCreateConvolutionDescriptor",
        "cudnnDestroyConvolutionDescriptor", "cudnnGetConvolution2dDescriptor",
        "cudnnGetConvolution2dForwardOutputDim",
        "cudnnGetConvolutionGroupCount", "cudnnGetConvolutionNdDescriptor",
        "cudnnGetConvolutionNdForwardOutputDim",
        "cudnnSetConvolution2dDescriptor", "cudnnSetConvolutionGroupCount",
        "cudnnSetConvolutionNdDescriptor", "cudnnConvolutionForward",
        "cudnnConvolutionBackwardData", "cudnnConvolutionBiasActivationForward",
        "cudnnConvolutionBackwardBias", "cudnnConvolutionBackwardFilter",
        "cudnnGetConvolutionForwardWorkspaceSize", "cudnnGetConvolutionBackwardDataWorkspaceSize",
        "cudnnGetConvolutionBackwardFilterWorkspaceSize",
        "cudnnGetNormalizationBackwardWorkspaceSize",
        "cudnnCreateRNNDescriptor", "cudnnCreateRNNDataDescriptor", "cudnnDestroyRNNDescriptor",
        "cudnnDestroyRNNDataDescriptor", "cudnnSetRNNDataDescriptor", "cudnnGetRNNDataDescriptor",
        "cudnnSetRNNDescriptor_v8", "cudnnGetRNNDescriptor_v8", "cudnnGetRNNWeightSpaceSize",
        "cudnnGetRNNTempSpaceSizes", "cudnnRNNForward", "cudnnRNNBackwardData_v8",
        "cudnnRNNBackwardWeights_v8", "cudnnDropoutGetStatesSize",
        "cudnnCreateDropoutDescriptor", "cudnnSetDropoutDescriptor",
        "cudnnGetDropoutDescriptor", "cudnnDropoutGetReserveSpaceSize",
        "cudnnRestoreDropoutDescriptor", "cudnnDropoutForward", "cudnnDropoutBackward",
        "cudnnDestroyDropoutDescriptor", "cudnnGetVersion",
        "cudnnGetConvolutionBackwardFilterAlgorithm",
        "cudnnGetConvolutionBackwardDataAlgorithm",
        "cudnnGetConvolutionForwardAlgorithm");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(CuDNNAPI())), AssignedStmt()))
          .bind("Call"),
      this);
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(CuDNNAPI())), unless(AssignedStmt())))
          .bind("AssignedCall"),
      this);
}

void CuDNNAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "Call");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "AssignedCall")))
      return;
    IsAssigned = true;
  }
  llvm::StringRef FuncName;
  if (auto DC = CE->getDirectCallee()) {
    FuncName = DC->getName();
  }
  if (FuncName == "cudnnRNNBackwardData_v8" ||
      FuncName == "cudnnRNNBackwardWeights_v8") {
    RnnBackwardFuncInfo FuncInfo;
    auto &Global = DpctGlobalInfo::getInstance();
    // 1.RnnDescIndex, 2.HXDataIndex, 3.YDataIndex
    unsigned RnnInputArgIndex[3] = {1, 9, 4};
    auto CERange = getDefinitionRange(CE->getBeginLoc(), CE->getEndLoc());
    auto CELocInfo = Global.getLocInfo(CERange.getBegin());
    auto FileInfo = Global.insertFile(CELocInfo.first);
    auto &BackwardFuncInfo = FileInfo->getRnnBackwardFuncInfo();
    auto &RnnInputMap = DpctGlobalInfo::getRnnInputMap();
    unsigned BeginOffset = CELocInfo.second;
    unsigned EndOffset = Global.getLocInfo(CERange.getEnd()).second;
    FuncInfo.isAssigned = IsAssigned;
    FuncInfo.Length = EndOffset - BeginOffset + 1;
    FuncInfo.FilePath = CELocInfo.first;
    FuncInfo.Offset = CELocInfo.second;
    FuncInfo.isDataGradient = true;
    unsigned int ArgsNum = CE->getNumArgs();
    auto Condition = [&](const clang::DynTypedNode &Node) -> bool {
      return Node.get<IfStmt>() || Node.get<WhileStmt>() ||
             Node.get<ForStmt>() || Node.get<DoStmt>() ||
             Node.get<CaseStmt>() || Node.get<SwitchStmt>() ||
             Node.get<CompoundStmt>();
    };

    if (auto CS = DpctGlobalInfo::findAncestor<CompoundStmt>(CE, Condition)) {
      auto LocInfo = Global.getLocInfo(CS->getBeginLoc());
      FuncInfo.CompoundLoc = LocInfo.first + std::to_string(LocInfo.second);
    } else {
      report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, FuncName);
      return;
    }
    if (FuncName == "cudnnRNNBackwardWeights_v8") {
      RnnInputArgIndex[1] = 7;
      RnnInputArgIndex[2] = 9;
      FuncInfo.isDataGradient = false;
      if (!IsAssigned) {
        auto Tok =
            Lexer::findNextToken(CERange.getEnd(), Global.getSourceManager(),
                                 LangOptions())
                .value();
        if (Tok.is(tok::TokenKind::semi)) {
          FuncInfo.Length +=
              Global.getLocInfo(Tok.getLocation()).second - EndOffset;
        }
      }
    }
    for(int i = 0; i < 3; i++) {
      if (auto RnnInputDRE = dyn_cast<DeclRefExpr>(
              CE->getArg(RnnInputArgIndex[i])->IgnoreImplicitAsWritten())) {
        if (auto RnnInputDecl = RnnInputDRE->getDecl()) {
          auto ArgName = RnnInputDecl->getName();
          auto DeclLocInfo = Global.getLocInfo(RnnInputDecl->getBeginLoc());
          std::string MapKey =
              DeclLocInfo.first + std::to_string(DeclLocInfo.second) + ArgName.str();
          auto &SubMap = RnnInputMap[MapKey];
          if (SubMap.empty()) {
            std::vector<const clang::DeclRefExpr *> MatchResults;
            findAllVarRef(RnnInputDRE, MatchResults, true);
            for (auto Result : MatchResults) {
              auto DRELocInfo = Global.getLocInfo(Result->getBeginLoc());
              SubMap[DRELocInfo.first].push_back(DRELocInfo.second);
            }
          }
          FuncInfo.RnnInputDeclLoc.push_back(std::move(MapKey));
        }
      }
    }
    if (FuncName == "cudnnRNNBackwardData_v8") {
      FuncInfo.FuncArgs.reserve(ArgsNum);
      for (unsigned int ArgIndex = 0; ArgIndex < ArgsNum; ArgIndex++) {
        auto Arg = CE->getArg(ArgIndex)->IgnoreImplicitAsWritten();
        ExprAnalysis ArgEA(Arg);
        ArgEA.analyze();
        std::string ArgString = ArgEA.getReplacedString();
        FuncInfo.FuncArgs.push_back(ArgString);
      }
    } else {
      FuncInfo.FuncArgs.reserve(2);
      ExprAnalysis ArgEA;
      ArgEA.analyze(CE->getArg(5));
      FuncInfo.FuncArgs.push_back(ArgEA.getReplacedString());
      ArgEA.analyze(CE->getArg(11));
      FuncInfo.FuncArgs.push_back(ArgEA.getReplacedString());
    }
    if (FuncInfo.FuncArgs.empty() || FuncInfo.RnnInputDeclLoc.empty() ||
        (FuncInfo.RnnInputDeclLoc.size() != 3)) {
      report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, FuncName);
      return;
    }
    BackwardFuncInfo.push_back(std::move(FuncInfo));
  } else {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

} // namespace dpct
} // namespace clang