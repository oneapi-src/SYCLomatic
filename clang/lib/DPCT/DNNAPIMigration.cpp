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

using namespace clang;
using namespace clang::dpct;
using namespace clang::ast_matchers;

void CuDNNTypeRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      typeLoc(
          loc(qualType(hasDeclaration(namedDecl(hasAnyName(
              "cudnnHandle_t", "cudnnTensorDescriptor_t", "cudnnTensorFormat_t",
              "cudnnDataType_t", "cudnnActivationDescriptor_t",
              "cudnnActivationMode_t", "cudnnLRNDescriptor_t", "cudnnLRNMode_t",
              "cudnnPoolingDescriptor_t", "cudnnPoolingMode_t",
              "cudnnSoftmaxAlgorithm_t", "cudnnSoftmaxMode_t"))))))
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
  } else if (auto *E = getNodeAsType<DeclRefExpr>(Result, "CuDNNEnumConstant")) {
    if (!E)
      return;
    std::string EnumName = E->getNameInfo().getName().getAsString();
    if (EnumName == "CUDNN_DATA_DOUBLE") {
      report(E->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             "data type double");
      return;
    }
    auto Search = CuDNNEnumNamesMap.find(EnumName);
    if (Search == CuDNNEnumNamesMap.end()) {
      return;
    }

    emplaceTransformation(new ReplaceStmt(E, Search->second));
    requestHelperFeatureForEnumNames(EnumName, E);
  }
}

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
        "cudnnSoftmaxBackward", "cudnnSetTensor");
  };

  MF.addMatcher(callExpr(callee(functionDecl(CuDNNAPI()))).bind("call"), this);
}

void CuDNNAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call")) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

} // namespace dpct
} // namespace clang