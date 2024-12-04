//===--------------- FFTAPIMigration.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FFTAPIMigration.h"
#include "RuleInfra/ExprAnalysis.h"

using namespace clang::ast_matchers;

namespace clang {
namespace dpct {

TextModification* processFunctionPointer(const UnaryOperator *UO) {
  if (!UO)
    return nullptr;
  const DeclRefExpr *DRE = dyn_cast_or_null<DeclRefExpr>(UO->getSubExpr());
  if (!DRE)
    return nullptr;
  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(DRE->getDecl());
  if (!FD)
    return nullptr;
  StringRef FuncNameRef = FD->getName();
  std::string ParameterTypes = MapNames::getLibraryHelperNamespace() +
                               "fft::fft_engine_ptr engine";
  requestFeature(HelperFeatureEnum::device_ext);
  std::string Dir;
  std::string NewFuncName;
  if (FuncNameRef == "cufftExecC2C") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "float2 *in, " + MapNames::getClNamespace() +
                     "float2 *out, " + MapNames::getLibraryHelperNamespace() +
                     "fft::fft_direction dir";
    Dir = "dir";
    NewFuncName = "compute<" + MapNames::getClNamespace() + "float2, " +
                  MapNames::getClNamespace() + "float2>";
  } else if (FuncNameRef == "cufftExecZ2Z") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "double2 *in, " + MapNames::getClNamespace() +
                     "double2 *out, " + MapNames::getLibraryHelperNamespace() +
                     "fft::fft_direction dir";
    Dir = "dir";
    NewFuncName = "compute<" + MapNames::getClNamespace() + "double2, " +
                  MapNames::getClNamespace() + "double2>";
  } else if (FuncNameRef == "cufftExecR2C") {
    ParameterTypes = ParameterTypes + ", float *in, " +
                     MapNames::getClNamespace() + "float2 *out";
    Dir = MapNames::getLibraryHelperNamespace() + "fft::fft_direction::forward";
    NewFuncName = "compute<float, " + MapNames::getClNamespace() + "float2>";
  } else if (FuncNameRef == "cufftExecC2R") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "float2 *in, float *out";
    Dir = MapNames::getLibraryHelperNamespace() + "fft::fft_direction::backward";
    NewFuncName = "compute<" + MapNames::getClNamespace() + "float2, float>";
  } else if (FuncNameRef == "cufftExecD2Z") {
    ParameterTypes = ParameterTypes + ", double *in, " +
                     MapNames::getClNamespace() + "double2 *out";
    Dir = MapNames::getLibraryHelperNamespace() + "fft::fft_direction::forward";
    NewFuncName = "compute<double, " + MapNames::getClNamespace() + "double2>";
  } else if (FuncNameRef == "cufftExecZ2D") {
    ParameterTypes = ParameterTypes + ", " + MapNames::getClNamespace() +
                     "double2 *in, double *out";
    Dir = MapNames::getLibraryHelperNamespace() + "fft::fft_direction::backward";
    NewFuncName = "compute<" + MapNames::getClNamespace() + "double2, double>";
  } else {
    return nullptr;
  }
  std::string ReplStr = "[](" + ParameterTypes + "){" + getNL() + "  engine->" +
                        NewFuncName + "(in, out, " + Dir + ");" + getNL() +
                        "  return 0;" + getNL() + "}";
  ReplaceStmt *TM = new ReplaceStmt(UO, ReplStr);
  TM->setBlockLevelFormatFlag(true);
  return TM;
}

// Rule for FFT function calls.
void FFTFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName("cufftPlan1d", "cufftPlan2d", "cufftPlan3d",
                      "cufftPlanMany", "cufftMakePlan1d", "cufftMakePlan2d",
                      "cufftMakePlan3d", "cufftMakePlanMany",
                      "cufftMakePlanMany64", "cufftExecC2C", "cufftExecR2C",
                      "cufftExecC2R", "cufftExecZ2Z", "cufftExecZ2D",
                      "cufftExecD2Z", "cufftCreate", "cufftDestroy",
                      "cufftSetStream", "cufftGetVersion", "cufftGetProperty",
                      "cufftXtMakePlanMany", "cufftXtExec", "cufftGetSize1d",
                      "cufftGetSize2d", "cufftGetSize3d", "cufftGetSizeMany",
                      "cufftGetSize", "cufftEstimate1d", "cufftEstimate2d",
                      "cufftEstimate3d", "cufftEstimateMany",
                      "cufftSetAutoAllocation", "cufftGetSizeMany64",
                      "cufftSetWorkArea");
  };
  MF.addMatcher(callExpr(callee(functionDecl(functionName()))).bind("FuncCall"),
                this);

  // Currently, only exec functions support function pointer migration
  auto execFunctionName = [&]() {
    return hasAnyName("cufftExecC2C", "cufftExecR2C", "cufftExecC2R",
                      "cufftExecZ2Z", "cufftExecZ2D", "cufftExecD2Z");
  };
  MF.addMatcher(unaryOperator(hasOperatorName("&"),
                              hasUnaryOperand(declRefExpr(hasDeclaration(
                                  functionDecl(execFunctionName())))))
                    .bind("FuncPtr"),
                this);
}

void FFTFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FuncCall");
  const UnaryOperator *UO = getNodeAsType<UnaryOperator>(Result, "FuncPtr");

  if (!CE) {
    auto TM = processFunctionPointer(UO);
    if (TM) {
      emplaceTransformation(TM);
    }
    return;
  }

  auto &SM = DpctGlobalInfo::getSourceManager();
  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  if (FuncName == "cufftGetVersion" || FuncName == "cufftGetProperty") {
    DpctGlobalInfo::getInstance().insertHeader(
        SM.getExpansionLoc(CE->getBeginLoc()), HT_DPCT_COMMON_Utils);
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  } else if (FuncName == "cufftSetStream" ||
             FuncName == "cufftCreate" || FuncName == "cufftDestroy" ||
             FuncName == "cufftPlan1d" || FuncName == "cufftMakePlan1d" ||
             FuncName == "cufftPlan2d" || FuncName == "cufftMakePlan2d" ||
             FuncName == "cufftPlan3d" || FuncName == "cufftMakePlan3d" ||
             FuncName == "cufftPlanMany" || FuncName == "cufftMakePlanMany" ||
             FuncName == "cufftMakePlanMany64" || FuncName == "cufftXtMakePlanMany" ||
             FuncName == "cufftExecC2C" || FuncName == "cufftExecZ2Z" ||
             FuncName == "cufftExecC2R" || FuncName == "cufftExecR2C" ||
             FuncName == "cufftExecZ2D" || FuncName == "cufftExecD2Z" ||
             FuncName == "cufftXtExec" || FuncName == "cufftGetSize1d" ||
             FuncName == "cufftGetSize2d" || FuncName == "cufftGetSize3d" ||
             FuncName == "cufftGetSizeMany" || FuncName == "cufftGetSize" ||
             FuncName == "cufftEstimate1d" || FuncName == "cufftEstimate2d" ||
             FuncName == "cufftEstimate3d" || FuncName == "cufftEstimateMany" ||
             FuncName == "cufftSetAutoAllocation" || FuncName == "cufftGetSizeMany64" ||
             FuncName == "cufftSetWorkArea") {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
}

// Rule for FFT enums.
void FFTEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(
          to(enumConstantDecl(matchesName(
              "(CUFFT_SUCCESS|CUFFT_INVALID_PLAN|CUFFT_ALLOC_FAILED|CUFFT_"
              "INVALID_TYPE|CUFFT_INVALID_VALUE|CUFFT_INTERNAL_ERROR|CUFFT_"
              "EXEC_FAILED|CUFFT_SETUP_FAILED|CUFFT_INVALID_SIZE|CUFFT_"
              "UNALIGNED_DATA|CUFFT_INCOMPLETE_PARAMETER_LIST|CUFFT_INVALID_"
              "DEVICE|CUFFT_PARSE_ERROR|CUFFT_NO_WORKSPACE|CUFFT_NOT_"
              "IMPLEMENTED|CUFFT_LICENSE_ERROR|CUFFT_NOT_SUPPORTED)"))))
          .bind("FFTConstants"),
      this);
}

void FFTEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "FFTConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
    return;
  }
}

} // namespace dpct
} // namespace clang
