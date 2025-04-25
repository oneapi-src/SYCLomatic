//===---------------------- CUTensorAPIMigration.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===//

#include "CUTensorAPIMigration.h"
#include "RuleInfra/ExprAnalysis.h"

using namespace clang::dpct;
using namespace clang::ast_matchers;

void clang::dpct::CUTensorRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto CutensorAPIs = [&]() {
    return hasAnyName(
        // Helper Functions
        "cutensorCreate", "cutensorDestroy", "cutensorCreateTensorDescriptor",
        "cutensorDestroyTensorDescriptor", "cutensorGetErrorString",
        "cutensorGetVersion", "cutensorGetCudartVersion",
        // Element-wise Operations
        "cutensorCreateElementwiseTrinary", "cutensorElementwiseTrinaryExecute",
        "cutensorCreateElementwiseBinary", "cutensorElementwiseBinaryExecute",
        "cutensorCreatePermutation", "cutensorPermute",
        // Contraction Operations
        "cutensorCreateContraction", "cutensorContract",
        "cutensorCreateContractionTrinary", "cutensorContractTrinary",
        // Reduction Operations
        "cutensorCreateReduction", "cutensorReduce",
        // Generic Operation Functions
        "cutensorDestroyOperationDescriptor",
        "cutensorOperationDescriptorGetAttribute",
        "cutensorOperationDescriptorSetAttribute",
        "cutensorCreatePlanPreference", "cutensorDestroyPlanPreference",
        "cutensorPlanPreferenceSetAttribute", "cutensorEstimateWorkspaceSize",
        "cutensorCreatePlan", "cutensorDestroyPlan", "cutensorPlanGetAttribute",
        // Cache-related Operations
        "cutensorHandleResizePlanCache", "cutensorHandleReadPlanCacheFromFile",
        "cutensorHandleWritePlanCacheToFile", "cutensorReadKernelCacheFromFile",
        "cutensorWriteKernelCacheToFile",
        // Logger Functions
        "cutensorLoggerSetCallback", "cutensorLoggerSetFile",
        "cutensorLoggerOpenFile", "cutensorLoggerSetLevel",
        "cutensorLoggerSetMask", "cutensorLoggerForceDisable",
        // cuTENSORMg - General Operations
        "cutensorMgCreate", "cutensorMgDestroy",
        "cutensorMgCreateTensorDescriptor", "cutensorMgDestroyTensorDescriptor",
        "cutensorMgCreateCopyDescriptor", "cutensorMgDestroyCopyDescriptor",
        "cutensorMgCopyGetWorkspace", "cutensorMgCreateCopyPlan",
        "cutensorMgDestroyCopyPlan", "cutensorMgCopy",
        // cuTENSORMg - Contraction Operations
        "cutensorMgCreateContractionDescriptor",
        "cutensorMgDestroyContractionDescriptor",
        "cutensorMgCreateContractionFind", "cutensorMgDestroyContractionFind",
        "cutensorMgContractionGetWorkspace", "cutensorMgCreateContractionPlan",
        "cutensorMgDestroyContractionPlan", "cutensorMgContraction");
  };

  MF.addMatcher(callExpr(callee(functionDecl(CutensorAPIs()))).bind("call"),
                this);
}

void clang::dpct::CUTensorRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call")) {
    std::string FuncName = "";
    const FunctionDecl *FD = CE->getDirectCallee();
    if (FD) {
      FuncName = FD->getNameInfo().getName().getAsString();
    }

    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, FuncName);
  }

  return;
}
