//===------------------ SpBLASAPIMigration.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SpBLASAPIMigration.h"
#include "MapNamesBlas.h"
#include "RuleInfra/ASTmatcherCommon.h"
#include "RuleInfra/CallExprRewriter.h"
#include "RuleInfra/CallExprRewriterCommon.h"

namespace clang {
namespace dpct {

using namespace clang::ast_matchers;

void SpBLASTypeLocRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto TargetTypeName = [&]() {
    return hasAnyName("csrsv2Info_t", "cusparseSolvePolicy_t",
                      "cusparseAction_t");
  };

  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(TargetTypeName())))))
          .bind("loc"),
      this);
}

void SpBLASTypeLocRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto TL = getNodeAsType<TypeLoc>(Result, "loc")) {
    ExprAnalysis EA;
    EA.analyze(*TL);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

// Rule for spBLAS function calls.
void SPBLASFunctionCallRule::registerMatcher(MatchFinder &MF) {


  auto functionName = [&]() {
    return hasAnyName(
        /*management*/
        "cusparseCreate", "cusparseDestroy", "cusparseSetStream",
        "cusparseGetStream", "cusparseGetPointerMode", "cusparseSetPointerMode",
        "cusparseGetErrorName", "cusparseGetErrorString", "cusparseGetProperty",
        /*helper*/
        "cusparseCreateMatDescr", "cusparseDestroyMatDescr",
        "cusparseSetMatType", "cusparseGetMatType", "cusparseSetMatIndexBase",
        "cusparseGetMatIndexBase", "cusparseSetMatDiagType",
        "cusparseGetMatDiagType", "cusparseSetMatFillMode",
        "cusparseGetMatFillMode", "cusparseCreateSolveAnalysisInfo",
        "cusparseDestroySolveAnalysisInfo", "cusparseCreateCsrsv2Info",
        "cusparseDestroyCsrsv2Info",
        /*level 2*/
        "cusparseScsrmv", "cusparseDcsrmv", "cusparseCcsrmv", "cusparseZcsrmv",
        "cusparseScsrmv_mp", "cusparseDcsrmv_mp", "cusparseCcsrmv_mp",
        "cusparseZcsrmv_mp", "cusparseCsrmvEx", "cusparseCsrmvEx_bufferSize",
        "cusparseScsrsv_analysis", "cusparseDcsrsv_analysis",
        "cusparseCcsrsv_analysis", "cusparseZcsrsv_analysis",
        "cusparseScsrsv_solve", "cusparseDcsrsv_solve", "cusparseCcsrsv_solve",
        "cusparseZcsrsv_solve", "cusparseScsrsv2_bufferSize",
        "cusparseDcsrsv2_bufferSize", "cusparseCcsrsv2_bufferSize",
        "cusparseZcsrsv2_bufferSize", "cusparseScsrsv2_analysis",
        "cusparseDcsrsv2_analysis", "cusparseCcsrsv2_analysis",
        "cusparseZcsrsv2_analysis", "cusparseScsrsv2_solve",
        "cusparseDcsrsv2_solve", "cusparseCcsrsv2_solve",
        "cusparseZcsrsv2_solve", "cusparseCcsrsv2_bufferSizeExt",
        "cusparseDcsrsv2_bufferSizeExt", "cusparseScsrsv2_bufferSizeExt",
        "cusparseZcsrsv2_bufferSizeExt", "cusparseCsrsv_analysisEx",
        "cusparseCsrsv_solveEx",
        /*level 3*/
        "cusparseScsrmm", "cusparseDcsrmm", "cusparseCcsrmm", "cusparseZcsrmm",
        "cusparseScsrgemm", "cusparseDcsrgemm", "cusparseCcsrgemm",
        "cusparseZcsrgemm", "cusparseXcsrgemmNnz", "cusparseScsrmm2",
        "cusparseDcsrmm2", "cusparseCcsrmm2", "cusparseZcsrmm2",
        /*Generic*/
        "cusparseCreateCsr", "cusparseDestroySpMat", "cusparseCsrGet",
        "cusparseSpMatGetFormat", "cusparseSpMatGetIndexBase",
        "cusparseSpMatGetValues", "cusparseSpMatSetValues",
        "cusparseCreateDnMat", "cusparseDestroyDnMat", "cusparseDnMatGet",
        "cusparseDnMatGetValues", "cusparseDnMatSetValues",
        "cusparseCreateDnVec", "cusparseDestroyDnVec", "cusparseDnVecGet",
        "cusparseDnVecGetValues", "cusparseDnVecSetValues",
        "cusparseCsrSetPointers", "cusparseSpMatGetSize",
        "cusparseSpMatGetAttribute", "cusparseSpMatSetAttribute",
        "cusparseCreateConstDnVec", "cusparseConstDnVecGet",
        "cusparseConstDnVecGetValues", "cusparseSpMM",
        "cusparseSpMM_bufferSize", "cusparseSpMV", "cusparseSpMV_bufferSize",
        "cusparseSpMM_preprocess", "cusparseSpGEMM_compute",
        "cusparseSpGEMM_copy", "cusparseSpGEMM_createDescr",
        "cusparseSpGEMM_destroyDescr", "cusparseSpGEMM_workEstimation",
        "cusparseScsr2csc", "cusparseDcsr2csc", "cusparseCcsr2csc",
        "cusparseZcsr2csc", "cusparseCsr2cscEx2_bufferSize",
        "cusparseCsr2cscEx2", "cusparseSpSV_createDescr",
        "cusparseSpSV_destroyDescr", "cusparseSpSV_solve",
        "cusparseSpSV_bufferSize", "cusparseSpSV_analysis",
        "cusparseSpSM_analysis", "cusparseSpSM_bufferSize",
        "cusparseSpSM_createDescr", "cusparseSpSM_destroyDescr",
        "cusparseSpSM_solve", "cusparseCreateCoo");
  };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt()))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               unless(parentStmt())))
                    .bind("FunctionCallUsed"),
                this);
}

void SPBLASFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed")))
      return;
  }

  if (!CE->getDirectCallee())
    return;

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  StringRef FuncNameRef(FuncName);
  if (FuncNameRef.ends_with("csrmv") || FuncNameRef.ends_with("csrmv_mp")) {
    report(
        DpctGlobalInfo::getSourceManager().getExpansionLoc(CE->getBeginLoc()),
        Diagnostics::UNSUPPORT_MATRIX_TYPE, true,
        "general/symmetric/triangular");
  } else if (FuncNameRef.ends_with("csrmm") ||
             FuncNameRef.ends_with("csrmm2")) {
    report(
        DpctGlobalInfo::getSourceManager().getExpansionLoc(CE->getBeginLoc()),
        Diagnostics::UNSUPPORT_MATRIX_TYPE, true, "general");
  }

  if (CallExprRewriterFactoryBase::RewriterMap->find(FuncName) !=
      CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
  if (FuncName == "cusparseXcsrgemmNnz") {
    std::vector<std::string> MigratedArgs;
    for (const auto &Arg : CE->arguments()) {
      MigratedArgs.push_back(ExprAnalysis::ref(Arg));
    }
    // We need find the next cusparse<T>csrgemm API call which is using the
    // result of this API call, otherwise a warning will be emitted.
    auto findOuterCS = [](const Stmt *Input) {
      const CompoundStmt *CS = nullptr;
      DpctGlobalInfo::findAncestor<Stmt>(
          Input, [&](const DynTypedNode &Cur) -> bool {
            if (Cur.get<DoStmt>() || Cur.get<ForStmt>() ||
                Cur.get<WhileStmt>() || Cur.get<SwitchStmt>() ||
                Cur.get<IfStmt>())
              return true;
            if (const CompoundStmt *S = Cur.get<CompoundStmt>())
              CS = S;
            return false;
          });
      return CS;
    };
    const CompoundStmt *CS1 = findOuterCS(CE);
    // Find all the cusparse<T>csrgemm calls in this range.
    using namespace clang::ast_matchers;
    auto Matcher =
        findAll(callExpr(callee(functionDecl(hasAnyName(
                             "cusparseScsrgemm", "cusparseDcsrgemm",
                             "cusparseCcsrgemm", "cusparseZcsrgemm"))))
                    .bind("CallExpr"));
    auto CEResults = match(Matcher, *CS1, DpctGlobalInfo::getContext());
    // Find the correct call
    const CallExpr* CorrectCall = nullptr;
    for (auto &Result : CEResults) {
      const CallExpr *MatchedCE = Result.getNodeAs<CallExpr>("CallExpr");
      if (MatchedCE) {
        // 1. The context should be the same
        const CompoundStmt *CS2 = findOuterCS(MatchedCE);
        if (CS1 != CS2)
          continue;
        // 2. The args should be the same
        std::vector<std::string> MatchedCEMigratedArgs;
        for (const auto &Arg : MatchedCE->arguments()) {
          MatchedCEMigratedArgs.push_back(ExprAnalysis::ref(Arg));
        }
        if ([&]() -> bool {
              const static std::map<unsigned /*CE*/, unsigned /*MatchedCE*/>
                  IdxMap = {
                      {0, 0},   {1, 1},   {2, 2},   {3, 3},
                      {4, 4},   {5, 5},   {6, 6},   {7, 7},
                      {8, 9},   {9, 10},  {10, 11}, {11, 12},
                      {12, 14}, {13, 15}, {14, 16}, {15, 18},
                  };
              for (const auto &P : IdxMap) {
                if (MigratedArgs[P.first] != MatchedCEMigratedArgs[P.second]) {
                  return false;
                }
              }
              return true;
            }()) {
          CorrectCall = MatchedCE;
          break;
        }
      }
    }
    const constexpr int Placeholder = -1;
    std::map<int /*CE*/, int /*MatchedCE*/> InsertBeforeIdxMap;
    if (CorrectCall) {
      InsertBeforeIdxMap = {
          {8, 8},
          {12, 13},
      };
    } else {
      report(
          DpctGlobalInfo::getSourceManager().getExpansionLoc(CE->getBeginLoc()),
          Diagnostics::SPARSE_NNZ, true);
      InsertBeforeIdxMap = {
          {8, Placeholder},
          {12, Placeholder},
      };
    }
    std::string MigratedCall;
    MigratedCall = MapNames::getDpctNamespace() + "sparse::csrgemm_nnz(";
    for (unsigned i = 0; i < MigratedArgs.size(); i++) {
      if (auto Iter = InsertBeforeIdxMap.find(i);
          Iter != InsertBeforeIdxMap.end()) {
        if (Iter->second == Placeholder) {
          MigratedCall += ("dpct_placeholder, ");
        } else {
          MigratedCall += (ExprAnalysis::ref(
                               CorrectCall->getArg(InsertBeforeIdxMap.at(i))) +
                           ", ");
        }
      }
      MigratedCall += MigratedArgs[i];
      if (i != MigratedArgs.size() - 1)
        MigratedCall += ", ";
    }
    MigratedCall += ")";
    auto DefRange = getDefinitionRange(CE->getBeginLoc(), CE->getEndLoc());
    SourceLocation Begin = DefRange.getBegin();
    SourceLocation End = DefRange.getEnd();
    End = End.getLocWithOffset(
        Lexer::MeasureTokenLength(End, DpctGlobalInfo::getSourceManager(),
                                  DpctGlobalInfo::getContext().getLangOpts()));
    emplaceTransformation(replaceText(Begin, End, std::move(MigratedCall),
                                      DpctGlobalInfo::getSourceManager()));
    return;
  }
}


// Rule for spBLAS enums.
// Migrate spBLAS status values to corresponding int values
// Other spBLAS named values are migrated to corresponding named values
void SPBLASEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName(
                                "(CUSPARSE_STATUS.*)|(CUSPARSE_POINTER_MODE.*)|"
                                "(CUSPARSE_ALG.*)|(CUSPARSE_SOLVE_POLICY.*)|("
                                "CUSPARSE_SPSM_ALG_.*)"))))
                    .bind("SPBLASStatusConstants"),
                this);
  MF.addMatcher(
      declRefExpr(
          to(enumConstantDecl(matchesName(
              "(CUSPARSE_OPERATION_.*)|(CUSPARSE_FILL_MODE_.*)|("
              "CUSPARSE_DIAG_TYPE_.*)|(CUSPARSE_INDEX_.*)|(CUSPARSE_"
              "MATRIX_TYPE_.*)|(CUSPARSE_ORDER_.*)|(CUSPARSE_ACTION_.*)"))))
          .bind("SPBLASNamedValueConstants"),
      this);
}

void SPBLASEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SPBLASStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SPBLASNamedValueConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    std::string Name = EC->getNameAsString();
    auto Search = MapNamesBlas::SPBLASEnumsMap.find(Name);
    if (Search == MapNamesBlas::SPBLASEnumsMap.end()) {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected enum variable: " << Name;
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceStmt(DE, std::move(Replacement)));
  }
}



} // namespace dpct
} // namespace clang