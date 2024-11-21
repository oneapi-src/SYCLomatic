//===--------------- BLASAPIMigration.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BLASAPIMigration.h"
#include "MapNamesBlas.h"
#include "RuleInfra/ASTmatcherCommon.h"

namespace clang {
namespace dpct {

BLASEnumExpr BLASEnumExpr::create(const Expr *E,
                                  BLASEnumExpr::BLASEnumType BET) {
  BLASEnumExpr BEE;
  BEE.E = E;
  BEE.BET = BET;
  if (auto CSCE = dyn_cast<CStyleCastExpr>(E)) {
    BEE.SubExpr = CSCE->getSubExpr();
  }
  return BEE;
}

bool checkConstQualifierInDoublePointerType(
    const Expr *E, bool IsBaseValueNeedConst /* <T [DoesHereHaveConst] * *> */,
    bool IsFirstLevelPointerNeedConst /* <T * [DoesHereHaveConst] *> */) {
  const Expr *InputArg = E->IgnoreImpCasts();
  QualType InputArgPtrPtrType = InputArg->getType();
  QualType InputArgPtrType;
  if (InputArgPtrPtrType->isPointerType()) {
    InputArgPtrType = InputArgPtrPtrType->getPointeeType();
  } else if (InputArgPtrPtrType->isArrayType()) {
    const ArrayType *AT = dyn_cast<ArrayType>(InputArgPtrPtrType.getTypePtr());
    InputArgPtrType = AT->getElementType();
  } else {
    return false;
  }
  bool IsInputArgPtrConst = InputArgPtrType.isConstQualified();

  QualType InputArgBaseValueType;
  if (InputArgPtrType->isPointerType()) {
    InputArgBaseValueType = InputArgPtrType->getPointeeType();
  } else if (InputArgPtrType->isArrayType()) {
    const ArrayType *AT = dyn_cast<ArrayType>(InputArgPtrType.getTypePtr());
    InputArgBaseValueType = AT->getElementType();
  } else {
    return false;
  }
  bool IsInputArgBaseValueConst = InputArgBaseValueType.isConstQualified();

  if ((IsFirstLevelPointerNeedConst == IsInputArgPtrConst) &&
      (IsBaseValueNeedConst == IsInputArgBaseValueConst)) {
    return true;
  }
  return false;
}



// Rule for BLAS enums.
// Migrate BLAS status values to corresponding int values
// Other BLAS named values are migrated to corresponding named values
void BLASEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName(
                                "(CUBLAS_STATUS.*)|(CUDA_R_.*)|(CUDA_C_.*)|("
                                "CUBLAS_GEMM_.*)|(CUBLAS_POINTER_MODE.*)"))))
                    .bind("BLASStatusConstants"),
                this);
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName(
                      "(CUBLAS_OP.*)|(CUBLAS_SIDE.*)|(CUBLAS_FILL_"
                      "MODE.*)|(CUBLAS_DIAG.*)|(CUBLAS_.*_MATH)|CUBLAS_MATH_"
                      "DISALLOW_REDUCED_PRECISION_REDUCTION|(CUBLAS_COMPUTE_.*)"
                      "|(CUBLASLT_ORDER_.*)"
                      "|(CUBLASLT_POINTER_MODE_.*)|(CUBLASLT_MATRIX_LAYOUT_.*)|"
                      "(CUBLASLT_MATMUL_DESC_.*)|(CUBLASLT_MATRIX_TRANSFORM_"
                      "DESC_.*)|(CUBLASLT_EPILOGUE_.*)"))))
          .bind("BLASNamedValueConstants"),
      this);
}

void BLASEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "BLASStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "BLASNamedValueConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    std::string Name = EC->getNameAsString();
    auto Search = MapNamesBlas::BLASEnumsMap.find(Name);
    if (Search == MapNamesBlas::BLASEnumsMap.end()) {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected enum variable: " << Name;
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceStmt(DE, std::move(Replacement)));
  }
}
void BLASFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        /*Regular BLAS API*/
        /*Regular helper*/
        "cublasCreate_v2", "cublasDestroy_v2", "cublasSetVector",
        "cublasGetVector", "cublasSetVectorAsync", "cublasGetVectorAsync",
        "cublasSetMatrix", "cublasGetMatrix", "cublasSetMatrixAsync",
        "cublasGetMatrixAsync", "cublasSetStream_v2", "cublasGetStream_v2",
        "cublasGetPointerMode_v2", "cublasSetPointerMode_v2",
        "cublasGetAtomicsMode", "cublasSetAtomicsMode", "cublasGetVersion_v2",
        "cublasGetMathMode", "cublasSetMathMode", "cublasGetStatusString",
        "cublasSetWorkspace_v2",
        /*Regular level 1*/
        "cublasIsamax_v2", "cublasIdamax_v2", "cublasIcamax_v2",
        "cublasIzamax_v2", "cublasIsamin_v2", "cublasIdamin_v2",
        "cublasIcamin_v2", "cublasIzamin_v2", "cublasSasum_v2",
        "cublasDasum_v2", "cublasScasum_v2", "cublasDzasum_v2",
        "cublasSaxpy_v2", "cublasDaxpy_v2", "cublasCaxpy_v2", "cublasZaxpy_v2",
        "cublasScopy_v2", "cublasDcopy_v2", "cublasCcopy_v2", "cublasZcopy_v2",
        "cublasSdot_v2", "cublasDdot_v2", "cublasCdotu_v2", "cublasCdotc_v2",
        "cublasZdotu_v2", "cublasZdotc_v2", "cublasSnrm2_v2", "cublasDnrm2_v2",
        "cublasScnrm2_v2", "cublasDznrm2_v2", "cublasSrot_v2", "cublasDrot_v2",
        "cublasCsrot_v2", "cublasZdrot_v2", "cublasCrot_v2", "cublasZrot_v2",
        "cublasSrotg_v2", "cublasDrotg_v2", "cublasCrotg_v2", "cublasZrotg_v2",
        "cublasSrotm_v2", "cublasDrotm_v2", "cublasSrotmg_v2",
        "cublasDrotmg_v2", "cublasSscal_v2", "cublasDscal_v2", "cublasCscal_v2",
        "cublasCsscal_v2", "cublasZscal_v2", "cublasZdscal_v2",
        "cublasSswap_v2", "cublasDswap_v2", "cublasCswap_v2", "cublasZswap_v2",
        /*Regular level 2*/
        "cublasSgbmv_v2", "cublasDgbmv_v2", "cublasCgbmv_v2", "cublasZgbmv_v2",
        "cublasSgemv_v2", "cublasDgemv_v2", "cublasCgemv_v2", "cublasZgemv_v2",
        "cublasSger_v2", "cublasDger_v2", "cublasCgeru_v2", "cublasCgerc_v2",
        "cublasZgeru_v2", "cublasZgerc_v2", "cublasSsbmv_v2", "cublasDsbmv_v2",
        "cublasSspmv_v2", "cublasDspmv_v2", "cublasSspr_v2", "cublasDspr_v2",
        "cublasSspr2_v2", "cublasDspr2_v2", "cublasSsymv_v2", "cublasDsymv_v2",
        "cublasCsymv_v2", "cublasZsymv_v2", "cublasSsyr_v2", "cublasDsyr_v2",
        "cublasSsyr2_v2", "cublasDsyr2_v2", "cublasCsyr_v2", "cublasZsyr_v2",
        "cublasCsyr2_v2", "cublasZsyr2_v2", "cublasStbmv_v2", "cublasDtbmv_v2",
        "cublasCtbmv_v2", "cublasZtbmv_v2", "cublasStbsv_v2", "cublasDtbsv_v2",
        "cublasCtbsv_v2", "cublasZtbsv_v2", "cublasStpmv_v2", "cublasDtpmv_v2",
        "cublasCtpmv_v2", "cublasZtpmv_v2", "cublasStpsv_v2", "cublasDtpsv_v2",
        "cublasCtpsv_v2", "cublasZtpsv_v2", "cublasStrmv_v2", "cublasDtrmv_v2",
        "cublasCtrmv_v2", "cublasZtrmv_v2", "cublasStrsv_v2", "cublasDtrsv_v2",
        "cublasCtrsv_v2", "cublasZtrsv_v2", "cublasChemv_v2", "cublasZhemv_v2",
        "cublasChbmv_v2", "cublasZhbmv_v2", "cublasChpmv_v2", "cublasZhpmv_v2",
        "cublasCher_v2", "cublasZher_v2", "cublasCher2_v2", "cublasZher2_v2",
        "cublasChpr_v2", "cublasZhpr_v2", "cublasChpr2_v2", "cublasZhpr2_v2",
        /*Regular level 3*/
        "cublasSgemm_v2", "cublasDgemm_v2", "cublasCgemm_v2", "cublasZgemm_v2",
        "cublasHgemm", "cublasCgemm3m", "cublasZgemm3m",
        "cublasHgemmStridedBatched", "cublasSgemmStridedBatched",
        "cublasDgemmStridedBatched", "cublasCgemmStridedBatched",
        "cublasZgemmStridedBatched", "cublasSsymm_v2", "cublasDsymm_v2",
        "cublasCsymm_v2", "cublasZsymm_v2", "cublasSsyrk_v2", "cublasDsyrk_v2",
        "cublasCsyrk_v2", "cublasZsyrk_v2", "cublasSsyr2k_v2",
        "cublasDsyr2k_v2", "cublasCsyr2k_v2", "cublasZsyr2k_v2",
        "cublasStrsm_v2", "cublasDtrsm_v2", "cublasCtrsm_v2", "cublasZtrsm_v2",
        "cublasChemm_v2", "cublasZhemm_v2", "cublasCherk_v2", "cublasZherk_v2",
        "cublasCher2k_v2", "cublasZher2k_v2", "cublasSsyrkx", "cublasDsyrkx",
        "cublasCsyrkx", "cublasZsyrkx", "cublasCherkx", "cublasZherkx",
        "cublasStrmm_v2", "cublasDtrmm_v2", "cublasCtrmm_v2", "cublasZtrmm_v2",
        "cublasHgemmBatched", "cublasSgemmBatched", "cublasDgemmBatched",
        "cublasCgemmBatched", "cublasZgemmBatched", "cublasStrsmBatched",
        "cublasDtrsmBatched", "cublasCtrsmBatched", "cublasZtrsmBatched",
        /*Extensions*/
        "cublasSgetrfBatched", "cublasDgetrfBatched", "cublasCgetrfBatched",
        "cublasZgetrfBatched", "cublasSgetrsBatched", "cublasDgetrsBatched",
        "cublasCgetrsBatched", "cublasZgetrsBatched", "cublasSgetriBatched",
        "cublasDgetriBatched", "cublasCgetriBatched", "cublasZgetriBatched",
        "cublasSgeqrfBatched", "cublasDgeqrfBatched", "cublasCgeqrfBatched",
        "cublasZgeqrfBatched", "cublasSgelsBatched", "cublasDgelsBatched",
        "cublasCgelsBatched", "cublasZgelsBatched", "cublasGemmEx",
        "cublasSgemmEx", "cublasCgemmEx", "cublasCgemm3mEx", "cublasNrm2Ex",
        "cublasDotEx", "cublasDotcEx", "cublasScalEx", "cublasAxpyEx",
        "cublasRotEx", "cublasGemmBatchedEx", "cublasGemmStridedBatchedEx",
        "cublasSdgmm", "cublasDdgmm", "cublasCdgmm", "cublasZdgmm",
        "cublasSgeam", "cublasDgeam", "cublasCgeam", "cublasZgeam",
        "cublasCopyEx", "cublasSwapEx", "cublasIamaxEx", "cublasIaminEx",
        "cublasAsumEx", "cublasRotmEx", "cublasCsyrkEx", "cublasCsyrk3mEx",
        "cublasCherkEx", "cublasCherk3mEx",
        /*Legacy API*/
        "cublasInit", "cublasShutdown", "cublasGetError",
        "cublasSetKernelStream", "cublasGetVersion",
        /*level 1*/
        "cublasSnrm2", "cublasDnrm2", "cublasScnrm2", "cublasDznrm2",
        "cublasSdot", "cublasDdot", "cublasCdotu", "cublasCdotc", "cublasZdotu",
        "cublasZdotc", "cublasSscal", "cublasDscal", "cublasCscal",
        "cublasZscal", "cublasCsscal", "cublasZdscal", "cublasSaxpy",
        "cublasDaxpy", "cublasCaxpy", "cublasZaxpy", "cublasScopy",
        "cublasDcopy", "cublasCcopy", "cublasZcopy", "cublasSswap",
        "cublasDswap", "cublasCswap", "cublasZswap", "cublasIsamax",
        "cublasIdamax", "cublasIcamax", "cublasIzamax", "cublasIsamin",
        "cublasIdamin", "cublasIcamin", "cublasIzamin", "cublasSasum",
        "cublasDasum", "cublasScasum", "cublasDzasum", "cublasSrot",
        "cublasDrot", "cublasCsrot", "cublasZdrot", "cublasCrot", "cublasZrot",
        "cublasSrotg", "cublasDrotg", "cublasSrotm", "cublasDrotm",
        "cublasSrotmg", "cublasDrotmg",
        /*level 2*/
        "cublasSgemv", "cublasDgemv", "cublasCgemv", "cublasZgemv",
        "cublasSgbmv", "cublasDgbmv", "cublasCgbmv", "cublasZgbmv",
        "cublasStrmv", "cublasDtrmv", "cublasCtrmv", "cublasZtrmv",
        "cublasStbmv", "cublasDtbmv", "cublasCtbmv", "cublasZtbmv",
        "cublasStpmv", "cublasDtpmv", "cublasCtpmv", "cublasZtpmv",
        "cublasStrsv", "cublasDtrsv", "cublasCtrsv", "cublasZtrsv",
        "cublasStpsv", "cublasDtpsv", "cublasCtpsv", "cublasZtpsv",
        "cublasStbsv", "cublasDtbsv", "cublasCtbsv", "cublasZtbsv",
        "cublasSsymv", "cublasDsymv", "cublasChemv", "cublasZhemv",
        "cublasSsbmv", "cublasDsbmv", "cublasChbmv", "cublasZhbmv",
        "cublasSspmv", "cublasDspmv", "cublasChpmv", "cublasZhpmv",
        "cublasSger", "cublasDger", "cublasCgeru", "cublasCgerc", "cublasZgeru",
        "cublasZgerc", "cublasSsyr", "cublasDsyr", "cublasCher", "cublasZher",
        "cublasSspr", "cublasDspr", "cublasChpr", "cublasZhpr", "cublasSsyr2",
        "cublasDsyr2", "cublasCher2", "cublasZher2", "cublasSspr2",
        "cublasDspr2", "cublasChpr2", "cublasZhpr2",
        /*level 3*/
        "cublasSgemm", "cublasDgemm", "cublasCgemm", "cublasZgemm",
        "cublasSsyrk", "cublasDsyrk", "cublasCsyrk", "cublasZsyrk",
        "cublasCherk", "cublasZherk", "cublasSsyr2k", "cublasDsyr2k",
        "cublasCsyr2k", "cublasZsyr2k", "cublasCher2k", "cublasZher2k",
        "cublasSsymm", "cublasDsymm", "cublasCsymm", "cublasZsymm",
        "cublasChemm", "cublasZhemm", "cublasStrsm", "cublasDtrsm",
        "cublasCtrsm", "cublasZtrsm", "cublasStrmm", "cublasDtrmm",
        "cublasCtrmm", "cublasZtrmm",
        /*64-bit API*/
        /*helper*/
        "cublasSetVector_64", "cublasGetVector_64", "cublasSetMatrix_64",
        "cublasGetMatrix_64", "cublasSetVectorAsync_64",
        "cublasGetVectorAsync_64", "cublasSetMatrixAsync_64",
        "cublasGetMatrixAsync_64",
        /*level 1*/
        "cublasIsamax_v2_64", "cublasIdamax_v2_64", "cublasIcamax_v2_64",
        "cublasIzamax_v2_64", "cublasIsamin_v2_64", "cublasIdamin_v2_64",
        "cublasIcamin_v2_64", "cublasIzamin_v2_64", "cublasSnrm2_v2_64",
        "cublasDnrm2_v2_64", "cublasScnrm2_v2_64", "cublasDznrm2_v2_64",
        "cublasSdot_v2_64", "cublasDdot_v2_64", "cublasCdotu_v2_64",
        "cublasCdotc_v2_64", "cublasZdotu_v2_64", "cublasZdotc_v2_64",
        "cublasSscal_v2_64", "cublasDscal_v2_64", "cublasCscal_v2_64",
        "cublasCsscal_v2_64", "cublasZscal_v2_64", "cublasZdscal_v2_64",
        "cublasSaxpy_v2_64", "cublasDaxpy_v2_64", "cublasCaxpy_v2_64",
        "cublasZaxpy_v2_64", "cublasScopy_v2_64", "cublasDcopy_v2_64",
        "cublasCcopy_v2_64", "cublasZcopy_v2_64", "cublasSswap_v2_64",
        "cublasDswap_v2_64", "cublasCswap_v2_64", "cublasZswap_v2_64",
        "cublasSasum_v2_64", "cublasDasum_v2_64", "cublasScasum_v2_64",
        "cublasDzasum_v2_64", "cublasSrot_v2_64", "cublasDrot_v2_64",
        "cublasCrot_v2_64", "cublasCsrot_v2_64", "cublasZrot_v2_64",
        "cublasZdrot_v2_64", "cublasSrotm_v2_64", "cublasDrotm_v2_64",
        /*level 2*/
        "cublasSgemv_v2_64", "cublasDgemv_v2_64", "cublasCgemv_v2_64",
        "cublasZgemv_v2_64", "cublasSgbmv_v2_64", "cublasDgbmv_v2_64",
        "cublasCgbmv_v2_64", "cublasZgbmv_v2_64", "cublasStrmv_v2_64",
        "cublasDtrmv_v2_64", "cublasCtrmv_v2_64", "cublasZtrmv_v2_64",
        "cublasStbmv_v2_64", "cublasDtbmv_v2_64", "cublasCtbmv_v2_64",
        "cublasZtbmv_v2_64", "cublasStpmv_v2_64", "cublasDtpmv_v2_64",
        "cublasCtpmv_v2_64", "cublasZtpmv_v2_64", "cublasStrsv_v2_64",
        "cublasDtrsv_v2_64", "cublasCtrsv_v2_64", "cublasZtrsv_v2_64",
        "cublasStpsv_v2_64", "cublasDtpsv_v2_64", "cublasCtpsv_v2_64",
        "cublasZtpsv_v2_64", "cublasStbsv_v2_64", "cublasDtbsv_v2_64",
        "cublasCtbsv_v2_64", "cublasZtbsv_v2_64", "cublasSsymv_v2_64",
        "cublasDsymv_v2_64", "cublasCsymv_v2_64", "cublasZsymv_v2_64",
        "cublasChemv_v2_64", "cublasZhemv_v2_64", "cublasSsbmv_v2_64",
        "cublasDsbmv_v2_64", "cublasChbmv_v2_64", "cublasZhbmv_v2_64",
        "cublasSspmv_v2_64", "cublasDspmv_v2_64", "cublasChpmv_v2_64",
        "cublasZhpmv_v2_64", "cublasSger_v2_64", "cublasDger_v2_64",
        "cublasCgeru_v2_64", "cublasCgerc_v2_64", "cublasZgeru_v2_64",
        "cublasZgerc_v2_64", "cublasSsyr_v2_64", "cublasDsyr_v2_64",
        "cublasCsyr_v2_64", "cublasZsyr_v2_64", "cublasCher_v2_64",
        "cublasZher_v2_64", "cublasSspr_v2_64", "cublasDspr_v2_64",
        "cublasChpr_v2_64", "cublasZhpr_v2_64", "cublasSsyr2_v2_64",
        "cublasDsyr2_v2_64", "cublasCsyr2_v2_64", "cublasZsyr2_v2_64",
        "cublasCher2_v2_64", "cublasZher2_v2_64", "cublasSspr2_v2_64",
        "cublasDspr2_v2_64", "cublasChpr2_v2_64", "cublasZhpr2_v2_64",
        /*level 3*/
        "cublasSgemm_v2_64", "cublasDgemm_v2_64", "cublasCgemm_v2_64",
        "cublasZgemm_v2_64", "cublasSsyrk_v2_64", "cublasDsyrk_v2_64",
        "cublasCsyrk_v2_64", "cublasZsyrk_v2_64", "cublasSsymm_v2_64",
        "cublasDsymm_v2_64", "cublasCsymm_v2_64", "cublasZsymm_v2_64",
        "cublasStrsm_v2_64", "cublasDtrsm_v2_64", "cublasCtrsm_v2_64",
        "cublasZtrsm_v2_64", "cublasChemm_v2_64", "cublasZhemm_v2_64",
        "cublasCherk_v2_64", "cublasZherk_v2_64", "cublasSsyr2k_v2_64",
        "cublasDsyr2k_v2_64", "cublasCsyr2k_v2_64", "cublasZsyr2k_v2_64",
        "cublasCher2k_v2_64", "cublasZher2k_v2_64", "cublasSgeam_64",
        "cublasDgeam_64", "cublasCgeam_64", "cublasZgeam_64", "cublasSdgmm_64",
        "cublasDdgmm_64", "cublasCdgmm_64", "cublasZdgmm_64",
        "cublasStrmm_v2_64", "cublasDtrmm_v2_64", "cublasCtrmm_v2_64",
        "cublasZtrmm_v2_64", "cublasSsyrkx_64", "cublasDsyrkx_64",
        "cublasCsyrkx_64", "cublasZsyrkx_64", "cublasCherkx_64",
        "cublasZherkx_64", "cublasHgemm_64", "cublasCgemm3m_64",
        "cublasZgemm3m_64",
        /*extension*/
        "cublasNrm2Ex_64", "cublasDotEx_64", "cublasDotcEx_64",
        "cublasScalEx_64", "cublasAxpyEx_64", "cublasRotEx_64",
        "cublasGemmBatchedEx_64", "cublasGemmStridedBatchedEx_64",
        "cublasCopyEx_64", "cublasSwapEx_64", "cublasIamaxEx_64",
        "cublasIaminEx_64", "cublasAsumEx_64", "cublasRotmEx_64",
        "cublasSgemmEx_64", "cublasCgemmEx_64", "cublasCgemm3mEx_64",
        "cublasGemmEx_64", "cublasCsyrkEx_64", "cublasCsyrk3mEx_64",
        "cublasCherkEx_64", "cublasCherk3mEx_64",
        /*cublasLt*/
        "cublasLtCreate", "cublasLtDestroy", "cublasLtMatmulDescCreate",
        "cublasLtMatmulDescDestroy", "cublasLtMatmulDescSetAttribute",
        "cublasLtMatmulDescGetAttribute", "cublasLtMatrixLayoutCreate",
        "cublasLtMatrixLayoutDestroy", "cublasLtMatrixLayoutGetAttribute",
        "cublasLtMatrixLayoutSetAttribute", "cublasLtMatmul",
        "cublasLtMatmulPreferenceCreate", "cublasLtMatmulPreferenceDestroy",
        "cublasLtMatmulPreferenceSetAttribute",
        "cublasLtMatmulPreferenceGetAttribute",
        "cublasLtMatmulAlgoGetHeuristic", "cublasLtMatrixTransformDescCreate",
        "cublasLtMatrixTransformDescDestroy",
        "cublasLtMatrixTransformDescSetAttribute",
        "cublasLtMatrixTransformDescGetAttribute", "cublasLtMatrixTransform",
        "cublasLtGetVersion");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               hasAncestor(functionDecl(
                                   anyOf(hasAttr(attr::CUDADevice),
                                         hasAttr(attr::CUDAGlobal))))))
                    .bind("kernelCall"),
                this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), parentStmt(),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), unless(parentStmt()),
                unless(hasAncestor(varDecl())),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedNotInitializeVarDecl"),
      this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), hasAncestor(varDecl()),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedInitializeVarDecl"),
      this);

  MF.addMatcher(
      unresolvedLookupExpr(
          hasAnyDeclaration(namedDecl(functionName())),
          hasParent(callExpr(unless(parentStmt())).bind("callExprUsed")))
          .bind("unresolvedCallUsed"),
      this);
}

void BLASFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  auto getArgWithTypeCast = [&](const Expr* E, const std::string& CastType) {
    if (auto Cast = dyn_cast<CStyleCastExpr>(E->IgnoreImpCasts())) {
      return "(" + CastType + ")" + ExprAnalysis::ref(Cast->getSubExpr());
    } else {
      return "(" + CastType + ")" + ExprAnalysis::ref(E);
    }
  };

  bool IsAssigned = false;
  bool IsInitializeVarDecl = false;
  bool HasDeviceAttr = false;
  std::string FuncName = "";
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "kernelCall");
  if (CE) {
    HasDeviceAttr = true;
  } else if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCall"))) {
    if ((CE = getNodeAsType<CallExpr>(
             Result, "FunctionCallUsedNotInitializeVarDecl"))) {
      IsAssigned = true;
    } else if ((CE = getNodeAsType<CallExpr>(
                    Result, "FunctionCallUsedInitializeVarDecl"))) {
      IsAssigned = true;
      IsInitializeVarDecl = true;
    } else if (auto *ULE = getNodeAsType<UnresolvedLookupExpr>(
                   Result, "unresolvedCallUsed")) {
      CE = getAssistNodeAsType<CallExpr>(Result, "callExprUsed");
      FuncName = ULE->getName().getAsString();
    } else {
      return;
    }
  }

  if (FuncName == "") {
    if (!CE->getDirectCallee())
      return;
    FuncName = CE->getDirectCallee()->getNameInfo().getName().getAsString();
  }

  const SourceManager *SM = Result.SourceManager;
  auto Loc = DpctGlobalInfo::getLocInfo(SM->getExpansionLoc(CE->getBeginLoc()));
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(
          Loc.first.getCanonicalPath().str() + std::to_string(Loc.second)));

  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // There are some macros like "#define API API_v2"
  // so the function names we match should have the
  // suffix "_v2".
  bool IsMacroArg = SM->isMacroArgExpansion(CE->getBeginLoc()) &&
                    SM->isMacroArgExpansion(CE->getEndLoc());

  if (FuncNameBegin.isMacroID() && IsMacroArg) {
    FuncNameBegin = SM->getImmediateSpellingLoc(FuncNameBegin);
    FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
  } else if (FuncNameBegin.isMacroID()) {
    FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
  }

  if (FuncCallEnd.isMacroID() && IsMacroArg) {
    FuncCallEnd = SM->getImmediateSpellingLoc(FuncCallEnd);
    FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);
  } else if (FuncCallEnd.isMacroID()) {
    FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);
  }

  // Offset 1 is the length of the last token ")"
  FuncCallEnd = FuncCallEnd.getLocWithOffset(1);
  auto SR = getScopeInsertRange(CE, FuncNameBegin, FuncCallEnd);
  SourceLocation PrefixInsertLoc = SR.getBegin(), SuffixInsertLoc = SR.getEnd();

  auto FuncCallLength =
      SM->getCharacterData(FuncCallEnd) - SM->getCharacterData(FuncNameBegin);

  bool CanAvoidUsingLambda = false;
  SourceLocation OuterInsertLoc;
  std::string OriginStmtType;
  bool NeedUseLambda = isConditionOfFlowControl(
      CE, OriginStmtType, CanAvoidUsingLambda, OuterInsertLoc);
  bool IsInReturnStmt = isInReturnStmt(CE, OuterInsertLoc);
  bool CanAvoidBrace = false;
  const CompoundStmt *CS = findImmediateBlock(CE);
  if (CS && (CS->size() == 1)) {
    const Stmt *S = *(CS->child_begin());
    if (CE == S || dyn_cast<ReturnStmt>(S))
      CanAvoidBrace = true;
  }

  if (NeedUseLambda) {
    PrefixInsertLoc = FuncNameBegin;
    SuffixInsertLoc = FuncCallEnd;
  } else if (IsMacroArg) {
    NeedUseLambda = true;
    SourceRange SR = getFunctionRange(CE);
    PrefixInsertLoc = SR.getBegin();
    SuffixInsertLoc = SR.getEnd();
  } else if (IsInReturnStmt) {
    NeedUseLambda = true;
    CanAvoidUsingLambda = true;
    OriginStmtType = "return";
    // For some Legacy BLAS API (return the calculated value), below two
    // variables are needed. Although the function call is in return stmt, it
    // cannot move out and must use lambda.
    PrefixInsertLoc = FuncNameBegin;
    SuffixInsertLoc = FuncCallEnd;
  }

  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
      (FuncName == "cublasHgemmBatched" || FuncName == "cublasSgemmBatched" ||
       FuncName == "cublasDgemmBatched" || FuncName == "cublasCgemmBatched" ||
       FuncName == "cublasZgemmBatched" || FuncName == "cublasStrsmBatched" ||
       FuncName == "cublasDtrsmBatched" || FuncName == "cublasCtrsmBatched" ||
       FuncName == "cublasZtrsmBatched" || FuncName == "cublasGemmBatchedEx" ||
       FuncName == "cublasGemmBatchedEx_64")) {
    report(FuncNameBegin, Diagnostics::API_NOT_MIGRATED, false, FuncName);
    return;
  }

  std::string IndentStr = getIndent(PrefixInsertLoc, *SM).str();
  // PrefixInsertStr: stmt + NL + indent
  // SuffixInsertStr: NL + indent + stmt
  std::string PrefixInsertStr, SuffixInsertStr;
  // Clean it before starting migration
  CallExprReplStr = "";
  // TODO: Need to process the situation when scalar pointers (alpha, beta)
  // are device pointers.

  auto Item = MapNamesBlas::BLASAPIWithRewriter.find(FuncName);
  if (Item != MapNamesBlas::BLASAPIWithRewriter.end()) {
    std::string NewFunctionName = Item->second;
    if (HasDeviceAttr && !NewFunctionName.empty()) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), NewFunctionName);
      return;
    }
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  } else if (MapNamesBlas::LegacyBLASFuncReplInfoMap.find(FuncName) !=
             MapNamesBlas::LegacyBLASFuncReplInfoMap.end()) {
    auto ReplInfoPair = MapNamesBlas::LegacyBLASFuncReplInfoMap.find(FuncName);
    MapNamesBlas::BLASFuncComplexReplInfo ReplInfo = ReplInfoPair->second;
    requestFeature(HelperFeatureEnum::device_ext);
    CallExprReplStr = CallExprReplStr + ReplInfo.ReplName + "(" +
                      MapNames::getLibraryHelperNamespace() +
                      "blas::descriptor::get_saved_queue()";
    std::string IndentStr =
        getIndent(PrefixInsertLoc, (Result.Context)->getSourceManager()).str();

    std::string VarType;
    std::string VarName;
    std::string DeclOutOfBrace;
    const VarDecl *VD = 0;
    if (IsInitializeVarDecl) {
      VD = getAncestralVarDecl(CE);
      if (VD) {
        VarType = VD->getType().getAsString();
        if (VarType == "cuComplex" || VarType == "cuFloatComplex") {
          VarType = MapNames::getClNamespace() + "float2";
        }
        if (VarType == "cuDoubleComplex") {
          VarType = MapNames::getClNamespace() + "double2";
        }
        VarName = VD->getNameAsString();
      } else {
        assert(0 && "Fail to get VarDecl information");
        return;
      }
      DeclOutOfBrace = VarType + " " + VarName + ";" + getNL() + IndentStr;
    }
    std::vector<std::string> ParamsStrsVec =
        getParamsAsStrs(CE, *(Result.Context));
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
          if ((FuncName == "cublasSrotm" || FuncName == "cublasDrotm") &&
              i == 5) {
            CallExprReplStr = CallExprReplStr + ", const_cast<" +
                              ReplInfo.BufferTypeInfo[IndexTemp] + "*>(" +
                              ExprAnalysis::ref(CE->getArg(5)) + ")";
          } else if (ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<float>" ||
                     ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<double>") {
            CallExprReplStr =
                CallExprReplStr + ", " +
                getArgWithTypeCast(CE->getArg(i),
                                   ReplInfo.BufferTypeInfo[IndexTemp] + "*");
          } else {
            CallExprReplStr = CallExprReplStr + ", " + ParamsStrsVec[i];
          }
        } else {
          requestFeature(HelperFeatureEnum::device_ext);
          std::string BufferDecl;
          std::string BufferName = getBufferNameAndDeclStr(
              CE->getArg(i), ReplInfo.BufferTypeInfo[IndexTemp], IndentStr,
              BufferDecl);
          CallExprReplStr = CallExprReplStr + ", " + BufferName;
          PrefixInsertStr = PrefixInsertStr + BufferDecl;
        }
      } else if (isReplIndex(i, ReplInfo.PointerIndexInfo, IndexTemp)) {
        if (ReplInfo.PointerTypeInfo[IndexTemp] == "float" ||
            ReplInfo.PointerTypeInfo[IndexTemp] == "double") {
          // This code path is only for legacy cublasSrotmg and cublasDrotmg
          CallExprReplStr = CallExprReplStr + ", " +
                            getDrefName(CE->getArg(i)->IgnoreImplicit());
        } else {
          if (isAnIdentifierOrLiteral(CE->getArg(i)))
            CallExprReplStr =
                CallExprReplStr + ", " + ReplInfo.PointerTypeInfo[IndexTemp] +
                "(" + ParamsStrsVec[i] + ".x()," + ParamsStrsVec[i] + ".y())";
          else
            CallExprReplStr = CallExprReplStr + ", " +
                              ReplInfo.PointerTypeInfo[IndexTemp] + "((" +
                              ParamsStrsVec[i] + ").x(),(" + ParamsStrsVec[i] +
                              ").y())";
        }
      } else if (isReplIndex(i, ReplInfo.OperationIndexInfo, IndexTemp)) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'N' || Value == 'n') {
            CallExprReplStr =
                CallExprReplStr + ", oneapi::mkl::transpose::nontrans";
          } else if (Value == 'T' || Value == 't') {
            CallExprReplStr =
                CallExprReplStr + ", oneapi::mkl::transpose::trans";
          } else {
            CallExprReplStr =
                CallExprReplStr + ", oneapi::mkl::transpose::conjtrans";
          }
        } else {
          std::string TransParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            TransParamName =
                "transpose_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + TransParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            TransParamName = ParamsStrsVec[i];
          }
          CallExprReplStr = CallExprReplStr + ", " + "(" + TransParamName +
                            "=='N'||" + TransParamName +
                            "=='n') ? oneapi::mkl::transpose::nontrans: ((" +
                            TransParamName + "=='T'||" + TransParamName +
                            "=='t') ? oneapi::mkl::transpose::"
                            "trans : oneapi::mkl::transpose::conjtrans)";
        }
      } else if (ReplInfo.FillModeIndexInfo == i) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'U' || Value == 'u') {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::uplo::upper";
          } else {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::uplo::lower";
          }
        } else {
          std::string FillParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            FillParamName =
                "fillmode_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + FillParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            FillParamName = ParamsStrsVec[i];
          }
          CallExprReplStr =
              CallExprReplStr + ", " + "(" + FillParamName + "=='L'||" +
              FillParamName +
              "=='l') ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper";
        }
      } else if (ReplInfo.SideModeIndexInfo == i) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'L' || Value == 'l') {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::side::left";
          } else {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::side::right";
          }
        } else {
          std::string SideParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            SideParamName =
                "sidemode_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + SideParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            SideParamName = ParamsStrsVec[i];
          }
          CallExprReplStr =
              CallExprReplStr + ", " + "(" + SideParamName + "=='L'||" +
              SideParamName +
              "=='l') ? oneapi::mkl::side::left : oneapi::mkl::side::right";
        }
      } else if (ReplInfo.DiagTypeIndexInfo == i) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'N' || Value == 'n') {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::diag::nonunit";
          } else {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::diag::unit";
          }
        } else {
          std::string DiagParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            DiagParamName =
                "diagtype_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + DiagParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            DiagParamName = ParamsStrsVec[i];
          }
          CallExprReplStr =
              CallExprReplStr + ", " + "(" + DiagParamName + "=='N'||" +
              DiagParamName +
              "=='n') ? oneapi::mkl::diag::nonunit : oneapi::mkl::diag::unit";
        }
      } else {
        CallExprReplStr = CallExprReplStr + ", " + ParamsStrsVec[i];
      }
    }

    // All legacy APIs are synchronous
    if (FuncName == "cublasSnrm2" || FuncName == "cublasDnrm2" ||
        FuncName == "cublasScnrm2" || FuncName == "cublasDznrm2" ||
        FuncName == "cublasSdot" || FuncName == "cublasDdot" ||
        FuncName == "cublasCdotu" || FuncName == "cublasCdotc" ||
        FuncName == "cublasZdotu" || FuncName == "cublasZdotc" ||
        FuncName == "cublasIsamax" || FuncName == "cublasIdamax" ||
        FuncName == "cublasIcamax" || FuncName == "cublasIzamax" ||
        FuncName == "cublasIsamin" || FuncName == "cublasIdamin" ||
        FuncName == "cublasIcamin" || FuncName == "cublasIzamin" ||
        FuncName == "cublasSasum" || FuncName == "cublasDasum" ||
        FuncName == "cublasScasum" || FuncName == "cublasDzasum") {
      // APIs which have return value
      std::string ResultTempPtr =
          "res_temp_ptr_ct" +
          std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ResultTempVal =
          "res_temp_val_ct" +
          std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ResultTempBuf =
          "res_temp_buf_ct" +
          std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ResultType =
          ReplInfo.BufferTypeInfo[ReplInfo.BufferTypeInfo.size() - 1];
      std::string ReturnValueParamsStr;
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
        requestFeature(HelperFeatureEnum::device_ext);
        auto DefaultQueue = DpctGlobalInfo::getDefaultQueue(CE);
        PrefixInsertStr = PrefixInsertStr + ResultType + "* " + ResultTempPtr +
                          " = " + MapNames::getClNamespace() +
                          "malloc_shared<" + ResultType + ">(1, " + DefaultQueue + ");" +
                          getNL() + IndentStr + CallExprReplStr + ", " +
                          ResultTempPtr + ").wait();" + getNL() + IndentStr;

        ReturnValueParamsStr =
            "(" + ResultTempPtr + "->real(), " + ResultTempPtr + "->imag())";

        if (NeedUseLambda) {
          PrefixInsertStr = PrefixInsertStr + ResultType + " " + ResultTempVal +
                            " = *" + ResultTempPtr + ";" + getNL() + IndentStr +
                            MapNames::getClNamespace() + "free(" +
                            ResultTempPtr + ", " + DefaultQueue + ");" +
                            getNL() + IndentStr;
          ReturnValueParamsStr =
              "(" + ResultTempVal + ".real(), " + ResultTempVal + ".imag())";
        } else {
          SuffixInsertStr = SuffixInsertStr + getNL() + IndentStr +
                            MapNames::getClNamespace() + "free(" +
                            ResultTempPtr + ", " + DefaultQueue + ");";
        }
      } else {
        PrefixInsertStr = PrefixInsertStr + MapNames::getClNamespace() +
                          "buffer<" + ResultType + "> " + ResultTempBuf + "(" +
                          MapNames::getClNamespace() + "range<1>(1));" +
                          getNL() + IndentStr + CallExprReplStr + ", " +
                          ResultTempBuf + ");" + getNL() + IndentStr;
        ReturnValueParamsStr =
            "(" + ResultTempBuf + ".get_access<" + MapNames::getClNamespace() +
            "access_mode::read>()[0].real(), " + ResultTempBuf +
            ".get_access<" + MapNames::getClNamespace() +
            "access_mode::read>()[0].imag())";
      }

      std::string Repl;
      if (FuncName == "cublasCdotu" || FuncName == "cublasCdotc") {
        Repl = MapNames::getClNamespace() + "float2" + ReturnValueParamsStr;
      } else if (FuncName == "cublasZdotu" || FuncName == "cublasZdotc") {
        Repl = MapNames::getClNamespace() + "double2" + ReturnValueParamsStr;
      } else {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
          if (NeedUseLambda)
            Repl = ResultTempVal;
          else
            Repl = "*" + ResultTempPtr;
        } else {
          Repl = ResultTempBuf + ".get_access<" + MapNames::getClNamespace() +
                 "access_mode::read>()[0]";
        }
      }
      if (NeedUseLambda) {
        std::string CallRepl = "return " + Repl;

        insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                          std::string("[&](){") + getNL() + IndentStr +
                              PrefixInsertStr,
                          ";" + SuffixInsertStr + getNL() + IndentStr + "}()");
        emplaceTransformation(new ReplaceStmt(CE, CallRepl));
      } else {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            DeclOutOfBrace + "{" + getNL() + IndentStr +
                                PrefixInsertStr,
                            SuffixInsertStr + getNL() + IndentStr + "}");
        else
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            DeclOutOfBrace + PrefixInsertStr,
                            std::move(SuffixInsertStr));

        if (IsInitializeVarDecl) {
          auto ParentNodes = (Result.Context)->getParents(*VD);
          const DeclStmt *DS = 0;
          if ((DS = ParentNodes[0].get<DeclStmt>())) {
            emplaceTransformation(
                new ReplaceStmt(DS, VarName + " = " + Repl + ";"));
          } else {
            assert(0 && "Fail to get Var Decl Stmt");
            return;
          }
        } else {
          emplaceTransformation(new ReplaceStmt(CE, Repl));
        }
      }
    } else {
      // APIs which haven't return value
      if (NeedUseLambda) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
          CallExprReplStr = CallExprReplStr + ").wait()";
        } else {
          CallExprReplStr = CallExprReplStr + ")";
        }
        if (CanAvoidUsingLambda) {
          std::string InsertStr;
          if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
            InsertStr = DeclOutOfBrace + "{" + getNL() + IndentStr +
                        PrefixInsertStr + CallExprReplStr + ";" +
                        SuffixInsertStr + getNL() + IndentStr + "}" + getNL() +
                        IndentStr;
          else
            InsertStr = DeclOutOfBrace + PrefixInsertStr + CallExprReplStr +
                        ";" + SuffixInsertStr + getNL() + IndentStr;
          emplaceTransformation(
              new InsertText(OuterInsertLoc, std::move(InsertStr)));
          // APIs in this code path haven't return value, so remove the CallExpr
          emplaceTransformation(
              new ReplaceText(FuncNameBegin, FuncCallLength, ""));
        } else {
          emplaceTransformation(
              new ReplaceStmt(CE, std::move(CallExprReplStr)));
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            std::string("[&](){") + getNL() + IndentStr +
                                PrefixInsertStr,
                            getNL() + IndentStr + std::string("}()"));
        }
      } else {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
          CallExprReplStr = CallExprReplStr + ").wait()";
        } else {
          CallExprReplStr = CallExprReplStr + ")";
        }
        emplaceTransformation(new ReplaceStmt(CE, std::move(CallExprReplStr)));
        if (!PrefixInsertStr.empty()) {
          if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
            insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                              std::string("{") + getNL() + IndentStr +
                                  PrefixInsertStr,
                              getNL() + IndentStr + std::string("}"));
          else
            insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                              std::move(PrefixInsertStr), "");
        }
      }
    }
  } else if (FuncName == "cublasInit" || FuncName == "cublasShutdown" ||
             FuncName == "cublasGetError") {
    // Remove these three function calls.
    // TODO: migrate functions when they are in template
    auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
    SourceRange SR = getFunctionRange(CE);
    auto Len = SM->getDecomposedLoc(SR.getEnd()).second -
               SM->getDecomposedLoc(SR.getBegin()).second;
    if (SM->isMacroArgExpansion(CE->getBeginLoc()) &&
        SM->isMacroArgExpansion(CE->getEndLoc())) {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceText(SR.getBegin(), Len, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceText(SR.getBegin(), Len, "0"));
      }
    } else {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceStmt(CE, false, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceStmt(CE, false, ""));
      }
    }
  } else if (FuncName == "cublasGetPointerMode_v2" ||
             FuncName == "cublasSetPointerMode_v2" ||
             FuncName == "cublasGetAtomicsMode" ||
             FuncName == "cublasSetAtomicsMode" ||
             FuncName == "cublasSetWorkspace_v2") {
    std::string Msg = "this functionality is redundant in SYCL.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, true, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, true, ""));
    }
  } else if (FuncName == "cublasSetVector" || FuncName == "cublasGetVector" ||
             FuncName == "cublasSetVectorAsync" ||
             FuncName == "cublasGetVectorAsync" ||
             FuncName == "cublasSetVector_64" ||
             FuncName == "cublasGetVector_64" ||
             FuncName == "cublasSetVectorAsync_64" ||
             FuncName == "cublasGetVectorAsync_64") {
    if (HasDeviceAttr) {
      report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName),
             MapNames::getLibraryHelperNamespace() + "blas::matrix_mem_copy");
      return;
    }
    const Expr *IncxExpr = CE->getArg(3);
    const Expr *IncyExpr = CE->getArg(5);
    const std::string MigratedIncx = ExprAnalysis::ref(IncxExpr);
    const std::string MigratedIncy = ExprAnalysis::ref(IncyExpr);
    Expr::EvalResult IncxExprResult, IncyExprResult;
    if (IncxExpr->EvaluateAsInt(IncxExprResult, *Result.Context) &&
        IncyExpr->EvaluateAsInt(IncyExprResult, *Result.Context)) {
      int64_t IncxValue = IncxExprResult.Val.getInt().getExtValue();
      int64_t IncyValue = IncyExprResult.Val.getInt().getExtValue();
      if (IncxValue != IncyValue) {
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE,
               false, MapNames::ITFName.at(FuncName),
               "parameter " + MigratedIncx + " does not equal to parameter " +
                   MigratedIncy);
      } else if ((IncxValue == IncyValue) && (IncxValue != 1)) {
        // incx equals to incy, but does not equal to 1. Performance issue may
        // occur.
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE,
               false, MapNames::ITFName.at(FuncName),
               "parameter " + MigratedIncx + " equals to parameter " +
                   MigratedIncy + " but greater than 1");
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE, false,
             MapNames::ITFName.at(FuncName),
             "parameter(s) " + MigratedIncx + " and/or " + MigratedIncy +
                 " could not be evaluated");
    }

    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  } else if (FuncName == "cublasSetMatrix" || FuncName == "cublasGetMatrix" ||
             FuncName == "cublasSetMatrixAsync" ||
             FuncName == "cublasGetMatrixAsync" ||
             FuncName == "cublasSetMatrix_64" ||
             FuncName == "cublasGetMatrix_64" ||
             FuncName == "cublasSetMatrixAsync_64" ||
             FuncName == "cublasGetMatrixAsync_64") {
    if (HasDeviceAttr) {
      report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName),
             MapNames::getLibraryHelperNamespace() + "blas::matrix_mem_copy");
      return;
    }
    const Expr *RowsExpr = CE->getArg(0);
    const Expr *LdaExpr = CE->getArg(4);
    const Expr *LdbExpr = CE->getArg(6);
    const std::string MigratedRows = ExprAnalysis::ref(RowsExpr);
    const std::string MigratedLda = ExprAnalysis::ref(LdaExpr);
    const std::string MigratedLdb = ExprAnalysis::ref(LdbExpr);
    Expr::EvalResult LdaExprResult, LdbExprResult;
    if (LdaExpr->EvaluateAsInt(LdaExprResult, *Result.Context) &&
        LdbExpr->EvaluateAsInt(LdbExprResult, *Result.Context)) {
      int64_t LdaValue = LdaExprResult.Val.getInt().getExtValue();
      int64_t LdbValue = LdbExprResult.Val.getInt().getExtValue();
      if (LdaValue != LdbValue) {
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE,
               false, MapNames::ITFName.at(FuncName),
               "parameter " + MigratedLda + " does not equal to parameter " +
                   MigratedLdb);
      } else {
        Expr::EvalResult RowsExprResult;
        if (RowsExpr->EvaluateAsInt(RowsExprResult, *Result.Context)) {
          int64_t RowsValue = RowsExprResult.Val.getInt().getExtValue();
          if (LdaValue > RowsValue) {
            // lda > rows. Performance issue may occur.
            report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE,
                   false, MapNames::ITFName.at(FuncName),
                   "parameter " + MigratedRows + " is smaller than parameter " +
                       MigratedLda);
          }
        } else {
          // rows cannot be evaluated. Performance issue may occur.
          report(
              CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE,
              false, MapNames::ITFName.at(FuncName),
              "parameter " + MigratedRows +
                  " could not be evaluated and may be smaller than parameter " +
                  MigratedLda);
        }
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE, false,
             MapNames::ITFName.at(FuncName),
             "parameter(s) " + MigratedLda + " and/or " + MigratedLdb +
                 " could not be evaluated");
    }

    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  } else if (FuncName == "cublasGetVersion" ||
             FuncName == "cublasGetVersion_v2") {
    if (FuncName == "cublasGetVersion" || FuncName == "cublasGetVersion_v2") {
      DpctGlobalInfo::getInstance().insertHeader(
          SM->getExpansionLoc(CE->getBeginLoc()), HT_DPCT_COMMON_Utils);
    }

    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else {
    assert(0 && "Unknown function name");
  }
}

// Check whether the input expression is CallExpr or UnaryExprOrTypeTraitExpr
// (sizeof or alignof) or an identifier or literal
bool BLASFunctionCallRule::isCEOrUETTEOrAnIdentifierOrLiteral(const Expr *E) {
  auto CE = dyn_cast<CallExpr>(E->IgnoreImpCasts());
  if (CE != nullptr) {
    return true;
  }
  auto UETTE = dyn_cast<UnaryExprOrTypeTraitExpr>(E->IgnoreImpCasts());
  if (UETTE != nullptr) {
    return true;
  }
  if (isAnIdentifierOrLiteral(E)) {
    return true;
  }
  return false;
}

bool BLASFunctionCallRule::isReplIndex(int Input,
                                       const std::vector<int> &IndexInfo,
                                       int &IndexTemp) {
  for (int i = 0; i < static_cast<int>(IndexInfo.size()); ++i) {
    if (IndexInfo[i] == Input) {
      IndexTemp = i;
      return true;
    }
  }
  return false;
}

std::vector<std::string>
BLASFunctionCallRule::getParamsAsStrs(const CallExpr *CE,
                                      const ASTContext &Context) {
  std::vector<std::string> ParamsStrVec;
  for (auto Arg : CE->arguments())
    ParamsStrVec.emplace_back(ExprAnalysis::ref(Arg));
  return ParamsStrVec;
}

// sample code looks like:
//   Complex-type res1 = API(...);
//   res2 = API(...);
//
// migrated code looks like:
//   Complex-type res1;
//   {
//   buffer res_buffer;
//   mklAPI(res_buffer);
//   res1 = res_buffer.get_access()[0];
//   }
//   {
//   buffer res_buffer;
//   mklAPI(res_buffer);
//   res2 = res_buffer.get_access()[0];
//   }
//
// If the API return value initializes the var declaration, we need to put the
// var declaration out of the scope and assign it in the scope, otherwise users
// cannot use this var out of the scope.
// So we need to find the Decl node of the var, which is the CallExpr node's
// ancestor.
// The initial node is the matched CallExpr node. Then visit the parent node of
// the current node until the current node is a VarDecl node.
const clang::VarDecl *
BLASFunctionCallRule::getAncestralVarDecl(const clang::CallExpr *CE) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*CE);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<VarDecl>();
    if (Parent) {
      return Parent;
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }
  return nullptr;
}

} // namespace dpct
} // namespace clang
