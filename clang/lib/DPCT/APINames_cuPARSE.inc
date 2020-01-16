//===--- APINames_cuPARSE.inc ---------------------------------------===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
/****************************************************************************
*
* ENTRY(Interface APIName, APINAME, ProcessedOrNot, Flag, Priority, MigrationDesc)
*  Interface APIName: name of library Interface API available in official document
*  APIName: name of library API
*
*  ProcessedOrNot:  false|true
*    false: this API has not been processed.
*    true:  this API has been processed, need check MigrationDesc field to
            see if it has be migrated successfully.
*
*  Flag: Bit map for all kind of flag for this API.
*    default(all bits are 0): NO_FLAG
*    bit 0: API_CALL_REMOVED
*    bit 1: API_CALL_UNSUPPORTED
*  Priority: migration priority, eg. P0 - P4
*    P0:  should support in Beta release.
*    P4:  should support in Gold.
*
*  MigrationDesc: Successful[: Comment] | Unsupported: Comment
*    Successful:   this API has been migrated.
*    Unsupported:  this API can not be migrated currently.
*                  Must add Comment to explain why this API is Unsupported.
*    Comment:      extra comment on the migration.
*
****************************************************************************/
ENTRY(cusparseCreate, cusparseCreate, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateSolveAnalysisInfo, cusparseCreateSolveAnalysisInfo, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateHybMat, cusparseCreateHybMat, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateMatDescr, cusparseCreateMatDescr, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroy, cusparseDestroy, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroySolveAnalysisInfo, cusparseDestroySolveAnalysisInfo, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyHybMat, cusparseDestroyHybMat, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyMatDescr, cusparseDestroyMatDescr, false, NO_FLAG, P4, "comment")
ENTRY(cusparseGetLevelInfo, cusparseGetLevelInfo, false, NO_FLAG, P4, "comment")
ENTRY(cusparseGetMatDiagType, cusparseGetMatDiagType, false, NO_FLAG, P4, "comment")
ENTRY(cusparseGetMatFillMode, cusparseGetMatFillMode, false, NO_FLAG, P4, "comment")
ENTRY(cusparseGetMatIndexBase, cusparseGetMatIndexBase, false, NO_FLAG, P4, "comment")
ENTRY(cusparseGetMatType, cusparseGetMatType, false, NO_FLAG, P4, "comment")
ENTRY(cusparseGetPointerMode, cusparseGetPointerMode, false, NO_FLAG, P4, "comment")
ENTRY(cusparseGetVersion, cusparseGetVersion, false, NO_FLAG, P4, "comment")
ENTRY(cusparseSetMatDiagType, cusparseSetMatDiagType, false, NO_FLAG, P4, "comment")
ENTRY(cusparseSetMatFillMode, cusparseSetMatFillMode, false, NO_FLAG, P4, "comment")
ENTRY(cusparseSetMatIndexBase, cusparseSetMatIndexBase, false, NO_FLAG, P4, "comment")
ENTRY(cusparseSetMatType, cusparseSetMatType, false, NO_FLAG, P4, "comment")
ENTRY(cusparseSetPointerMode, cusparseSetPointerMode, false, NO_FLAG, P4, "comment")
ENTRY(cusparseSetStream, cusparseSetStream, false, NO_FLAG, P4, "comment")
ENTRY(cusparseGetStream, cusparseGetStream, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateCsrsv2Info, cusparseCreateCsrsv2Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyCsrsv2Info, cusparseDestroyCsrsv2Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateCsrsm2Info, cusparseCreateCsrsm2Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyCsrsm2Info, cusparseDestroyCsrsm2Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateCsric02Info, cusparseCreateCsric02Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyCsric02Info, cusparseDestroyCsric02Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateCsrilu02Info, cusparseCreateCsrilu02Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyCsrilu02Info, cusparseDestroyCsrilu02Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateBsrsv2Info, cusparseCreateBsrsv2Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyBsrsv2Info, cusparseDestroyBsrsv2Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateBsrsm2Info, cusparseCreateBsrsm2Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyBsrsm2Info, cusparseDestroyBsrsm2Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateBsric02Info, cusparseCreateBsric02Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyBsric02Info, cusparseDestroyBsric02Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateBsrilu02Info, cusparseCreateBsrilu02Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyBsrilu02Info, cusparseDestroyBsrilu02Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreateCsrgemm2Info, cusparseCreateCsrgemm2Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyCsrgemm2Info, cusparseDestroyCsrgemm2Info, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCreatePruneInfo, cusparseCreatePruneInfo, false, NO_FLAG, P4, "comment")
ENTRY(cusparseDestroyPruneInfo, cusparseDestroyPruneInfo, false, NO_FLAG, P4, "comment")
ENTRY(axpyi, axpyi, false, NO_FLAG, P4, "comment")
ENTRY(doti, doti, false, NO_FLAG, P4, "comment")
ENTRY(dotci, dotci, false, NO_FLAG, P4, "comment")
ENTRY(gthr, gthr, false, NO_FLAG, P4, "comment")
ENTRY(gthrz, gthrz, false, NO_FLAG, P4, "comment")
ENTRY(roti, roti, false, NO_FLAG, P4, "comment")
ENTRY(sctr, sctr, false, NO_FLAG, P4, "comment")
ENTRY(bsrmv, bsrmv, false, NO_FLAG, P4, "comment")
ENTRY(bsrxmv, bsrxmv, false, NO_FLAG, P4, "comment")
ENTRY(csrmv, csrmv, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCsrmvEx, cusparseCsrmvEx, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCsrmvEx_bufferSize, cusparseCsrmvEx_bufferSize, false, NO_FLAG, P4, "comment")
ENTRY(csrmv_mp, csrmv_mp, false, NO_FLAG, P4, "comment")
ENTRY(gemvi, gemvi, false, NO_FLAG, P4, "comment")
ENTRY(gemvi_bufferSize, gemvi_bufferSize, false, NO_FLAG, P4, "comment")
ENTRY(bsrsv2_bufferSize, bsrsv2_bufferSize, false, NO_FLAG, P4, "comment")
ENTRY(bsrsv2_analysis, bsrsv2_analysis, false, NO_FLAG, P4, "comment")
ENTRY(bsrsv2_solve, bsrsv2_solve, false, NO_FLAG, P4, "comment")
ENTRY(cusparseXbsrsv2_zeroPivot, cusparseXbsrsv2_zeroPivot, false, NO_FLAG, P4, "comment")
ENTRY(csrsv_analysis, csrsv_analysis, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCsrsv_analysisEx, cusparseCsrsv_analysisEx, false, NO_FLAG, P4, "comment")
ENTRY(csrsv_solve, csrsv_solve, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCsrsv_solveEx, cusparseCsrsv_solveEx, false, NO_FLAG, P4, "comment")
ENTRY(csrsv2_bufferSize, csrsv2_bufferSize, false, NO_FLAG, P4, "comment")
ENTRY(csrsv2_analysis, csrsv2_analysis, false, NO_FLAG, P4, "comment")
ENTRY(csrsv2_solve, csrsv2_solve, false, NO_FLAG, P4, "comment")
ENTRY(cusparseXcsrsv2_zeroPivot, cusparseXcsrsv2_zeroPivot, false, NO_FLAG, P4, "comment")
ENTRY(hybmv, hybmv, false, NO_FLAG, P4, "comment")
ENTRY(hybsv_analysis, hybsv_analysis, false, NO_FLAG, P4, "comment")
ENTRY(hybsv_solve, hybsv_solve, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCsr2cscEx2, cusparseCsr2cscEx2, false, NO_FLAG, P4, "comment")
ENTRY(cusparseCsr2cscEx2_bufferSize, cusparseCsr2cscEx2_bufferSize, false, NO_FLAG, P4, "comment")
ENTRY(csrmm, csrmm, false, NO_FLAG, P4, "comment")
ENTRY(csrmm2, csrmm2, false, NO_FLAG, P4, "comment")
ENTRY(csrsm_analysis, csrsm_analysis, false, NO_FLAG, P4, "comment")
ENTRY(csrsm_solve, csrsm_solve, false, NO_FLAG, P4, "comment")
ENTRY(csrsm2_bufferSizeExt, csrsm2_bufferSizeExt, false, NO_FLAG, P4, "comment")
ENTRY(csrsm2_analysis, csrsm2_analysis, false, NO_FLAG, P4, "comment")
ENTRY(csrsm2_solve, csrsm2_solve, false, NO_FLAG, P4, "comment")
ENTRY(cusparseXcsrsm2_zeroPivot, cusparseXcsrsm2_zeroPivot, false, NO_FLAG, P4, "comment")
ENTRY(bsrmm, bsrmm, false, NO_FLAG, P4, "comment")
ENTRY(bsrsm2_bufferSize, bsrsm2_bufferSize, false, NO_FLAG, P4, "comment")
ENTRY(bsrsm2_analysis, bsrsm2_analysis, false, NO_FLAG, P4, "comment")
ENTRY(bsrsm2_solve, bsrsm2_solve, false, NO_FLAG, P4, "comment")
ENTRY(cusparseXbsrsm2_zeroPivot, cusparseXbsrsm2_zeroPivot, false, NO_FLAG, P4, "comment")
ENTRY(gemmi, gemmi, false, NO_FLAG, P4, "comment")
ENTRY(csrgeam, csrgeam, false, NO_FLAG, P4, "comment")
ENTRY(csrgeam2, csrgeam2, false, NO_FLAG, P4, "comment")
ENTRY(csrgemm, csrgemm, false, NO_FLAG, P4, "comment")
ENTRY(csrgemm2, csrgemm2, false, NO_FLAG, P4, "comment")
