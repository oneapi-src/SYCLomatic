// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreateSolveAnalysisInfo | FileCheck %s -check-prefix=cusparseCreateSolveAnalysisInfo
// cusparseCreateSolveAnalysisInfo: CUDA API:
// cusparseCreateSolveAnalysisInfo-NEXT:   cusparseSolveAnalysisInfo_t info;
// cusparseCreateSolveAnalysisInfo-NEXT:   cusparseCreateSolveAnalysisInfo(&info /*cusparseSolveAnalysisInfo_t **/);
// cusparseCreateSolveAnalysisInfo-NEXT: Is migrated to:
// cusparseCreateSolveAnalysisInfo-NEXT:   std::shared_ptr<dpct::sparse::optimize_info> info;
// cusparseCreateSolveAnalysisInfo-NEXT:   info = std::make_shared<dpct::sparse::optimize_info>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDestroySolveAnalysisInfo | FileCheck %s -check-prefix=cusparseDestroySolveAnalysisInfo
// cusparseDestroySolveAnalysisInfo: CUDA API:
// cusparseDestroySolveAnalysisInfo-NEXT:   cusparseDestroySolveAnalysisInfo(info /*cusparseSolveAnalysisInfo_t*/);
// cusparseDestroySolveAnalysisInfo-NEXT: Is migrated to:
// cusparseDestroySolveAnalysisInfo-NEXT:   info.reset();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrmv | FileCheck %s -check-prefix=cusparseScsrmv
// cusparseScsrmv: CUDA API:
// cusparseScsrmv-NEXT:   cusparseScsrmv(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseScsrmv-NEXT:                  m /*int*/, n /*int*/, nnz /*int*/, alpha /*const float **/,
// cusparseScsrmv-NEXT:                  desc /*cusparseMatDescr_t*/, value /*const float **/,
// cusparseScsrmv-NEXT:                  row_ptr /*const int **/, col_idx /*const int **/,
// cusparseScsrmv-NEXT:                  x /*const float **/, beta /*const float **/, y /*float **/);
// cusparseScsrmv-NEXT: Is migrated to:
// cusparseScsrmv-NEXT:   dpct::sparse::csrmv(handle->get_queue(), trans, m, n, alpha, desc, value, row_ptr, col_idx, x, beta, y);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrmv | FileCheck %s -check-prefix=cusparseDcsrmv
// cusparseDcsrmv: CUDA API:
// cusparseDcsrmv-NEXT:   cusparseDcsrmv(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseDcsrmv-NEXT:                  m /*int*/, n /*int*/, nnz /*int*/, alpha /*const double **/,
// cusparseDcsrmv-NEXT:                  desc /*cusparseMatDescr_t*/, value /*const double **/,
// cusparseDcsrmv-NEXT:                  row_ptr /*const int **/, col_idx /*const int **/,
// cusparseDcsrmv-NEXT:                  x /*const double **/, beta /*const double **/, y /*double **/);
// cusparseDcsrmv-NEXT: Is migrated to:
// cusparseDcsrmv-NEXT:   dpct::sparse::csrmv(handle->get_queue(), trans, m, n, alpha, desc, value, row_ptr, col_idx, x, beta, y);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrmv | FileCheck %s -check-prefix=cusparseCcsrmv
// cusparseCcsrmv: CUDA API:
// cusparseCcsrmv-NEXT:   cusparseCcsrmv(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseCcsrmv-NEXT:                  m /*int*/, n /*int*/, nnz /*int*/, alpha /*const cuComplex **/,
// cusparseCcsrmv-NEXT:                  desc /*cusparseMatDescr_t*/, value /*const cuComplex **/,
// cusparseCcsrmv-NEXT:                  row_ptr /*const int **/, col_idx /*const int **/,
// cusparseCcsrmv-NEXT:                  x /*const cuComplex **/, beta /*const cuComplex **/,
// cusparseCcsrmv-NEXT:                  y /*cuComplex **/);
// cusparseCcsrmv-NEXT: Is migrated to:
// cusparseCcsrmv-NEXT:   dpct::sparse::csrmv(handle->get_queue(), trans, m, n, alpha, desc, value, row_ptr, col_idx, x, beta, y);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrmv | FileCheck %s -check-prefix=cusparseZcsrmv
// cusparseZcsrmv: CUDA API:
// cusparseZcsrmv-NEXT:   cusparseZcsrmv(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseZcsrmv-NEXT:                  m /*int*/, n /*int*/, nnz /*int*/,
// cusparseZcsrmv-NEXT:                  alpha /*const cuDoubleComplex **/, desc /*cusparseMatDescr_t*/,
// cusparseZcsrmv-NEXT:                  value /*const cuDoubleComplex **/, row_ptr /*const int **/,
// cusparseZcsrmv-NEXT:                  col_idx /*const int **/, x /*const cuDoubleComplex **/,
// cusparseZcsrmv-NEXT:                  beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/);
// cusparseZcsrmv-NEXT: Is migrated to:
// cusparseZcsrmv-NEXT:   dpct::sparse::csrmv(handle->get_queue(), trans, m, n, alpha, desc, value, row_ptr, col_idx, x, beta, y);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrmv_mp | FileCheck %s -check-prefix=cusparseScsrmv_mp
// cusparseScsrmv_mp: CUDA API:
// cusparseScsrmv_mp-NEXT:   cusparseScsrmv_mp(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseScsrmv_mp-NEXT:                     m /*int*/, n /*int*/, nnz /*int*/, alpha /*const float **/,
// cusparseScsrmv_mp-NEXT:                     desc /*cusparseMatDescr_t*/, value /*const float **/,
// cusparseScsrmv_mp-NEXT:                     row_ptr /*const int **/, col_idx /*const int **/,
// cusparseScsrmv_mp-NEXT:                     x /*const float **/, beta /*const float **/, y /*float **/);
// cusparseScsrmv_mp-NEXT: Is migrated to:
// cusparseScsrmv_mp-NEXT:   dpct::sparse::csrmv(handle->get_queue(), trans, m, n, alpha, desc, value, row_ptr, col_idx, x, beta, y);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrmv_mp | FileCheck %s -check-prefix=cusparseDcsrmv_mp
// cusparseDcsrmv_mp: CUDA API:
// cusparseDcsrmv_mp-NEXT:   cusparseDcsrmv_mp(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseDcsrmv_mp-NEXT:                     m /*int*/, n /*int*/, nnz /*int*/, alpha /*const double **/,
// cusparseDcsrmv_mp-NEXT:                     desc /*cusparseMatDescr_t*/, value /*const double **/,
// cusparseDcsrmv_mp-NEXT:                     row_ptr /*const int **/, col_idx /*const int **/,
// cusparseDcsrmv_mp-NEXT:                     x /*const double **/, beta /*const double **/,
// cusparseDcsrmv_mp-NEXT:                     y /*double **/);
// cusparseDcsrmv_mp-NEXT: Is migrated to:
// cusparseDcsrmv_mp-NEXT:   dpct::sparse::csrmv(handle->get_queue(), trans, m, n, alpha, desc, value, row_ptr, col_idx, x, beta, y);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrmv_mp | FileCheck %s -check-prefix=cusparseCcsrmv_mp
// cusparseCcsrmv_mp: CUDA API:
// cusparseCcsrmv_mp-NEXT:   cusparseCcsrmv_mp(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseCcsrmv_mp-NEXT:                     m /*int*/, n /*int*/, nnz /*int*/,
// cusparseCcsrmv_mp-NEXT:                     alpha /*const cuComplex **/, desc /*cusparseMatDescr_t*/,
// cusparseCcsrmv_mp-NEXT:                     value /*const cuComplex **/, row_ptr /*const int **/,
// cusparseCcsrmv_mp-NEXT:                     col_idx /*const int **/, x /*const cuComplex **/,
// cusparseCcsrmv_mp-NEXT:                     beta /*const cuComplex **/, y /*cuComplex **/);
// cusparseCcsrmv_mp-NEXT: Is migrated to:
// cusparseCcsrmv_mp-NEXT:   dpct::sparse::csrmv(handle->get_queue(), trans, m, n, alpha, desc, value, row_ptr, col_idx, x, beta, y);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrmv_mp | FileCheck %s -check-prefix=cusparseZcsrmv_mp
// cusparseZcsrmv_mp: CUDA API:
// cusparseZcsrmv_mp-NEXT:   cusparseZcsrmv_mp(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseZcsrmv_mp-NEXT:                     m /*int*/, n /*int*/, nnz /*int*/,
// cusparseZcsrmv_mp-NEXT:                     alpha /*const cuDoubleComplex **/,
// cusparseZcsrmv_mp-NEXT:                     desc /*cusparseMatDescr_t*/,
// cusparseZcsrmv_mp-NEXT:                     value /*const cuDoubleComplex **/, row_ptr /*const int **/,
// cusparseZcsrmv_mp-NEXT:                     col_idx /*const int **/, x /*const cuDoubleComplex **/,
// cusparseZcsrmv_mp-NEXT:                     beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/);
// cusparseZcsrmv_mp-NEXT: Is migrated to:
// cusparseZcsrmv_mp-NEXT:   dpct::sparse::csrmv(handle->get_queue(), trans, m, n, alpha, desc, value, row_ptr, col_idx, x, beta, y);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrsv_analysis | FileCheck %s -check-prefix=cusparseScsrsv_analysis
// cusparseScsrsv_analysis: CUDA API:
// cusparseScsrsv_analysis-NEXT:   cusparseScsrsv_analysis(handle /*cusparseHandle_t*/,
// cusparseScsrsv_analysis-NEXT:                           trans /*cusparseOperation_t*/, m /*int*/, nnz /*int*/,
// cusparseScsrsv_analysis-NEXT:                           desc /*cusparseMatDescr_t*/, value /*const float **/,
// cusparseScsrsv_analysis-NEXT:                           row_ptr /*const int **/, col_idx /*const int **/,
// cusparseScsrsv_analysis-NEXT:                           info /*cusparseSolveAnalysisInfo_t*/);
// cusparseScsrsv_analysis-NEXT: Is migrated to:
// cusparseScsrsv_analysis-NEXT:   dpct::sparse::optimize_csrsv(handle->get_queue(), trans, m, desc, value, row_ptr, col_idx, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrsv_analysis | FileCheck %s -check-prefix=cusparseDcsrsv_analysis
// cusparseDcsrsv_analysis: CUDA API:
// cusparseDcsrsv_analysis-NEXT:   cusparseDcsrsv_analysis(handle /*cusparseHandle_t*/,
// cusparseDcsrsv_analysis-NEXT:                           trans /*cusparseOperation_t*/, m /*int*/, nnz /*int*/,
// cusparseDcsrsv_analysis-NEXT:                           desc /*cusparseMatDescr_t*/, value /*const double **/,
// cusparseDcsrsv_analysis-NEXT:                           row_ptr /*const int **/, col_idx /*const int **/,
// cusparseDcsrsv_analysis-NEXT:                           info /*cusparseSolveAnalysisInfo_t*/);
// cusparseDcsrsv_analysis-NEXT: Is migrated to:
// cusparseDcsrsv_analysis-NEXT:   dpct::sparse::optimize_csrsv(handle->get_queue(), trans, m, desc, value, row_ptr, col_idx, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrsv_analysis | FileCheck %s -check-prefix=cusparseCcsrsv_analysis
// cusparseCcsrsv_analysis: CUDA API:
// cusparseCcsrsv_analysis-NEXT:   cusparseCcsrsv_analysis(
// cusparseCcsrsv_analysis-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseCcsrsv_analysis-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*const cuComplex **/,
// cusparseCcsrsv_analysis-NEXT:       row_ptr /*const int **/, col_idx /*const int **/,
// cusparseCcsrsv_analysis-NEXT:       info /*cusparseSolveAnalysisInfo_t*/);
// cusparseCcsrsv_analysis-NEXT: Is migrated to:
// cusparseCcsrsv_analysis-NEXT:   dpct::sparse::optimize_csrsv(handle->get_queue(), trans, m, desc, value, row_ptr, col_idx, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrsv_analysis | FileCheck %s -check-prefix=cusparseZcsrsv_analysis
// cusparseZcsrsv_analysis: CUDA API:
// cusparseZcsrsv_analysis-NEXT:   cusparseZcsrsv_analysis(
// cusparseZcsrsv_analysis-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseZcsrsv_analysis-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/,
// cusparseZcsrsv_analysis-NEXT:       value /*const cuDoubleComplex **/, row_ptr /*const int **/,
// cusparseZcsrsv_analysis-NEXT:       col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/);
// cusparseZcsrsv_analysis-NEXT: Is migrated to:
// cusparseZcsrsv_analysis-NEXT:   dpct::sparse::optimize_csrsv(handle->get_queue(), trans, m, desc, value, row_ptr, col_idx, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCsrsv_analysisEx | FileCheck %s -check-prefix=cusparseCsrsv_analysisEx
// cusparseCsrsv_analysisEx: CUDA API:
// cusparseCsrsv_analysisEx-NEXT:   cusparseCsrsv_analysisEx(
// cusparseCsrsv_analysisEx-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseCsrsv_analysisEx-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*const void **/,
// cusparseCsrsv_analysisEx-NEXT:       value_type /*cudaDataType*/, row_ptr /*const int **/,
// cusparseCsrsv_analysisEx-NEXT:       col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/,
// cusparseCsrsv_analysisEx-NEXT:       exec_type /*cudaDataType*/);
// cusparseCsrsv_analysisEx-NEXT: Is migrated to:
// cusparseCsrsv_analysisEx-NEXT:   dpct::sparse::optimize_csrsv(handle->get_queue(), trans, m, desc, value, value_type, row_ptr, col_idx, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrsv_solve | FileCheck %s -check-prefix=cusparseScsrsv_solve
// cusparseScsrsv_solve: CUDA API:
// cusparseScsrsv_solve-NEXT:   cusparseScsrsv_solve(
// cusparseScsrsv_solve-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseScsrsv_solve-NEXT:       alpha /*const float **/, desc /*cusparseMatDescr_t*/,
// cusparseScsrsv_solve-NEXT:       value /*const float **/, row_ptr /*const int **/, col_idx /*const int **/,
// cusparseScsrsv_solve-NEXT:       info /*cusparseSolveAnalysisInfo_t*/, f /*const float **/, x /*float **/);
// cusparseScsrsv_solve-NEXT: Is migrated to:
// cusparseScsrsv_solve-NEXT:   dpct::sparse::csrsv(handle->get_queue(), trans, m, alpha, desc, value, row_ptr, col_idx, info, f, x);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrsv_solve | FileCheck %s -check-prefix=cusparseDcsrsv_solve
// cusparseDcsrsv_solve: CUDA API:
// cusparseDcsrsv_solve-NEXT:   cusparseDcsrsv_solve(
// cusparseDcsrsv_solve-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseDcsrsv_solve-NEXT:       alpha /*const double **/, desc /*cusparseMatDescr_t*/,
// cusparseDcsrsv_solve-NEXT:       value /*const double **/, row_ptr /*const int **/,
// cusparseDcsrsv_solve-NEXT:       col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/,
// cusparseDcsrsv_solve-NEXT:       f /*const double **/, x /*double **/);
// cusparseDcsrsv_solve-NEXT: Is migrated to:
// cusparseDcsrsv_solve-NEXT:   dpct::sparse::csrsv(handle->get_queue(), trans, m, alpha, desc, value, row_ptr, col_idx, info, f, x);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrsv_solve | FileCheck %s -check-prefix=cusparseCcsrsv_solve
// cusparseCcsrsv_solve: CUDA API:
// cusparseCcsrsv_solve-NEXT:   cusparseCcsrsv_solve(
// cusparseCcsrsv_solve-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseCcsrsv_solve-NEXT:       alpha /*const cuComplex **/, desc /*cusparseMatDescr_t*/,
// cusparseCcsrsv_solve-NEXT:       value /*const cuComplex **/, row_ptr /*const int **/,
// cusparseCcsrsv_solve-NEXT:       col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/,
// cusparseCcsrsv_solve-NEXT:       f /*const cuComplex **/, x /*cuComplex **/);
// cusparseCcsrsv_solve-NEXT: Is migrated to:
// cusparseCcsrsv_solve-NEXT:   dpct::sparse::csrsv(handle->get_queue(), trans, m, alpha, desc, value, row_ptr, col_idx, info, f, x);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrsv_solve | FileCheck %s -check-prefix=cusparseZcsrsv_solve
// cusparseZcsrsv_solve: CUDA API:
// cusparseZcsrsv_solve-NEXT:   cusparseZcsrsv_solve(
// cusparseZcsrsv_solve-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseZcsrsv_solve-NEXT:       alpha /*const cuDoubleComplex **/, desc /*cusparseMatDescr_t*/,
// cusparseZcsrsv_solve-NEXT:       value /*const cuDoubleComplex **/, row_ptr /*const int **/,
// cusparseZcsrsv_solve-NEXT:       col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/,
// cusparseZcsrsv_solve-NEXT:       f /*const cuDoubleComplex **/, x /*cuDoubleComplex **/);
// cusparseZcsrsv_solve-NEXT: Is migrated to:
// cusparseZcsrsv_solve-NEXT:   dpct::sparse::csrsv(handle->get_queue(), trans, m, alpha, desc, value, row_ptr, col_idx, info, f, x);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCsrsv_solveEx | FileCheck %s -check-prefix=cusparseCsrsv_solveEx
// cusparseCsrsv_solveEx: CUDA API:
// cusparseCsrsv_solveEx-NEXT:   cusparseCsrsv_solveEx(
// cusparseCsrsv_solveEx-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseCsrsv_solveEx-NEXT:       alpha /*const void **/, alpha_type /*cudaDataType*/,
// cusparseCsrsv_solveEx-NEXT:       desc /*cusparseMatDescr_t*/, value /*const void **/,
// cusparseCsrsv_solveEx-NEXT:       value_type /*cudaDataType*/, row_ptr /*const int **/,
// cusparseCsrsv_solveEx-NEXT:       col_idx /*const int **/, info /*cusparseSolveAnalysisInfo_t*/,
// cusparseCsrsv_solveEx-NEXT:       f /*const void **/, f_type /*cudaDataType*/, x /*void **/,
// cusparseCsrsv_solveEx-NEXT:       x_type /*cudaDataType*/, exec_type /*cudaDataType*/);
// cusparseCsrsv_solveEx-NEXT: Is migrated to:
// cusparseCsrsv_solveEx-NEXT:   dpct::sparse::csrsv(handle->get_queue(), trans, m, alpha, alpha_type, desc, value, value_type, row_ptr, col_idx, info, f, f_type, x, x_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrmm | FileCheck %s -check-prefix=cusparseScsrmm
// cusparseScsrmm: CUDA API:
// cusparseScsrmm-NEXT:   cusparseScsrmm(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseScsrmm-NEXT:                  m /*int*/, n /*int*/, k /*int*/, nnz /*int*/,
// cusparseScsrmm-NEXT:                  alpha /*const float **/, desc /*cusparseMatDescr_t*/,
// cusparseScsrmm-NEXT:                  value /*const float **/, row_ptr /*const int **/,
// cusparseScsrmm-NEXT:                  col_idx /*const int **/, B /*const float **/, ldb /*int*/,
// cusparseScsrmm-NEXT:                  beta /*const float **/, C /*float **/, ldc /*int*/);
// cusparseScsrmm-NEXT: Is migrated to:
// cusparseScsrmm-NEXT:   dpct::sparse::csrmm(handle->get_queue(), trans, m, n, k, alpha, desc, value, row_ptr, col_idx, B, ldb, beta, C, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrmm | FileCheck %s -check-prefix=cusparseDcsrmm
// cusparseDcsrmm: CUDA API:
// cusparseDcsrmm-NEXT:   cusparseDcsrmm(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseDcsrmm-NEXT:                  m /*int*/, n /*int*/, k /*int*/, nnz /*int*/,
// cusparseDcsrmm-NEXT:                  alpha /*const double **/, desc /*cusparseMatDescr_t*/,
// cusparseDcsrmm-NEXT:                  value /*const double **/, row_ptr /*const int **/,
// cusparseDcsrmm-NEXT:                  col_idx /*const int **/, B /*const double **/, ldb /*int*/,
// cusparseDcsrmm-NEXT:                  beta /*const double **/, C /*double **/, ldc /*int*/);
// cusparseDcsrmm-NEXT: Is migrated to:
// cusparseDcsrmm-NEXT:   dpct::sparse::csrmm(handle->get_queue(), trans, m, n, k, alpha, desc, value, row_ptr, col_idx, B, ldb, beta, C, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrmm | FileCheck %s -check-prefix=cusparseCcsrmm
// cusparseCcsrmm: CUDA API:
// cusparseCcsrmm-NEXT:   cusparseCcsrmm(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseCcsrmm-NEXT:                  m /*int*/, n /*int*/, k /*int*/, nnz /*int*/,
// cusparseCcsrmm-NEXT:                  alpha /*const cuComplex **/, desc /*cusparseMatDescr_t*/,
// cusparseCcsrmm-NEXT:                  value /*const cuComplex **/, row_ptr /*const int **/,
// cusparseCcsrmm-NEXT:                  col_idx /*const int **/, B /*const cuComplex **/, ldb /*int*/,
// cusparseCcsrmm-NEXT:                  beta /*const cuComplex **/, C /*cuComplex **/, ldc /*int*/);
// cusparseCcsrmm-NEXT: Is migrated to:
// cusparseCcsrmm-NEXT:   dpct::sparse::csrmm(handle->get_queue(), trans, m, n, k, alpha, desc, value, row_ptr, col_idx, B, ldb, beta, C, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrmm | FileCheck %s -check-prefix=cusparseZcsrmm
// cusparseZcsrmm: CUDA API:
// cusparseZcsrmm-NEXT:   cusparseZcsrmm(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseZcsrmm-NEXT:                  m /*int*/, n /*int*/, k /*int*/, nnz /*int*/,
// cusparseZcsrmm-NEXT:                  alpha /*const cuDoubleComplex **/, desc /*cusparseMatDescr_t*/,
// cusparseZcsrmm-NEXT:                  value /*const cuDoubleComplex **/, row_ptr /*const int **/,
// cusparseZcsrmm-NEXT:                  col_idx /*const int **/, B /*const cuDoubleComplex **/,
// cusparseZcsrmm-NEXT:                  ldb /*int*/, beta /*const cuDoubleComplex **/,
// cusparseZcsrmm-NEXT:                  C /*cuDoubleComplex **/, ldc /*int*/);
// cusparseZcsrmm-NEXT: Is migrated to:
// cusparseZcsrmm-NEXT:   dpct::sparse::csrmm(handle->get_queue(), trans, m, n, k, alpha, desc, value, row_ptr, col_idx, B, ldb, beta, C, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrmm2 | FileCheck %s -check-prefix=cusparseScsrmm2
// cusparseScsrmm2: CUDA API:
// cusparseScsrmm2-NEXT:   cusparseScsrmm2(handle /*cusparseHandle_t*/, trans_a /*cusparseOperation_t*/,
// cusparseScsrmm2-NEXT:                   trans_b /*cusparseOperation_t*/, m /*int*/, n /*int*/,
// cusparseScsrmm2-NEXT:                   k /*int*/, nnz /*int*/, alpha /*const float **/,
// cusparseScsrmm2-NEXT:                   desc /*cusparseMatDescr_t*/, value /*const float **/,
// cusparseScsrmm2-NEXT:                   row_ptr /*const int **/, col_idx /*const int **/,
// cusparseScsrmm2-NEXT:                   B /*const float **/, ldb /*int*/, beta /*const float **/,
// cusparseScsrmm2-NEXT:                   C /*float **/, ldc /*int*/);
// cusparseScsrmm2-NEXT: Is migrated to:
// cusparseScsrmm2-NEXT:   dpct::sparse::csrmm(handle->get_queue(), trans_a, trans_b, m, n, k, alpha, desc, value, row_ptr, col_idx, B, ldb, beta, C, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrmm2 | FileCheck %s -check-prefix=cusparseDcsrmm2
// cusparseDcsrmm2: CUDA API:
// cusparseDcsrmm2-NEXT:   cusparseDcsrmm2(handle /*cusparseHandle_t*/, trans_a /*cusparseOperation_t*/,
// cusparseDcsrmm2-NEXT:                   trans_b /*cusparseOperation_t*/, m /*int*/, n /*int*/,
// cusparseDcsrmm2-NEXT:                   k /*int*/, nnz /*int*/, alpha /*const double **/,
// cusparseDcsrmm2-NEXT:                   desc /*cusparseMatDescr_t*/, value /*const double **/,
// cusparseDcsrmm2-NEXT:                   row_ptr /*const int **/, col_idx /*const int **/,
// cusparseDcsrmm2-NEXT:                   B /*const double **/, ldb /*int*/, beta /*const double **/,
// cusparseDcsrmm2-NEXT:                   C /*double **/, ldc /*int*/);
// cusparseDcsrmm2-NEXT: Is migrated to:
// cusparseDcsrmm2-NEXT:   dpct::sparse::csrmm(handle->get_queue(), trans_a, trans_b, m, n, k, alpha, desc, value, row_ptr, col_idx, B, ldb, beta, C, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrmm2 | FileCheck %s -check-prefix=cusparseCcsrmm2
// cusparseCcsrmm2: CUDA API:
// cusparseCcsrmm2-NEXT:   cusparseCcsrmm2(handle /*cusparseHandle_t*/, trans_a /*cusparseOperation_t*/,
// cusparseCcsrmm2-NEXT:                   trans_b /*cusparseOperation_t*/, m /*int*/, n /*int*/,
// cusparseCcsrmm2-NEXT:                   k /*int*/, nnz /*int*/, alpha /*const cuComplex **/,
// cusparseCcsrmm2-NEXT:                   desc /*cusparseMatDescr_t*/, value /*const cuComplex **/,
// cusparseCcsrmm2-NEXT:                   row_ptr /*const int **/, col_idx /*const int **/,
// cusparseCcsrmm2-NEXT:                   B /*const cuComplex **/, ldb /*int*/,
// cusparseCcsrmm2-NEXT:                   beta /*const cuComplex **/, C /*cuComplex **/, ldc /*int*/);
// cusparseCcsrmm2-NEXT: Is migrated to:
// cusparseCcsrmm2-NEXT:   dpct::sparse::csrmm(handle->get_queue(), trans_a, trans_b, m, n, k, alpha, desc, value, row_ptr, col_idx, B, ldb, beta, C, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrmm2 | FileCheck %s -check-prefix=cusparseZcsrmm2
// cusparseZcsrmm2: CUDA API:
// cusparseZcsrmm2-NEXT:   cusparseZcsrmm2(handle /*cusparseHandle_t*/, trans_a /*cusparseOperation_t*/,
// cusparseZcsrmm2-NEXT:                   trans_b /*cusparseOperation_t*/, m /*int*/, n /*int*/,
// cusparseZcsrmm2-NEXT:                   k /*int*/, nnz /*int*/, alpha /*const cuDoubleComplex **/,
// cusparseZcsrmm2-NEXT:                   desc /*cusparseMatDescr_t*/,
// cusparseZcsrmm2-NEXT:                   value /*const cuDoubleComplex **/, row_ptr /*const int **/,
// cusparseZcsrmm2-NEXT:                   col_idx /*const int **/, B /*const cuDoubleComplex **/,
// cusparseZcsrmm2-NEXT:                   ldb /*int*/, beta /*const cuDoubleComplex **/,
// cusparseZcsrmm2-NEXT:                   C /*cuDoubleComplex **/, ldc /*int*/);
// cusparseZcsrmm2-NEXT: Is migrated to:
// cusparseZcsrmm2-NEXT:   dpct::sparse::csrmm(handle->get_queue(), trans_a, trans_b, m, n, k, alpha, desc, value, row_ptr, col_idx, B, ldb, beta, C, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsr2csc | FileCheck %s -check-prefix=cusparseScsr2csc
// cusparseScsr2csc: CUDA API:
// cusparseScsr2csc-NEXT:   cusparseScsr2csc(handle /*cusparseHandle_t*/, m /*int*/, n /*int*/,
// cusparseScsr2csc-NEXT:                    nnz /*int*/, csr_value /*const float **/,
// cusparseScsr2csc-NEXT:                    row_ptr /*const int **/, col_idx /*const int **/,
// cusparseScsr2csc-NEXT:                    csc_value /*float **/, row_ind /*int **/, col_ptr /*int **/,
// cusparseScsr2csc-NEXT:                    act /*cusparseAction_t*/, base /*cusparseIndexBase_t*/);
// cusparseScsr2csc-NEXT: Is migrated to:
// cusparseScsr2csc-NEXT:   dpct::sparse::csr2csc(handle->get_queue(), m, n, nnz, csr_value, row_ptr, col_idx, csc_value, row_ind, col_ptr, act, base);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsr2csc | FileCheck %s -check-prefix=cusparseDcsr2csc
// cusparseDcsr2csc: CUDA API:
// cusparseDcsr2csc-NEXT:   cusparseDcsr2csc(handle /*cusparseHandle_t*/, m /*int*/, n /*int*/,
// cusparseDcsr2csc-NEXT:                    nnz /*int*/, csr_value /*const double **/,
// cusparseDcsr2csc-NEXT:                    row_ptr /*const int **/, col_idx /*const int **/,
// cusparseDcsr2csc-NEXT:                    csc_value /*double **/, row_ind /*int **/, col_ptr /*int **/,
// cusparseDcsr2csc-NEXT:                    act /*cusparseAction_t*/, base /*cusparseIndexBase_t*/);
// cusparseDcsr2csc-NEXT: Is migrated to:
// cusparseDcsr2csc-NEXT:   dpct::sparse::csr2csc(handle->get_queue(), m, n, nnz, csr_value, row_ptr, col_idx, csc_value, row_ind, col_ptr, act, base);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsr2csc | FileCheck %s -check-prefix=cusparseCcsr2csc
// cusparseCcsr2csc: CUDA API:
// cusparseCcsr2csc-NEXT:   cusparseCcsr2csc(handle /*cusparseHandle_t*/, m /*int*/, n /*int*/,
// cusparseCcsr2csc-NEXT:                    nnz /*int*/, csr_value /*const cuComplex **/,
// cusparseCcsr2csc-NEXT:                    row_ptr /*const int **/, col_idx /*const int **/,
// cusparseCcsr2csc-NEXT:                    csc_value /*cuComplex **/, row_ind /*int **/,
// cusparseCcsr2csc-NEXT:                    col_ptr /*int **/, act /*cusparseAction_t*/,
// cusparseCcsr2csc-NEXT:                    base /*cusparseIndexBase_t*/);
// cusparseCcsr2csc-NEXT: Is migrated to:
// cusparseCcsr2csc-NEXT:   dpct::sparse::csr2csc(handle->get_queue(), m, n, nnz, csr_value, row_ptr, col_idx, csc_value, row_ind, col_ptr, act, base);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsr2csc | FileCheck %s -check-prefix=cusparseZcsr2csc
// cusparseZcsr2csc: CUDA API:
// cusparseZcsr2csc-NEXT:   cusparseZcsr2csc(handle /*cusparseHandle_t*/, m /*int*/, n /*int*/,
// cusparseZcsr2csc-NEXT:                    nnz /*int*/, csr_value /*const cuDoubleComplex **/,
// cusparseZcsr2csc-NEXT:                    row_ptr /*const int **/, col_idx /*const int **/,
// cusparseZcsr2csc-NEXT:                    csc_value /*cuDoubleComplex **/, row_ind /*int **/,
// cusparseZcsr2csc-NEXT:                    col_ptr /*int **/, act /*cusparseAction_t*/,
// cusparseZcsr2csc-NEXT:                    base /*cusparseIndexBase_t*/);
// cusparseZcsr2csc-NEXT: Is migrated to:
// cusparseZcsr2csc-NEXT:   dpct::sparse::csr2csc(handle->get_queue(), m, n, nnz, csr_value, row_ptr, col_idx, csc_value, row_ind, col_ptr, act, base);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrsv2_solve | FileCheck %s -check-prefix=cusparseScsrsv2_solve
// cusparseScsrsv2_solve: CUDA API:
// cusparseScsrsv2_solve-NEXT:   cusparseScsrsv2_solve(
// cusparseScsrsv2_solve-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseScsrsv2_solve-NEXT:       nnz /*int*/, alpha /*const float **/, desc /*cusparseMatDescr_t*/,
// cusparseScsrsv2_solve-NEXT:       value /*const float **/, row_ptr /*const int **/, col_idx /*const int **/,
// cusparseScsrsv2_solve-NEXT:       info /*csrsv2Info_t*/, f /*const float **/, x /*float **/,
// cusparseScsrsv2_solve-NEXT:       policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
// cusparseScsrsv2_solve-NEXT: Is migrated to:
// cusparseScsrsv2_solve-NEXT:   dpct::sparse::csrsv(handle->get_queue(), trans, m, alpha, desc, value, row_ptr, col_idx, info, f, x);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrsv2_solve | FileCheck %s -check-prefix=cusparseDcsrsv2_solve
// cusparseDcsrsv2_solve: CUDA API:
// cusparseDcsrsv2_solve-NEXT:   cusparseDcsrsv2_solve(
// cusparseDcsrsv2_solve-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseDcsrsv2_solve-NEXT:       nnz /*int*/, alpha /*const double **/, desc /*cusparseMatDescr_t*/,
// cusparseDcsrsv2_solve-NEXT:       value /*const double **/, row_ptr /*const int **/,
// cusparseDcsrsv2_solve-NEXT:       col_idx /*const int **/, info /*csrsv2Info_t*/, f /*const double **/,
// cusparseDcsrsv2_solve-NEXT:       x /*double **/, policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
// cusparseDcsrsv2_solve-NEXT: Is migrated to:
// cusparseDcsrsv2_solve-NEXT:   dpct::sparse::csrsv(handle->get_queue(), trans, m, alpha, desc, value, row_ptr, col_idx, info, f, x);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrsv2_solve | FileCheck %s -check-prefix=cusparseCcsrsv2_solve
// cusparseCcsrsv2_solve: CUDA API:
// cusparseCcsrsv2_solve-NEXT:   cusparseCcsrsv2_solve(
// cusparseCcsrsv2_solve-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseCcsrsv2_solve-NEXT:       nnz /*int*/, alpha /*const cuComplex **/, desc /*cusparseMatDescr_t*/,
// cusparseCcsrsv2_solve-NEXT:       value /*const cuComplex **/, row_ptr /*const int **/,
// cusparseCcsrsv2_solve-NEXT:       col_idx /*const int **/, info /*csrsv2Info_t*/, f /*const cuComplex **/,
// cusparseCcsrsv2_solve-NEXT:       x /*cuComplex **/, policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
// cusparseCcsrsv2_solve-NEXT: Is migrated to:
// cusparseCcsrsv2_solve-NEXT:   dpct::sparse::csrsv(handle->get_queue(), trans, m, alpha, desc, value, row_ptr, col_idx, info, f, x);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrsv2_solve | FileCheck %s -check-prefix=cusparseZcsrsv2_solve
// cusparseZcsrsv2_solve: CUDA API:
// cusparseZcsrsv2_solve-NEXT:   cusparseZcsrsv2_solve(
// cusparseZcsrsv2_solve-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseZcsrsv2_solve-NEXT:       nnz /*int*/, alpha /*const cuDoubleComplex **/,
// cusparseZcsrsv2_solve-NEXT:       desc /*cusparseMatDescr_t*/, value /*const cuDoubleComplex **/,
// cusparseZcsrsv2_solve-NEXT:       row_ptr /*const int **/, col_idx /*const int **/, info /*csrsv2Info_t*/,
// cusparseZcsrsv2_solve-NEXT:       f /*const cuDoubleComplex **/, x /*cuDoubleComplex **/,
// cusparseZcsrsv2_solve-NEXT:       policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
// cusparseZcsrsv2_solve-NEXT: Is migrated to:
// cusparseZcsrsv2_solve-NEXT:   dpct::sparse::csrsv(handle->get_queue(), trans, m, alpha, desc, value, row_ptr, col_idx, info, f, x);
