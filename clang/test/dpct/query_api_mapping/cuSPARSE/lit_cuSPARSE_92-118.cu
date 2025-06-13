// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6, cuda-12.8, cuda-12.9
// UNSUPPORTED: v8.0, v9.0, v9.1, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6, v12.8, v12.9

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreateCsrsm2Info | FileCheck %s -check-prefix=cusparseCreateCsrsm2Info
// cusparseCreateCsrsm2Info: CUDA API:
// cusparseCreateCsrsm2Info-NEXT:   csrsm2Info_t info;
// cusparseCreateCsrsm2Info-NEXT:   cusparseCreateCsrsm2Info(&info /*csrsm2Info_t **/);
// cusparseCreateCsrsm2Info-NEXT: Is migrated to:
// cusparseCreateCsrsm2Info-NEXT:   std::shared_ptr<dpct::sparse::optimize_info> info;
// cusparseCreateCsrsm2Info-NEXT:   info = std::make_shared<dpct::sparse::optimize_info>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDestroyCsrsm2Info | FileCheck %s -check-prefix=cusparseDestroyCsrsm2Info
// cusparseDestroyCsrsm2Info: CUDA API:
// cusparseDestroyCsrsm2Info-NEXT:   cusparseDestroyCsrsm2Info(info /*csrsm2Info_t*/);
// cusparseDestroyCsrsm2Info-NEXT: Is migrated to:
// cusparseDestroyCsrsm2Info-NEXT:   info.reset();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrsm2_bufferSizeExt | FileCheck %s -check-prefix=cusparseScsrsm2_bufferSizeExt
// cusparseScsrsm2_bufferSizeExt: CUDA API:
// cusparseScsrsm2_bufferSizeExt-NEXT:   cusparseScsrsm2_bufferSizeExt(
// cusparseScsrsm2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseScsrsm2_bufferSizeExt-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseScsrsm2_bufferSizeExt-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const float **/,
// cusparseScsrsm2_bufferSizeExt-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const float **/,
// cusparseScsrsm2_bufferSizeExt-NEXT:       row_ptr /*const int **/, col_ind /*const int **/, b /*const float **/,
// cusparseScsrsm2_bufferSizeExt-NEXT:       ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
// cusparseScsrsm2_bufferSizeExt-NEXT:       buffer_size /*size_t **/);
// cusparseScsrsm2_bufferSizeExt-NEXT: Is migrated to:
// cusparseScsrsm2_bufferSizeExt-NEXT:   *buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrsm2_bufferSizeExt | FileCheck %s -check-prefix=cusparseDcsrsm2_bufferSizeExt
// cusparseDcsrsm2_bufferSizeExt: CUDA API:
// cusparseDcsrsm2_bufferSizeExt-NEXT:   cusparseDcsrsm2_bufferSizeExt(
// cusparseDcsrsm2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseDcsrsm2_bufferSizeExt-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseDcsrsm2_bufferSizeExt-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const double **/,
// cusparseDcsrsm2_bufferSizeExt-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const double **/,
// cusparseDcsrsm2_bufferSizeExt-NEXT:       row_ptr /*const int **/, col_ind /*const int **/, b /*const double **/,
// cusparseDcsrsm2_bufferSizeExt-NEXT:       ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
// cusparseDcsrsm2_bufferSizeExt-NEXT:       buffer_size /*size_t **/);
// cusparseDcsrsm2_bufferSizeExt-NEXT: Is migrated to:
// cusparseDcsrsm2_bufferSizeExt-NEXT:   *buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrsm2_bufferSizeExt | FileCheck %s -check-prefix=cusparseCcsrsm2_bufferSizeExt
// cusparseCcsrsm2_bufferSizeExt: CUDA API:
// cusparseCcsrsm2_bufferSizeExt-NEXT:   cusparseCcsrsm2_bufferSizeExt(
// cusparseCcsrsm2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseCcsrsm2_bufferSizeExt-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseCcsrsm2_bufferSizeExt-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const cuComplex **/,
// cusparseCcsrsm2_bufferSizeExt-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const cuComplex **/,
// cusparseCcsrsm2_bufferSizeExt-NEXT:       row_ptr /*const int **/, col_ind /*const int **/, b /*const cuComplex **/,
// cusparseCcsrsm2_bufferSizeExt-NEXT:       ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
// cusparseCcsrsm2_bufferSizeExt-NEXT:       buffer_size /*size_t **/);
// cusparseCcsrsm2_bufferSizeExt-NEXT: Is migrated to:
// cusparseCcsrsm2_bufferSizeExt-NEXT:   *buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrsm2_bufferSizeExt | FileCheck %s -check-prefix=cusparseZcsrsm2_bufferSizeExt
// cusparseZcsrsm2_bufferSizeExt: CUDA API:
// cusparseZcsrsm2_bufferSizeExt-NEXT:   cusparseZcsrsm2_bufferSizeExt(
// cusparseZcsrsm2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseZcsrsm2_bufferSizeExt-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseZcsrsm2_bufferSizeExt-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const cuDoubleComplex **/,
// cusparseZcsrsm2_bufferSizeExt-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const cuDoubleComplex **/,
// cusparseZcsrsm2_bufferSizeExt-NEXT:       row_ptr /*const int **/, col_ind /*const int **/,
// cusparseZcsrsm2_bufferSizeExt-NEXT:       b /*const cuDoubleComplex **/, ldb /*int*/, info /*csrsm2Info_t*/,
// cusparseZcsrsm2_bufferSizeExt-NEXT:       policy /*cusparseSolvePolicy_t*/, buffer_size /*size_t **/);
// cusparseZcsrsm2_bufferSizeExt-NEXT: Is migrated to:
// cusparseZcsrsm2_bufferSizeExt-NEXT:   *buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrsm2_analysis | FileCheck %s -check-prefix=cusparseScsrsm2_analysis
// cusparseScsrsm2_analysis: CUDA API:
// cusparseScsrsm2_analysis-NEXT:   cusparseScsrsm2_analysis(
// cusparseScsrsm2_analysis-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseScsrsm2_analysis-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseScsrsm2_analysis-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const float **/,
// cusparseScsrsm2_analysis-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const float **/,
// cusparseScsrsm2_analysis-NEXT:       row_ptr /*const int **/, col_ind /*const int **/, b /*const float **/,
// cusparseScsrsm2_analysis-NEXT:       ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
// cusparseScsrsm2_analysis-NEXT:       buffer /*void **/);
// cusparseScsrsm2_analysis-NEXT: Is migrated to:
// cusparseScsrsm2_analysis-NEXT:   dpct::sparse::optimize_csrsm(handle->get_queue(), trans_a, trans_b, m, nrhs, descr, value, row_ptr, col_ind, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrsm2_analysis | FileCheck %s -check-prefix=cusparseDcsrsm2_analysis
// cusparseDcsrsm2_analysis: CUDA API:
// cusparseDcsrsm2_analysis-NEXT:   cusparseDcsrsm2_analysis(
// cusparseDcsrsm2_analysis-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseDcsrsm2_analysis-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseDcsrsm2_analysis-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const double **/,
// cusparseDcsrsm2_analysis-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const double **/,
// cusparseDcsrsm2_analysis-NEXT:       row_ptr /*const int **/, col_ind /*const int **/, b /*const double **/,
// cusparseDcsrsm2_analysis-NEXT:       ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
// cusparseDcsrsm2_analysis-NEXT:       buffer /*void **/);
// cusparseDcsrsm2_analysis-NEXT: Is migrated to:
// cusparseDcsrsm2_analysis-NEXT:   dpct::sparse::optimize_csrsm(handle->get_queue(), trans_a, trans_b, m, nrhs, descr, value, row_ptr, col_ind, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrsm2_analysis | FileCheck %s -check-prefix=cusparseCcsrsm2_analysis
// cusparseCcsrsm2_analysis: CUDA API:
// cusparseCcsrsm2_analysis-NEXT:   cusparseCcsrsm2_analysis(
// cusparseCcsrsm2_analysis-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseCcsrsm2_analysis-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseCcsrsm2_analysis-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const cuComplex **/,
// cusparseCcsrsm2_analysis-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const cuComplex **/,
// cusparseCcsrsm2_analysis-NEXT:       row_ptr /*const int **/, col_ind /*const int **/, b /*const cuComplex **/,
// cusparseCcsrsm2_analysis-NEXT:       ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
// cusparseCcsrsm2_analysis-NEXT:       buffer /*void **/);
// cusparseCcsrsm2_analysis-NEXT: Is migrated to:
// cusparseCcsrsm2_analysis-NEXT:   dpct::sparse::optimize_csrsm(handle->get_queue(), trans_a, trans_b, m, nrhs, descr, value, row_ptr, col_ind, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrsm2_analysis | FileCheck %s -check-prefix=cusparseZcsrsm2_analysis
// cusparseZcsrsm2_analysis: CUDA API:
// cusparseZcsrsm2_analysis-NEXT:   cusparseZcsrsm2_analysis(
// cusparseZcsrsm2_analysis-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseZcsrsm2_analysis-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseZcsrsm2_analysis-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const cuDoubleComplex **/,
// cusparseZcsrsm2_analysis-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const cuDoubleComplex **/,
// cusparseZcsrsm2_analysis-NEXT:       row_ptr /*const int **/, col_ind /*const int **/,
// cusparseZcsrsm2_analysis-NEXT:       b /*const cuDoubleComplex **/, ldb /*int*/, info /*csrsm2Info_t*/,
// cusparseZcsrsm2_analysis-NEXT:       policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
// cusparseZcsrsm2_analysis-NEXT: Is migrated to:
// cusparseZcsrsm2_analysis-NEXT:   dpct::sparse::optimize_csrsm(handle->get_queue(), trans_a, trans_b, m, nrhs, descr, value, row_ptr, col_ind, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrsm2_solve | FileCheck %s -check-prefix=cusparseScsrsm2_solve
// cusparseScsrsm2_solve: CUDA API:
// cusparseScsrsm2_solve-NEXT:   cusparseScsrsm2_solve(
// cusparseScsrsm2_solve-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseScsrsm2_solve-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseScsrsm2_solve-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const float **/,
// cusparseScsrsm2_solve-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const float **/,
// cusparseScsrsm2_solve-NEXT:       row_ptr /*const int **/, col_ind /*const int **/, b /*float **/,
// cusparseScsrsm2_solve-NEXT:       ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
// cusparseScsrsm2_solve-NEXT:       buffer /*void **/);
// cusparseScsrsm2_solve-NEXT: Is migrated to:
// cusparseScsrsm2_solve-NEXT:   dpct::sparse::csrsm(handle->get_queue(), trans_a, trans_b, m, nrhs, alpha, descr, value, row_ptr, col_ind, b, ldb, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrsm2_solve | FileCheck %s -check-prefix=cusparseDcsrsm2_solve
// cusparseDcsrsm2_solve: CUDA API:
// cusparseDcsrsm2_solve-NEXT:   cusparseDcsrsm2_solve(
// cusparseDcsrsm2_solve-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseDcsrsm2_solve-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseDcsrsm2_solve-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const double **/,
// cusparseDcsrsm2_solve-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const double **/,
// cusparseDcsrsm2_solve-NEXT:       row_ptr /*const int **/, col_ind /*const int **/, b /*double **/,
// cusparseDcsrsm2_solve-NEXT:       ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
// cusparseDcsrsm2_solve-NEXT:       buffer /*void **/);
// cusparseDcsrsm2_solve-NEXT: Is migrated to:
// cusparseDcsrsm2_solve-NEXT:   dpct::sparse::csrsm(handle->get_queue(), trans_a, trans_b, m, nrhs, alpha, descr, value, row_ptr, col_ind, b, ldb, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrsm2_solve | FileCheck %s -check-prefix=cusparseCcsrsm2_solve
// cusparseCcsrsm2_solve: CUDA API:
// cusparseCcsrsm2_solve-NEXT:   cusparseCcsrsm2_solve(
// cusparseCcsrsm2_solve-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseCcsrsm2_solve-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseCcsrsm2_solve-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const cuComplex **/,
// cusparseCcsrsm2_solve-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const cuComplex **/,
// cusparseCcsrsm2_solve-NEXT:       row_ptr /*const int **/, col_ind /*const int **/, b /*cuComplex **/,
// cusparseCcsrsm2_solve-NEXT:       ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
// cusparseCcsrsm2_solve-NEXT:       buffer /*void **/);
// cusparseCcsrsm2_solve-NEXT: Is migrated to:
// cusparseCcsrsm2_solve-NEXT:   dpct::sparse::csrsm(handle->get_queue(), trans_a, trans_b, m, nrhs, alpha, descr, value, row_ptr, col_ind, b, ldb, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrsm2_solve | FileCheck %s -check-prefix=cusparseZcsrsm2_solve
// cusparseZcsrsm2_solve: CUDA API:
// cusparseZcsrsm2_solve-NEXT:   cusparseZcsrsm2_solve(
// cusparseZcsrsm2_solve-NEXT:       handle /*cusparseHandle_t*/, algo /*int*/,
// cusparseZcsrsm2_solve-NEXT:       trans_a /*cusparseOperation_t*/, trans_b /*cusparseOperation_t*/,
// cusparseZcsrsm2_solve-NEXT:       m /*int*/, nrhs /*int*/, nnz /*int*/, alpha /*const cuDoubleComplex **/,
// cusparseZcsrsm2_solve-NEXT:       descr /*const cusparseMatDescr_t*/, value /*const cuDoubleComplex **/,
// cusparseZcsrsm2_solve-NEXT:       row_ptr /*const int **/, col_ind /*const int **/, b /*cuDoubleComplex **/,
// cusparseZcsrsm2_solve-NEXT:       ldb /*int*/, info /*csrsm2Info_t*/, policy /*cusparseSolvePolicy_t*/,
// cusparseZcsrsm2_solve-NEXT:       buffer /*void **/);
// cusparseZcsrsm2_solve-NEXT: Is migrated to:
// cusparseZcsrsm2_solve-NEXT:   dpct::sparse::csrsm(handle->get_queue(), trans_a, trans_b, m, nrhs, alpha, descr, value, row_ptr, col_ind, b, ldb, info);
