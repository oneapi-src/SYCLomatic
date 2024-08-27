// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreateCsrsv2Info | FileCheck %s -check-prefix=cusparseCreateCsrsv2Info
// cusparseCreateCsrsv2Info: CUDA API:
// cusparseCreateCsrsv2Info-NEXT:   csrsv2Info_t info;
// cusparseCreateCsrsv2Info-NEXT:   cusparseCreateCsrsv2Info(&info /*csrsv2Info_t **/);
// cusparseCreateCsrsv2Info-NEXT: Is migrated to:
// cusparseCreateCsrsv2Info-NEXT:   std::shared_ptr<dpct::sparse::optimize_info> info;
// cusparseCreateCsrsv2Info-NEXT:   info = std::make_shared<dpct::sparse::optimize_info>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDestroyCsrsv2Info | FileCheck %s -check-prefix=cusparseDestroyCsrsv2Info
// cusparseDestroyCsrsv2Info: CUDA API:
// cusparseDestroyCsrsv2Info-NEXT:   cusparseDestroyCsrsv2Info(info /*csrsv2Info_t*/);
// cusparseDestroyCsrsv2Info-NEXT: Is migrated to:
// cusparseDestroyCsrsv2Info-NEXT:   info.reset();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrsv2_bufferSizeExt | FileCheck %s -check-prefix=cusparseScsrsv2_bufferSizeExt
// cusparseScsrsv2_bufferSizeExt: CUDA API:
// cusparseScsrsv2_bufferSizeExt-NEXT:   size_t buffer_size;
// cusparseScsrsv2_bufferSizeExt-NEXT:   cusparseScsrsv2_bufferSizeExt(
// cusparseScsrsv2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseScsrsv2_bufferSizeExt-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*float **/,
// cusparseScsrsv2_bufferSizeExt-NEXT:       row_ptr /*const int **/, con_ind /*const int **/, info /*csrsv2Info_t*/,
// cusparseScsrsv2_bufferSizeExt-NEXT:       &buffer_size /*size_t **/);
// cusparseScsrsv2_bufferSizeExt-NEXT: Is migrated to:
// cusparseScsrsv2_bufferSizeExt-NEXT:   size_t buffer_size;
// cusparseScsrsv2_bufferSizeExt-NEXT:   buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrsv2_bufferSizeExt | FileCheck %s -check-prefix=cusparseDcsrsv2_bufferSizeExt
// cusparseDcsrsv2_bufferSizeExt: CUDA API:
// cusparseDcsrsv2_bufferSizeExt-NEXT:   size_t buffer_size;
// cusparseDcsrsv2_bufferSizeExt-NEXT:   cusparseDcsrsv2_bufferSizeExt(
// cusparseDcsrsv2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseDcsrsv2_bufferSizeExt-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*double **/,
// cusparseDcsrsv2_bufferSizeExt-NEXT:       row_ptr /*const int **/, con_ind /*const int **/, info /*csrsv2Info_t*/,
// cusparseDcsrsv2_bufferSizeExt-NEXT:       &buffer_size /*size_t **/);
// cusparseDcsrsv2_bufferSizeExt-NEXT: Is migrated to:
// cusparseDcsrsv2_bufferSizeExt-NEXT:   size_t buffer_size;
// cusparseDcsrsv2_bufferSizeExt-NEXT:   buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrsv2_bufferSizeExt | FileCheck %s -check-prefix=cusparseCcsrsv2_bufferSizeExt
// cusparseCcsrsv2_bufferSizeExt: CUDA API:
// cusparseCcsrsv2_bufferSizeExt-NEXT:   size_t buffer_size;
// cusparseCcsrsv2_bufferSizeExt-NEXT:   cusparseCcsrsv2_bufferSizeExt(
// cusparseCcsrsv2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseCcsrsv2_bufferSizeExt-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*cuComplex **/,
// cusparseCcsrsv2_bufferSizeExt-NEXT:       row_ptr /*const int **/, con_ind /*const int **/, info /*csrsv2Info_t*/,
// cusparseCcsrsv2_bufferSizeExt-NEXT:       &buffer_size /*size_t **/);
// cusparseCcsrsv2_bufferSizeExt-NEXT: Is migrated to:
// cusparseCcsrsv2_bufferSizeExt-NEXT:   size_t buffer_size;
// cusparseCcsrsv2_bufferSizeExt-NEXT:   buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrsv2_bufferSizeExt | FileCheck %s -check-prefix=cusparseZcsrsv2_bufferSizeExt
// cusparseZcsrsv2_bufferSizeExt: CUDA API:
// cusparseZcsrsv2_bufferSizeExt-NEXT:   size_t buffer_size;
// cusparseZcsrsv2_bufferSizeExt-NEXT:   cusparseZcsrsv2_bufferSizeExt(
// cusparseZcsrsv2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseZcsrsv2_bufferSizeExt-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*cuDoubleComplex **/,
// cusparseZcsrsv2_bufferSizeExt-NEXT:       row_ptr /*const int **/, con_ind /*const int **/, info /*csrsv2Info_t*/,
// cusparseZcsrsv2_bufferSizeExt-NEXT:       &buffer_size /*size_t **/);
// cusparseZcsrsv2_bufferSizeExt-NEXT: Is migrated to:
// cusparseZcsrsv2_bufferSizeExt-NEXT:   size_t buffer_size;
// cusparseZcsrsv2_bufferSizeExt-NEXT:   buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCsrmvEx | FileCheck %s -check-prefix=cusparseCsrmvEx
// cusparseCsrmvEx: CUDA API:
// cusparseCsrmvEx-NEXT:   cusparseCsrmvEx(
// cusparseCsrmvEx-NEXT:       handle /*cusparseHandle_t*/, algo /*cusparseAlgMode_t*/,
// cusparseCsrmvEx-NEXT:       trans /*cusparseOperation_t*/, m /*int*/, n /*int*/, nnz /*int*/,
// cusparseCsrmvEx-NEXT:       alpha /*const void **/, alpha_type /*cudaDataType*/,
// cusparseCsrmvEx-NEXT:       desc /*cusparseMatDescr_t*/, value /*const void **/,
// cusparseCsrmvEx-NEXT:       value_type /*cudaDataType*/, row_ptr /*const int **/,
// cusparseCsrmvEx-NEXT:       col_idx /*const int **/, x /*const void **/, x_type /*cudaDataType*/,
// cusparseCsrmvEx-NEXT:       beta /*const void **/, beta_type /*cudaDataType*/, y /*void **/,
// cusparseCsrmvEx-NEXT:       y_type /*cudaDataType*/, exec_type /*cudaDataType*/, buffer /*void **/);
// cusparseCsrmvEx-NEXT: Is migrated to:
// cusparseCsrmvEx-NEXT:   dpct::sparse::csrmv(handle->get_queue(), trans, m, n, alpha, alpha_type, desc, value, value_type, row_ptr, col_idx, x, x_type, beta, beta_type, y, y_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCsrmvEx_bufferSize | FileCheck %s -check-prefix=cusparseCsrmvEx_bufferSize
// cusparseCsrmvEx_bufferSize: CUDA API:
// cusparseCsrmvEx_bufferSize-NEXT:   size_t buffer_size_in_bytes;
// cusparseCsrmvEx_bufferSize-NEXT:   cusparseCsrmvEx_bufferSize(
// cusparseCsrmvEx_bufferSize-NEXT:       handle /*cusparseHandle_t*/, algo /*cusparseAlgMode_t*/,
// cusparseCsrmvEx_bufferSize-NEXT:       trans /*cusparseOperation_t*/, m /*int*/, n /*int*/, nnz /*int*/,
// cusparseCsrmvEx_bufferSize-NEXT:       alpha /*const void **/, alpha_type /*cudaDataType*/,
// cusparseCsrmvEx_bufferSize-NEXT:       desc /*cusparseMatDescr_t*/, value /*const void **/,
// cusparseCsrmvEx_bufferSize-NEXT:       value_type /*cudaDataType*/, row_ptr /*const int **/,
// cusparseCsrmvEx_bufferSize-NEXT:       col_idx /*const int **/, x /*const void **/, x_type /*cudaDataType*/,
// cusparseCsrmvEx_bufferSize-NEXT:       beta /*const void **/, beta_type /*cudaDataType*/, y /*void **/,
// cusparseCsrmvEx_bufferSize-NEXT:       y_type /*cudaDataType*/, exec_type /*cudaDataType*/,
// cusparseCsrmvEx_bufferSize-NEXT:       &buffer_size_in_bytes /*size_t **/);
// cusparseCsrmvEx_bufferSize-NEXT: Is migrated to:
// cusparseCsrmvEx_bufferSize-NEXT:   size_t buffer_size_in_bytes;
// cusparseCsrmvEx_bufferSize-NEXT:   buffer_size_in_bytes = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrsv2_bufferSize | FileCheck %s -check-prefix=cusparseScsrsv2_bufferSize
// cusparseScsrsv2_bufferSize: CUDA API:
// cusparseScsrsv2_bufferSize-NEXT:   int buffer_size_in_bytes;
// cusparseScsrsv2_bufferSize-NEXT:   cusparseScsrsv2_bufferSize(
// cusparseScsrsv2_bufferSize-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseScsrsv2_bufferSize-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*float **/,
// cusparseScsrsv2_bufferSize-NEXT:       row_ptr /*const int **/, col_idx /*const int **/, info /*csrsv2Info_t*/,
// cusparseScsrsv2_bufferSize-NEXT:       &buffer_size_in_bytes /*int **/);
// cusparseScsrsv2_bufferSize-NEXT: Is migrated to:
// cusparseScsrsv2_bufferSize-NEXT:   int buffer_size_in_bytes;
// cusparseScsrsv2_bufferSize-NEXT:   buffer_size_in_bytes = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrsv2_bufferSize | FileCheck %s -check-prefix=cusparseDcsrsv2_bufferSize
// cusparseDcsrsv2_bufferSize: CUDA API:
// cusparseDcsrsv2_bufferSize-NEXT:   int buffer_size_in_bytes;
// cusparseDcsrsv2_bufferSize-NEXT:   cusparseDcsrsv2_bufferSize(
// cusparseDcsrsv2_bufferSize-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseDcsrsv2_bufferSize-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*double **/,
// cusparseDcsrsv2_bufferSize-NEXT:       row_ptr /*const int **/, col_idx /*const int **/, info /*csrsv2Info_t*/,
// cusparseDcsrsv2_bufferSize-NEXT:       &buffer_size_in_bytes /*int **/);
// cusparseDcsrsv2_bufferSize-NEXT: Is migrated to:
// cusparseDcsrsv2_bufferSize-NEXT:   int buffer_size_in_bytes;
// cusparseDcsrsv2_bufferSize-NEXT:   buffer_size_in_bytes = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrsv2_bufferSize | FileCheck %s -check-prefix=cusparseCcsrsv2_bufferSize
// cusparseCcsrsv2_bufferSize: CUDA API:
// cusparseCcsrsv2_bufferSize-NEXT:   int buffer_size_in_bytes;
// cusparseCcsrsv2_bufferSize-NEXT:   cusparseCcsrsv2_bufferSize(
// cusparseCcsrsv2_bufferSize-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseCcsrsv2_bufferSize-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*cuComplex **/,
// cusparseCcsrsv2_bufferSize-NEXT:       row_ptr /*const int **/, col_idx /*const int **/, info /*csrsv2Info_t*/,
// cusparseCcsrsv2_bufferSize-NEXT:       &buffer_size_in_bytes /*int **/);
// cusparseCcsrsv2_bufferSize-NEXT: Is migrated to:
// cusparseCcsrsv2_bufferSize-NEXT:   int buffer_size_in_bytes;
// cusparseCcsrsv2_bufferSize-NEXT:   buffer_size_in_bytes = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrsv2_bufferSize | FileCheck %s -check-prefix=cusparseZcsrsv2_bufferSize
// cusparseZcsrsv2_bufferSize: CUDA API:
// cusparseZcsrsv2_bufferSize-NEXT:   int buffer_size_in_bytes;
// cusparseZcsrsv2_bufferSize-NEXT:   cusparseZcsrsv2_bufferSize(
// cusparseZcsrsv2_bufferSize-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseZcsrsv2_bufferSize-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*cuDoubleComplex **/,
// cusparseZcsrsv2_bufferSize-NEXT:       row_ptr /*const int **/, col_idx /*const int **/, info /*csrsv2Info_t*/,
// cusparseZcsrsv2_bufferSize-NEXT:       &buffer_size_in_bytes /*int **/);
// cusparseZcsrsv2_bufferSize-NEXT: Is migrated to:
// cusparseZcsrsv2_bufferSize-NEXT:   int buffer_size_in_bytes;
// cusparseZcsrsv2_bufferSize-NEXT:   buffer_size_in_bytes = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrsv2_analysis | FileCheck %s -check-prefix=cusparseScsrsv2_analysis
// cusparseScsrsv2_analysis: CUDA API:
// cusparseScsrsv2_analysis-NEXT:   cusparseScsrsv2_analysis(
// cusparseScsrsv2_analysis-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseScsrsv2_analysis-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*const float **/,
// cusparseScsrsv2_analysis-NEXT:       row_ptr /*const int **/, col_idx /*const int **/, info /*csrsv2Info_t*/,
// cusparseScsrsv2_analysis-NEXT:       policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
// cusparseScsrsv2_analysis-NEXT: Is migrated to:
// cusparseScsrsv2_analysis-NEXT:   dpct::sparse::optimize_csrsv(handle->get_queue(), trans, m, desc, value, row_ptr, col_idx, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrsv2_analysis | FileCheck %s -check-prefix=cusparseDcsrsv2_analysis
// cusparseDcsrsv2_analysis: CUDA API:
// cusparseDcsrsv2_analysis-NEXT:   cusparseDcsrsv2_analysis(
// cusparseDcsrsv2_analysis-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseDcsrsv2_analysis-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*const double **/,
// cusparseDcsrsv2_analysis-NEXT:       row_ptr /*const int **/, col_idx /*const int **/, info /*csrsv2Info_t*/,
// cusparseDcsrsv2_analysis-NEXT:       policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
// cusparseDcsrsv2_analysis-NEXT: Is migrated to:
// cusparseDcsrsv2_analysis-NEXT:   dpct::sparse::optimize_csrsv(handle->get_queue(), trans, m, desc, value, row_ptr, col_idx, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrsv2_analysis | FileCheck %s -check-prefix=cusparseCcsrsv2_analysis
// cusparseCcsrsv2_analysis: CUDA API:
// cusparseCcsrsv2_analysis-NEXT:   cusparseCcsrsv2_analysis(
// cusparseCcsrsv2_analysis-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseCcsrsv2_analysis-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/, value /*const cuComplex **/,
// cusparseCcsrsv2_analysis-NEXT:       row_ptr /*const int **/, col_idx /*const int **/, info /*csrsv2Info_t*/,
// cusparseCcsrsv2_analysis-NEXT:       policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
// cusparseCcsrsv2_analysis-NEXT: Is migrated to:
// cusparseCcsrsv2_analysis-NEXT:   dpct::sparse::optimize_csrsv(handle->get_queue(), trans, m, desc, value, row_ptr, col_idx, info);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrsv2_analysis | FileCheck %s -check-prefix=cusparseZcsrsv2_analysis
// cusparseZcsrsv2_analysis: CUDA API:
// cusparseZcsrsv2_analysis-NEXT:   cusparseZcsrsv2_analysis(
// cusparseZcsrsv2_analysis-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/, m /*int*/,
// cusparseZcsrsv2_analysis-NEXT:       nnz /*int*/, desc /*cusparseMatDescr_t*/,
// cusparseZcsrsv2_analysis-NEXT:       value /*const cuDoubleComplex **/, row_ptr /*const int **/,
// cusparseZcsrsv2_analysis-NEXT:       col_idx /*const int **/, info /*csrsv2Info_t*/,
// cusparseZcsrsv2_analysis-NEXT:       policy /*cusparseSolvePolicy_t*/, buffer /*void **/);
// cusparseZcsrsv2_analysis-NEXT: Is migrated to:
// cusparseZcsrsv2_analysis-NEXT:   dpct::sparse::optimize_csrsv(handle->get_queue(), trans, m, desc, value, row_ptr, col_idx, info);
