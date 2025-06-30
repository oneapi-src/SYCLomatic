// UNSUPPORTED: cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6, cuda-12.8, cuda-12.9
// UNSUPPORTED: v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6, v12.8, v12.9

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreateCsrgemm2Info | FileCheck %s -check-prefix=cusparseCreateCsrgemm2Info
// cusparseCreateCsrgemm2Info: CUDA API:
// cusparseCreateCsrgemm2Info-NEXT:   csrgemm2Info_t info;
// cusparseCreateCsrgemm2Info-NEXT:   cusparseCreateCsrgemm2Info(&info /*csrgemm2Info_t **/);
// cusparseCreateCsrgemm2Info-NEXT: Is migrated to:
// cusparseCreateCsrgemm2Info-NEXT:   std::shared_ptr<dpct::sparse::csrgemm2_info> info;
// cusparseCreateCsrgemm2Info-NEXT:   info = std::make_shared<dpct::sparse::csrgemm2_info>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDestroyCsrgemm2Info | FileCheck %s -check-prefix=cusparseDestroyCsrgemm2Info
// cusparseDestroyCsrgemm2Info: CUDA API:
// cusparseDestroyCsrgemm2Info-NEXT:   cusparseDestroyCsrgemm2Info(info /*csrgemm2Info_t*/);
// cusparseDestroyCsrgemm2Info-NEXT: Is migrated to:
// cusparseDestroyCsrgemm2Info-NEXT:   info.reset();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrgemm2_bufferSizeExt | FileCheck %s -check-prefix=cusparseScsrgemm2_bufferSizeExt
// cusparseScsrgemm2_bufferSizeExt: CUDA API:
// cusparseScsrgemm2_bufferSizeExt-NEXT:   cusparseScsrgemm2_bufferSizeExt(
// cusparseScsrgemm2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseScsrgemm2_bufferSizeExt-NEXT:       alpha /*const float **/, descr_a /*const cusparseMatDescr_t*/,
// cusparseScsrgemm2_bufferSizeExt-NEXT:       nnz_a /*int*/, row_ptr_a /*const int **/, col_idx_a /*const int **/,
// cusparseScsrgemm2_bufferSizeExt-NEXT:       descr_b /*const cusparseMatDescr_t*/, nnz_b /*int*/,
// cusparseScsrgemm2_bufferSizeExt-NEXT:       row_ptr_b /*const int **/, col_idx_b /*const int **/,
// cusparseScsrgemm2_bufferSizeExt-NEXT:       beta /*const float **/, descr_d /*const cusparseMatDescr_t*/,
// cusparseScsrgemm2_bufferSizeExt-NEXT:       nnz_d /*int*/, row_ptr_d /*const int **/, col_idx_d /*const int **/,
// cusparseScsrgemm2_bufferSizeExt-NEXT:       info /*csrgemm2Info_t*/, buffer_size /*size_t **/);
// cusparseScsrgemm2_bufferSizeExt-NEXT: Is migrated to:
// cusparseScsrgemm2_bufferSizeExt-NEXT:   dpct::sparse::csrgemm2_get_buffer_size<float>(handle, m, n, k, alpha, descr_a, nnz_a, row_ptr_a, col_idx_a, descr_b, nnz_b, row_ptr_b, col_idx_b, beta, descr_d, nnz_d, row_ptr_d, col_idx_d, info, buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrgemm2_bufferSizeExt | FileCheck %s -check-prefix=cusparseDcsrgemm2_bufferSizeExt
// cusparseDcsrgemm2_bufferSizeExt: CUDA API:
// cusparseDcsrgemm2_bufferSizeExt-NEXT:   cusparseDcsrgemm2_bufferSizeExt(
// cusparseDcsrgemm2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseDcsrgemm2_bufferSizeExt-NEXT:       alpha /*const double **/, descr_a /*const cusparseMatDescr_t*/,
// cusparseDcsrgemm2_bufferSizeExt-NEXT:       nnz_a /*int*/, row_ptr_a /*const int **/, col_idx_a /*const int **/,
// cusparseDcsrgemm2_bufferSizeExt-NEXT:       descr_b /*const cusparseMatDescr_t*/, nnz_b /*int*/,
// cusparseDcsrgemm2_bufferSizeExt-NEXT:       row_ptr_b /*const int **/, col_idx_b /*const int **/,
// cusparseDcsrgemm2_bufferSizeExt-NEXT:       beta /*const double **/, descr_d /*const cusparseMatDescr_t*/,
// cusparseDcsrgemm2_bufferSizeExt-NEXT:       nnz_d /*int*/, row_ptr_d /*const int **/, col_idx_d /*const int **/,
// cusparseDcsrgemm2_bufferSizeExt-NEXT:       info /*csrgemm2Info_t*/, buffer_size /*size_t **/);
// cusparseDcsrgemm2_bufferSizeExt-NEXT: Is migrated to:
// cusparseDcsrgemm2_bufferSizeExt-NEXT:   dpct::sparse::csrgemm2_get_buffer_size<double>(handle, m, n, k, alpha, descr_a, nnz_a, row_ptr_a, col_idx_a, descr_b, nnz_b, row_ptr_b, col_idx_b, beta, descr_d, nnz_d, row_ptr_d, col_idx_d, info, buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrgemm2_bufferSizeExt | FileCheck %s -check-prefix=cusparseCcsrgemm2_bufferSizeExt
// cusparseCcsrgemm2_bufferSizeExt: CUDA API:
// cusparseCcsrgemm2_bufferSizeExt-NEXT:   cusparseCcsrgemm2_bufferSizeExt(
// cusparseCcsrgemm2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseCcsrgemm2_bufferSizeExt-NEXT:       alpha /*const cuComplex **/, descr_a /*const cusparseMatDescr_t*/,
// cusparseCcsrgemm2_bufferSizeExt-NEXT:       nnz_a /*int*/, row_ptr_a /*const int **/, col_idx_a /*const int **/,
// cusparseCcsrgemm2_bufferSizeExt-NEXT:       descr_b /*const cusparseMatDescr_t*/, nnz_b /*int*/,
// cusparseCcsrgemm2_bufferSizeExt-NEXT:       row_ptr_b /*const int **/, col_idx_b /*const int **/,
// cusparseCcsrgemm2_bufferSizeExt-NEXT:       beta /*const cuComplex **/, descr_d /*const cusparseMatDescr_t*/,
// cusparseCcsrgemm2_bufferSizeExt-NEXT:       nnz_d /*int*/, row_ptr_d /*const int **/, col_idx_d /*const int **/,
// cusparseCcsrgemm2_bufferSizeExt-NEXT:       info /*csrgemm2Info_t*/, buffer_size /*size_t **/);
// cusparseCcsrgemm2_bufferSizeExt-NEXT: Is migrated to:
// cusparseCcsrgemm2_bufferSizeExt-NEXT:   dpct::sparse::csrgemm2_get_buffer_size<sycl::float2>(handle, m, n, k, alpha, descr_a, nnz_a, row_ptr_a, col_idx_a, descr_b, nnz_b, row_ptr_b, col_idx_b, beta, descr_d, nnz_d, row_ptr_d, col_idx_d, info, buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrgemm2_bufferSizeExt | FileCheck %s -check-prefix=cusparseZcsrgemm2_bufferSizeExt
// cusparseZcsrgemm2_bufferSizeExt: CUDA API:
// cusparseZcsrgemm2_bufferSizeExt-NEXT:   cusparseZcsrgemm2_bufferSizeExt(
// cusparseZcsrgemm2_bufferSizeExt-NEXT:       handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseZcsrgemm2_bufferSizeExt-NEXT:       alpha /*const cuDoubleComplex **/, descr_a /*const cusparseMatDescr_t*/,
// cusparseZcsrgemm2_bufferSizeExt-NEXT:       nnz_a /*int*/, row_ptr_a /*const int **/, col_idx_a /*const int **/,
// cusparseZcsrgemm2_bufferSizeExt-NEXT:       descr_b /*const cusparseMatDescr_t*/, nnz_b /*int*/,
// cusparseZcsrgemm2_bufferSizeExt-NEXT:       row_ptr_b /*const int **/, col_idx_b /*const int **/,
// cusparseZcsrgemm2_bufferSizeExt-NEXT:       beta /*const cuDoubleComplex **/, descr_d /*const cusparseMatDescr_t*/,
// cusparseZcsrgemm2_bufferSizeExt-NEXT:       nnz_d /*int*/, row_ptr_d /*const int **/, col_idx_d /*const int **/,
// cusparseZcsrgemm2_bufferSizeExt-NEXT:       info /*csrgemm2Info_t*/, buffer_size /*size_t **/);
// cusparseZcsrgemm2_bufferSizeExt-NEXT: Is migrated to:
// cusparseZcsrgemm2_bufferSizeExt-NEXT:   dpct::sparse::csrgemm2_get_buffer_size<sycl::double2>(handle, m, n, k, alpha, descr_a, nnz_a, row_ptr_a, col_idx_a, descr_b, nnz_b, row_ptr_b, col_idx_b, beta, descr_d, nnz_d, row_ptr_d, col_idx_d, info, buffer_size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseXcsrgemm2Nnz | FileCheck %s -check-prefix=cusparseXcsrgemm2Nnz
// cusparseXcsrgemm2Nnz: CUDA API:
// cusparseXcsrgemm2Nnz-NEXT:   cusparseXcsrgemm2Nnz(
// cusparseXcsrgemm2Nnz-NEXT:       handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseXcsrgemm2Nnz-NEXT:       descr_a /*const cusparseMatDescr_t*/, nnz_a /*int*/,
// cusparseXcsrgemm2Nnz-NEXT:       row_ptr_a /*const int **/, col_idx_a /*const int **/,
// cusparseXcsrgemm2Nnz-NEXT:       descr_b /*const cusparseMatDescr_t*/, nnz_b /*int*/,
// cusparseXcsrgemm2Nnz-NEXT:       row_ptr_b /*const int **/, col_idx_b /*const int **/,
// cusparseXcsrgemm2Nnz-NEXT:       descr_d /*const cusparseMatDescr_t*/, nnz_d /*int*/,
// cusparseXcsrgemm2Nnz-NEXT:       row_ptr_d /*const int **/, col_idx_d /*const int **/,
// cusparseXcsrgemm2Nnz-NEXT:       descr_c /*const cusparseMatDescr_t*/, row_ptr_c /*int **/, nnz /*int **/,
// cusparseXcsrgemm2Nnz-NEXT:       info /*const csrgemm2Info_t*/, buffer /*void **/);
// cusparseXcsrgemm2Nnz-NEXT: Is migrated to:
// cusparseXcsrgemm2Nnz-NEXT:   dpct::sparse::csrgemm2_nnz(handle, m, n, k, descr_a, nnz_a, row_ptr_a, col_idx_a, descr_b, nnz_b, row_ptr_b, col_idx_b, descr_d, nnz_d, row_ptr_d, col_idx_d, descr_c, row_ptr_c, nnz, info, buffer);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseScsrgemm2 | FileCheck %s -check-prefix=cusparseScsrgemm2
// cusparseScsrgemm2: CUDA API:
// cusparseScsrgemm2-NEXT:   cusparseScsrgemm2(
// cusparseScsrgemm2-NEXT:       handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseScsrgemm2-NEXT:       alpha /*const float **/, descr_a /*const cusparseMatDescr_t*/,
// cusparseScsrgemm2-NEXT:       nnz_a /*int*/, value_a /*const float **/, row_ptr_a /*const int **/,
// cusparseScsrgemm2-NEXT:       col_idx_a /*const int **/, descr_b /*const cusparseMatDescr_t*/,
// cusparseScsrgemm2-NEXT:       nnz_b /*int*/, value_b /*const float **/, row_ptr_b /*const int **/,
// cusparseScsrgemm2-NEXT:       col_idx_b /*const int **/, beta /*const float **/,
// cusparseScsrgemm2-NEXT:       descr_d /*const cusparseMatDescr_t*/, nnz_d /*int*/,
// cusparseScsrgemm2-NEXT:       value_d /*const float **/, row_ptr_d /*const int **/,
// cusparseScsrgemm2-NEXT:       col_idx_d /*const int **/, descr_c /*const cusparseMatDescr_t*/,
// cusparseScsrgemm2-NEXT:       value_c /*float **/, row_ptr_c /*const int **/, col_idx_c /*int **/,
// cusparseScsrgemm2-NEXT:       info /*const csrgemm2Info_t*/, buffer /*void **/);
// cusparseScsrgemm2-NEXT: Is migrated to:
// cusparseScsrgemm2-NEXT:   dpct::sparse::csrgemm2<float>(handle, m, n, k, alpha, descr_a, nnz_a, value_a, row_ptr_a, col_idx_a, descr_b, nnz_b, value_b, row_ptr_b, col_idx_b, beta, descr_d, nnz_d, value_d, row_ptr_d, col_idx_d, descr_c, value_c, row_ptr_c, col_idx_c, info, buffer);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDcsrgemm2 | FileCheck %s -check-prefix=cusparseDcsrgemm2
// cusparseDcsrgemm2: CUDA API:
// cusparseDcsrgemm2-NEXT:   cusparseDcsrgemm2(
// cusparseDcsrgemm2-NEXT:       handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseDcsrgemm2-NEXT:       alpha /*const double **/, descr_a /*const cusparseMatDescr_t*/,
// cusparseDcsrgemm2-NEXT:       nnz_a /*int*/, value_a /*const double **/, row_ptr_a /*const int **/,
// cusparseDcsrgemm2-NEXT:       col_idx_a /*const int **/, descr_b /*const cusparseMatDescr_t*/,
// cusparseDcsrgemm2-NEXT:       nnz_b /*int*/, value_b /*const double **/, row_ptr_b /*const int **/,
// cusparseDcsrgemm2-NEXT:       col_idx_b /*const int **/, beta /*const double **/,
// cusparseDcsrgemm2-NEXT:       descr_d /*const cusparseMatDescr_t*/, nnz_d /*int*/,
// cusparseDcsrgemm2-NEXT:       value_d /*const double **/, row_ptr_d /*const int **/,
// cusparseDcsrgemm2-NEXT:       col_idx_d /*const int **/, descr_c /*const cusparseMatDescr_t*/,
// cusparseDcsrgemm2-NEXT:       value_c /*double **/, row_ptr_c /*const int **/, col_idx_c /*int **/,
// cusparseDcsrgemm2-NEXT:       info /*const csrgemm2Info_t*/, buffer /*void **/);
// cusparseDcsrgemm2-NEXT: Is migrated to:
// cusparseDcsrgemm2-NEXT:   dpct::sparse::csrgemm2<double>(handle, m, n, k, alpha, descr_a, nnz_a, value_a, row_ptr_a, col_idx_a, descr_b, nnz_b, value_b, row_ptr_b, col_idx_b, beta, descr_d, nnz_d, value_d, row_ptr_d, col_idx_d, descr_c, value_c, row_ptr_c, col_idx_c, info, buffer);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCcsrgemm2 | FileCheck %s -check-prefix=cusparseCcsrgemm2
// cusparseCcsrgemm2: CUDA API:
// cusparseCcsrgemm2-NEXT:   cusparseCcsrgemm2(
// cusparseCcsrgemm2-NEXT:       handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseCcsrgemm2-NEXT:       alpha /*const cuComplex **/, descr_a /*const cusparseMatDescr_t*/,
// cusparseCcsrgemm2-NEXT:       nnz_a /*int*/, value_a /*const cuComplex **/, row_ptr_a /*const int **/,
// cusparseCcsrgemm2-NEXT:       col_idx_a /*const int **/, descr_b /*const cusparseMatDescr_t*/,
// cusparseCcsrgemm2-NEXT:       nnz_b /*int*/, value_b /*const cuComplex **/, row_ptr_b /*const int **/,
// cusparseCcsrgemm2-NEXT:       col_idx_b /*const int **/, beta /*const cuComplex **/,
// cusparseCcsrgemm2-NEXT:       descr_d /*const cusparseMatDescr_t*/, nnz_d /*int*/,
// cusparseCcsrgemm2-NEXT:       value_d /*const cuComplex **/, row_ptr_d /*const int **/,
// cusparseCcsrgemm2-NEXT:       col_idx_d /*const int **/, descr_c /*const cusparseMatDescr_t*/,
// cusparseCcsrgemm2-NEXT:       value_c /*cuComplex **/, row_ptr_c /*const int **/, col_idx_c /*int **/,
// cusparseCcsrgemm2-NEXT:       info /*const csrgemm2Info_t*/, buffer /*void **/);
// cusparseCcsrgemm2-NEXT: Is migrated to:
// cusparseCcsrgemm2-NEXT:   dpct::sparse::csrgemm2<sycl::float2>(handle, m, n, k, alpha, descr_a, nnz_a, value_a, row_ptr_a, col_idx_a, descr_b, nnz_b, value_b, row_ptr_b, col_idx_b, beta, descr_d, nnz_d, value_d, row_ptr_d, col_idx_d, descr_c, value_c, row_ptr_c, col_idx_c, info, buffer);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseZcsrgemm2 | FileCheck %s -check-prefix=cusparseZcsrgemm2
// cusparseZcsrgemm2: CUDA API:
// cusparseZcsrgemm2-NEXT:   cusparseZcsrgemm2(
// cusparseZcsrgemm2-NEXT:       handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, k /*int*/,
// cusparseZcsrgemm2-NEXT:       alpha /*const cuDoubleComplex **/, descr_a /*const cusparseMatDescr_t*/,
// cusparseZcsrgemm2-NEXT:       nnz_a /*int*/, value_a /*const cuDoubleComplex **/,
// cusparseZcsrgemm2-NEXT:       row_ptr_a /*const int **/, col_idx_a /*const int **/,
// cusparseZcsrgemm2-NEXT:       descr_b /*const cusparseMatDescr_t*/, nnz_b /*int*/,
// cusparseZcsrgemm2-NEXT:       value_b /*const cuDoubleComplex **/, row_ptr_b /*const int **/,
// cusparseZcsrgemm2-NEXT:       col_idx_b /*const int **/, beta /*const cuDoubleComplex **/,
// cusparseZcsrgemm2-NEXT:       descr_d /*const cusparseMatDescr_t*/, nnz_d /*int*/,
// cusparseZcsrgemm2-NEXT:       value_d /*const cuDoubleComplex **/, row_ptr_d /*const int **/,
// cusparseZcsrgemm2-NEXT:       col_idx_d /*const int **/, descr_c /*const cusparseMatDescr_t*/,
// cusparseZcsrgemm2-NEXT:       value_c /*cuDoubleComplex **/, row_ptr_c /*const int **/,
// cusparseZcsrgemm2-NEXT:       col_idx_c /*int **/, info /*const csrgemm2Info_t*/, buffer /*void **/);
// cusparseZcsrgemm2-NEXT: Is migrated to:
// cusparseZcsrgemm2-NEXT:   dpct::sparse::csrgemm2<sycl::double2>(handle, m, n, k, alpha, descr_a, nnz_a, value_a, row_ptr_a, col_idx_a, descr_b, nnz_b, value_b, row_ptr_b, col_idx_b, beta, descr_d, nnz_d, value_d, row_ptr_d, col_idx_d, descr_c, value_c, row_ptr_c, col_idx_c, info, buffer);
