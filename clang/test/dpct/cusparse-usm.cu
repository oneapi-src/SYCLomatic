// UNSUPPORTED: v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2
// UNSUPPORTED: cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2
// RUN: dpct --format-range=none --out-root %T/cusparse-usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusparse-usm/cusparse-usm.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

int m, n, nnz, k, ldb, ldc;
double alpha;
const double* csrValA;
const int* csrRowPtrA;
const int* csrColIndA;
const double* x;
double beta;
double* y;
//CHECK: sycl::queue* handle;
//CHECK-NEXT: oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
//CHECK-NEXT: std::shared_ptr<dpct::sparse::matrix_info> descrA;
cusparseHandle_t handle;
cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
cusparseMatDescr_t descrA;

int foo(int aaaaa){
  //CHECK: std::shared_ptr<dpct::sparse::matrix_info> descr1 = 0, descr2 = 0;
  //CHECK-NEXT: std::shared_ptr<dpct::sparse::matrix_info> descr3 = 0;
  cusparseMatDescr_t descr1 = 0, descr2 = 0;
  cusparseMatDescr_t descr3 = 0;

  //CHECK: int mode = 1;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseGetPointerMode was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseSetPointerMode was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  cusparsePointerMode_t mode = CUSPARSE_POINTER_MODE_DEVICE;
  cusparseGetPointerMode(handle, &mode);
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

  //CHECK: oneapi::mkl::diag diag0 = oneapi::mkl::diag::nonunit;
  //CHECK-NEXT: oneapi::mkl::uplo fill0 = oneapi::mkl::uplo::lower;
  //CHECK-NEXT: oneapi::mkl::index_base base0 = oneapi::mkl::index_base::zero;
  //CHECK-NEXT: dpct::sparse::matrix_info::matrix_type type0 = dpct::sparse::matrix_info::matrix_type::ge;
  //CHECK-NEXT: descrA->set_diag((oneapi::mkl::diag)aaaaa);
  //CHECK-NEXT: descrA->set_uplo((oneapi::mkl::uplo)aaaaa);
  //CHECK-NEXT: descrA->set_index_base((oneapi::mkl::index_base)aaaaa);
  //CHECK-NEXT: descrA->set_matrix_type((dpct::sparse::matrix_info::matrix_type)aaaaa);
  //CHECK-NEXT: diag0 = descrA->get_diag();
  //CHECK-NEXT: fill0 = descrA->get_uplo();
  //CHECK-NEXT: base0 = descrA->get_index_base();
  //CHECK-NEXT: type0 = descrA->get_matrix_type();
  cusparseDiagType_t diag0 = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseFillMode_t fill0 = CUSPARSE_FILL_MODE_LOWER;
  cusparseIndexBase_t base0 = CUSPARSE_INDEX_BASE_ZERO;
  cusparseMatrixType_t type0 = CUSPARSE_MATRIX_TYPE_GENERAL;
  cusparseSetMatDiagType(descrA, (cusparseDiagType_t)aaaaa);
  cusparseSetMatFillMode(descrA, (cusparseFillMode_t)aaaaa);
  cusparseSetMatIndexBase(descrA, (cusparseIndexBase_t)aaaaa);
  cusparseSetMatType(descrA, (cusparseMatrixType_t)aaaaa);
  diag0 = cusparseGetMatDiagType(descrA);
  fill0 = cusparseGetMatFillMode(descrA);
  base0 = cusparseGetMatIndexBase(descrA);
  type0 = cusparseGetMatType(descrA);

  //CHECK: handle = &dpct::get_in_order_queue();
  //CHECK-NEXT: descrA = std::make_shared<dpct::sparse::matrix_info>();
  //CHECK-NEXT: descrA->set_matrix_type((dpct::sparse::matrix_info::matrix_type)aaaaa);
  //CHECK-NEXT: descrA->set_index_base(oneapi::mkl::index_base::zero);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmv(*handle, (oneapi::mkl::transpose)aaaaa, m, n, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, (cusparseMatrixType_t)aaaaa);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseDcsrmv(handle, (cusparseOperation_t)aaaaa, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);

  cuDoubleComplex alpha_Z, beta_Z, *csrValA_Z, *x_Z, *y_Z;

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmv(*handle, transA, m, n, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, &beta_Z, y_Z);
  cusparseZcsrmv(handle, transA, m, n, nnz, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, &beta_Z, y_Z);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmv(*handle, transA, m, n, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
  cusparseDcsrmv_mp(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmv(*handle, transA, m, n, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, &beta_Z, y_Z);
  cusparseZcsrmv_mp(handle, transA, m, n, nnz, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, &beta_Z, y_Z);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmm(*handle, transA, m, n, k, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);
  cusparseDcsrmm(handle, transA, m, n, k, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmm(*handle, transA, m, n, k, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, ldb, &beta_Z, y_Z, ldc);
  cusparseZcsrmm(handle, transA, m, n, k, nnz, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, ldb, &beta_Z, y_Z, ldc);

  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmm(*handle, transA, transB, m, n, k, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);
  cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmm(*handle, transA, transB, m, n, k, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, ldb, &beta_Z, y_Z, ldc);
  cusparseZcsrmm2(handle, transA, transB, m, n, k, nnz, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, ldb, &beta_Z, y_Z, ldc);

  size_t ws_size;
  void *ws;
  //CHECK: int alg = 0;
  //CHECK-NEXT: alg = 1;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseCsrmvEx_bufferSize was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmv(*handle, transA, m, n, &alpha, dpct::library_data_t::real_double, descrA, csrValA, dpct::library_data_t::real_double, csrRowPtrA, csrColIndA, x, dpct::library_data_t::real_double, &beta, dpct::library_data_t::real_double, y, dpct::library_data_t::real_double);
  cusparseAlgMode_t alg = CUSPARSE_ALG0;
  alg = CUSPARSE_ALG1;
  cusparseCsrmvEx_bufferSize(handle, alg, transA, m, n, nnz, &alpha, CUDA_R_64F, descrA, csrValA, CUDA_R_64F, csrRowPtrA, csrColIndA, x, CUDA_R_64F, &beta, CUDA_R_64F, y, CUDA_R_64F, CUDA_R_64F, &ws_size);
  cusparseCsrmvEx(handle, alg, transA, m, n, nnz, &alpha, CUDA_R_64F, descrA, csrValA, CUDA_R_64F, csrRowPtrA, csrColIndA, x, CUDA_R_64F, &beta, CUDA_R_64F, y, CUDA_R_64F, CUDA_R_64F, ws);

  //CHECK:int status;
  cusparseStatus_t status;

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: if(status = DPCT_CHECK_ERROR(dpct::sparse::csrmv(*handle, transA, m, n, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y))){}
  if(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: for(status = DPCT_CHECK_ERROR(dpct::sparse::csrmv(*handle, transA, m, n, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y));;){}
  for(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);;){}

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: switch(status = DPCT_CHECK_ERROR(dpct::sparse::csrmv(*handle, transA, m, n, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y))){}
  switch(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  //CHECK: std::shared_ptr<dpct::sparse::optimize_info> info;
  //CHECK-NEXT: info = std::make_shared<dpct::sparse::optimize_info>();
  //CHECK-NEXT: dpct::sparse::optimize_csrsv(*handle, transA, m, descrA, csrValA, csrRowPtrA, csrColIndA, info);
  //CHECK-NEXT: info.reset();
  cusparseSolveAnalysisInfo_t info;
  cusparseCreateSolveAnalysisInfo(&info);
  cusparseDcsrsv_analysis(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info);
  cusparseDestroySolveAnalysisInfo(info);

  //CHECK: dpct::sparse::optimize_csrsv(*handle, transA, m, descrA, csrValA_Z, csrRowPtrA, csrColIndA, info);
  cusparseZcsrsv_analysis(handle, transA, m, nnz, descrA, csrValA_Z, csrRowPtrA, csrColIndA, info);

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDestroyMatDescr was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: handle = nullptr;
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
}

//CHECK: int foo(std::shared_ptr<dpct::sparse::matrix_info> descrB) try {
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
//CHECK-NEXT:   */
//CHECK-NEXT:   return DPCT_CHECK_ERROR(dpct::sparse::csrmv(*handle, transA, m, n, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y));
//CHECK-NEXT: }
int foo(cusparseMatDescr_t descrB){
  return cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}

