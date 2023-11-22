// UNSUPPORTED: v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2
// UNSUPPORTED: cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2
// RUN: dpct --format-range=none --usm-level=none --out-root %T/cusparse %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cusparse/cusparse.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

int m, n, nnz, k, ldb, ldc;
float alpha;
const float* csrValA;
const int* csrRowPtrA;
const int* csrColIndA;
const float* x;
float beta;
float* y;
//CHECK: sycl::queue* handle;
//CHECK-NEXT: oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
//CHECK-NEXT: std::shared_ptr<dpct::sparse::matrix_info> descrA;
cusparseHandle_t handle;
cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
cusparseMatDescr_t descrA;

int main(){
  //CHECK: std::shared_ptr<dpct::sparse::matrix_info> descr1 = 0, descr2 = 0;
  //CHECK-NEXT:std::shared_ptr<dpct::sparse::matrix_info> descr3 = 0;
  //CHECK-NEXT: dpct::queue_ptr s;
  cusparseMatDescr_t descr1 = 0, descr2 = 0;
  cusparseMatDescr_t descr3 = 0;
  cudaStream_t s;

  //CHECK: int mode = 0;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseGetPointerMode was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseSetPointerMode was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  cusparsePointerMode_t mode = CUSPARSE_POINTER_MODE_HOST;
  cusparseGetPointerMode(handle, &mode);
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

  constexpr int zero = 0;
  //CHECK: oneapi::mkl::diag diag0 = oneapi::mkl::diag::nonunit;
  //CHECK-NEXT: oneapi::mkl::uplo fill0 = oneapi::mkl::uplo::lower;
  //CHECK-NEXT: oneapi::mkl::index_base base0 = oneapi::mkl::index_base::zero;
  //CHECK-NEXT: dpct::sparse::matrix_info::matrix_type type0 = dpct::sparse::matrix_info::matrix_type::ge;
  //CHECK-NEXT: descrA->set_diag((oneapi::mkl::diag)zero);
  //CHECK-NEXT: descrA->set_uplo((oneapi::mkl::uplo)zero);
  //CHECK-NEXT: descrA->set_index_base((oneapi::mkl::index_base)zero);
  //CHECK-NEXT: descrA->set_matrix_type((dpct::sparse::matrix_info::matrix_type)zero);
  //CHECK-NEXT: diag0 = descrA->get_diag();
  //CHECK-NEXT: fill0 = descrA->get_uplo();
  //CHECK-NEXT: base0 = descrA->get_index_base();
  //CHECK-NEXT: type0 = descrA->get_matrix_type();
  cusparseDiagType_t diag0 = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseFillMode_t fill0 = CUSPARSE_FILL_MODE_LOWER;
  cusparseIndexBase_t base0 = CUSPARSE_INDEX_BASE_ZERO;
  cusparseMatrixType_t type0 = CUSPARSE_MATRIX_TYPE_GENERAL;
  cusparseSetMatDiagType(descrA, (cusparseDiagType_t)zero);
  cusparseSetMatFillMode(descrA, (cusparseFillMode_t)zero);
  cusparseSetMatIndexBase(descrA, (cusparseIndexBase_t)zero);
  cusparseSetMatType(descrA, (cusparseMatrixType_t)zero);
  diag0 = cusparseGetMatDiagType(descrA);
  fill0 = cusparseGetMatFillMode(descrA);
  base0 = cusparseGetMatIndexBase(descrA);
  type0 = cusparseGetMatType(descrA);

  //CHECK: handle = &dpct::get_out_of_order_queue();
  //CHECK-NEXT: handle = s;
  //CHECK-NEXT: s = handle;
  cusparseCreate(&handle);
  cusparseSetStream(handle,s);
  cusparseGetStream(handle,&s);

  //CHECK: descrA = std::make_shared<dpct::sparse::matrix_info>();
  //CHECK-NEXT: descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);
  //CHECK-NEXT: descrA->set_index_base(oneapi::mkl::index_base::zero);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmv(*handle, (oneapi::mkl::transpose)zero, m, n, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseScsrmv(handle, (cusparseOperation_t)zero, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);

  cuComplex alpha_C, beta_C, *csrValA_C, *x_C, *y_C;

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmv(*handle, transA, m, n, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, &beta_C, y_C);
  cusparseCcsrmv(handle, transA, m, n, nnz, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, &beta_C, y_C);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmv(*handle, transA, m, n, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
  cusparseScsrmv_mp(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmv(*handle, transA, m, n, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, &beta_C, y_C);
  cusparseCcsrmv_mp(handle, transA, m, n, nnz, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, &beta_C, y_C);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmm(*handle, transA, m, n, k, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);
  cusparseScsrmm(handle, transA, m, n, k, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmm(*handle, transA, m, n, k, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, ldb, &beta_C, y_C, ldc);
  cusparseCcsrmm(handle, transA, m, n, k, nnz, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, ldb, &beta_C, y_C, ldc);

  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmm(*handle, transA, transB, m, n, k, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);
  cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: dpct::sparse::csrmm(*handle, transA, transB, m, n, k, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, ldb, &beta_C, y_C, ldc);
  cusparseCcsrmm2(handle, transA, transB, m, n, k, nnz, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, ldb, &beta_C, y_C, ldc);

  //CHECK:int status;
  cusparseStatus_t status;

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: if(status = DPCT_CHECK_ERROR(dpct::sparse::csrmv(*handle, transA, m, n, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y))){}
  if(status = cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: for(status = DPCT_CHECK_ERROR(dpct::sparse::csrmv(*handle, transA, m, n, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y));;){}
  for(status = cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);;){}

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: switch(status = DPCT_CHECK_ERROR(dpct::sparse::csrmv(*handle, transA, m, n, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y))){}
  switch(status = cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  //CHECK: std::shared_ptr<dpct::sparse::optimize_info> info;
  //CHECK-NEXT: info = std::make_shared<dpct::sparse::optimize_info>();
  //CHECK-NEXT: dpct::sparse::optimize_csrsv(*handle, transA, m, descrA, csrValA, csrRowPtrA, csrColIndA, info);
  //CHECK-NEXT: info.reset();
  cusparseSolveAnalysisInfo_t info;
  cusparseCreateSolveAnalysisInfo(&info);
  cusparseScsrsv_analysis(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info);
  cusparseDestroySolveAnalysisInfo(info);

  //CHECK: dpct::sparse::optimize_csrsv(*handle, transA, m, descrA, csrValA_C, csrRowPtrA, csrColIndA, info);
  cusparseCcsrsv_analysis(handle, transA, m, nnz, descrA, csrValA_C, csrRowPtrA, csrColIndA, info);

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDestroyMatDescr was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: handle = nullptr;
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
}

//CHECK: int foo(std::shared_ptr<dpct::sparse::matrix_info> descrB) try {
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  return DPCT_CHECK_ERROR(dpct::sparse::csrmv(*handle, transA, m, n, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y));
//CHECK-NEXT:}
int foo(cusparseMatDescr_t descrB){
  return cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}

//CHECK: void foo2(std::shared_ptr<dpct::sparse::matrix_info> descrB){
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general/symmetric/triangular sparse matrix type. You may need to adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  dpct::sparse::csrmv(*handle, transA, m, n, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
//CHECK-NEXT:}
void foo2(cusparseMatDescr_t descrB){
  cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}

void foo3(){
  int c_nnz;
  cusparseMatDescr_t descrB;
  cusparseMatDescr_t descrC;

  const float* val_a_s;
  const double* val_a_d;
  const float2* val_a_c;
  const double2* val_a_z;
  const int* row_ptr_a;
  const int* col_ind_a;

  const float* val_b_s;
  const double* val_b_d;
  const float2* val_b_c;
  const double2* val_b_z;

  float* val_c_s;
  double* val_c_d;
  float2* val_c_c;
  double2* val_c_z;

  const float* alpha_s;
  const double* alpha_d;
  const float2* alpha_c;
  const double2* alpha_z;

  const float* beta_s;
  const double* beta_d;
  const float2* beta_c;
  const double2* beta_z;

  // CHECK: /*
  // CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_s, descrA, val_a_s, row_ptr_a, col_ind_a, val_b_s, 5, beta_s, val_c_s, 4);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_d, descrA, val_a_d, row_ptr_a, col_ind_a, val_b_d, 5, beta_d, val_c_d, 4);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_c, descrA, val_a_c, row_ptr_a, col_ind_a, val_b_c, 5, beta_c, val_c_c, 4);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_z, descrA, val_a_z, row_ptr_a, col_ind_a, val_b_z, 5, beta_z, val_c_z, 4);
  cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 2, 5, 9, alpha_s, descrA, val_a_s, row_ptr_a, col_ind_a, val_b_s, 5, beta_s, val_c_s, 4);
  cusparseDcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 2, 5, 9, alpha_d, descrA, val_a_d, row_ptr_a, col_ind_a, val_b_d, 5, beta_d, val_c_d, 4);
  cusparseCcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 2, 5, 9, alpha_c, descrA, val_a_c, row_ptr_a, col_ind_a, val_b_c, 5, beta_c, val_c_c, 4);
  cusparseZcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 2, 5, 9, alpha_z, descrA, val_a_z, row_ptr_a, col_ind_a, val_b_z, 5, beta_z, val_c_z, 4);
}

void foo4(){
  int c_nnz;
  cusparseMatDescr_t descrB;
  cusparseMatDescr_t descrC;

  const float* val_a_s;
  const double* val_a_d;
  const float2* val_a_c;
  const double2* val_a_z;
  const int* row_ptr_a;
  const int* col_ind_a;

  const float* val_b_s;
  const double* val_b_d;
  const float2* val_b_c;
  const double2* val_b_z;
  const int* row_ptr_b;
  const int* col_ind_b;

  float* val_c_s;
  double* val_c_d;
  float2* val_c_c;
  double2* val_c_z;
  int* row_ptr_c;
  int* col_ind_c;

  // CHECK: dpct::sparse::csrgemm_nnz_estimiate(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 3, 3, 4, descrA, 4, row_ptr_a, col_ind_a, descrB, 5, row_ptr_b, col_ind_b, descrC, row_ptr_c, &c_nnz);
  // CHECK-NEXT: dpct::sparse::csrgemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 3, 3, 4, descrA, val_a_s, row_ptr_a, col_ind_a, descrB, val_b_s, row_ptr_b, col_ind_b, descrC, val_c_s, row_ptr_c, col_ind_c);
  // CHECK-NEXT: dpct::sparse::csrgemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 3, 3, 4, descrA, val_a_d, row_ptr_a, col_ind_a, descrB, val_b_d, row_ptr_b, col_ind_b, descrC, val_c_d, row_ptr_c, col_ind_c);
  // CHECK-NEXT: dpct::sparse::csrgemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 3, 3, 4, descrA, val_a_c, row_ptr_a, col_ind_a, descrB, val_b_c, row_ptr_b, col_ind_b, descrC, val_c_c, row_ptr_c, col_ind_c);
  // CHECK-NEXT: dpct::sparse::csrgemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 3, 3, 4, descrA, val_a_z, row_ptr_a, col_ind_a, descrB, val_b_z, row_ptr_b, col_ind_b, descrC, val_c_z, row_ptr_c, col_ind_c);
  cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 3, 4, descrA, 4, row_ptr_a,col_ind_a, descrB, 5, row_ptr_b, col_ind_b, descrC, row_ptr_c, &c_nnz);
  cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 3, 4, descrA, 4, val_a_s, row_ptr_a, col_ind_a, descrB, 5, val_b_s, row_ptr_b, col_ind_b, descrC, val_c_s, row_ptr_c, col_ind_c);
  cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 3, 4, descrA, 4, val_a_d, row_ptr_a, col_ind_a, descrB, 5, val_b_d, row_ptr_b, col_ind_b, descrC, val_c_d, row_ptr_c, col_ind_c);
  cusparseCcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 3, 4, descrA, 4, val_a_c, row_ptr_a, col_ind_a, descrB, 5, val_b_c, row_ptr_b, col_ind_b, descrC, val_c_c, row_ptr_c, col_ind_c);
  cusparseZcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 3, 4, descrA, 4, val_a_z, row_ptr_a, col_ind_a, descrB, 5, val_b_z, row_ptr_b, col_ind_b, descrC, val_c_z, row_ptr_c, col_ind_c);
}

