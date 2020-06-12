// RUN: dpct --format-range=none --usm-level=none --out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusparse.dp.cpp --match-full-lines %s
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
//CHECK-NEXT: mkl::transpose transA = mkl::transpose::nontrans;
//CHECK-NEXT: dpct::mat_prop descrA;
cusparseHandle_t handle;
cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
cusparseMatDescr_t descrA;

int main(){
  //CHECK: dpct::mat_prop descr1 , descr2 ;
  //CHECK-NEXT:dpct::mat_prop descr3 ;
  //CHECK-NEXT: sycl::queue *s;
  cusparseMatDescr_t descr1 = 0, descr2 = 0;
  cusparseMatDescr_t descr3 = 0;
  cudaStream_t s;

  //CHECK: int mode = 0;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseGetPointerMode was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseSetPointerMode was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  cusparsePointerMode_t mode = CUSPARSE_POINTER_MODE_HOST;
  cusparseGetPointerMode(handle, &mode);
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

  int zero = 0;
  //CHECK: mkl::diag diag0 = mkl::diag::nonunit;
  //CHECK-NEXT: mkl::uplo fill0 = mkl::uplo::lower;
  //CHECK-NEXT: mkl::index_base base0 = mkl::index_base::zero;
  //CHECK-NEXT: dpct::mat_type type0 = dpct::mat_type::ge;
  //CHECK-NEXT: descrA.type = (mkl::diag)zero;
  //CHECK-NEXT: descrA.uplo = dpct::get_uplo(zero);
  //CHECK-NEXT: descrA.index = (mkl::index_base)zero;
  //CHECK-NEXT: descrA.type = (dpct::mat_type)zero;
  //CHECK-NEXT: diag0 = descrA.type;
  //CHECK-NEXT: fill0 = descrA.uplo;
  //CHECK-NEXT: base0 = descrA.index;
  //CHECK-NEXT: type0 = descrA.type;
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

  //CHECK: handle = &dpct::get_default_queue();
  //CHECK-NEXT: handle = s;
  //CHECK-NEXT: s = handle;
  cusparseCreate(&handle);
  cusparseSetStream(handle,s);
  cusparseGetStream(handle,&s);

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseCreateMatDescr was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: descrA.type = dpct::mat_type::ge;
  //CHECK-NEXT: descrA.index = mkl::index_base::zero;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The oneMKL library does not support Hermitian sparse matrix for computing a sparse matrix-dense vector product. If the martix type is unsupported, you need adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: {
  //CHECK-NEXT: auto csrValA_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(csrValA);
  //CHECK-NEXT: auto csrRowPtrA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrRowPtrA);
  //CHECK-NEXT: auto csrColIndA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrColIndA);
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA.index, csrRowPtrA_buf_ct{{[0-9]+}}, csrColIndA_buf_ct{{[0-9]+}}, csrValA_buf_ct{{[0-9]+}});
  //CHECK-NEXT: dpct::sparse_csr_mv_wrapper(*handle, dpct::get_transpose(zero), dpct::get_value(&alpha, *handle), descrA, mat_handle_ct{{[0-9]+}}, x, dpct::get_value(&beta, *handle), y);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: }
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseScsrmv(handle, (cusparseOperation_t)zero, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);

  cuComplex alpha_C, beta_C, *csrValA_C, *x_C, *y_C;

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The oneMKL library does not support Hermitian sparse matrix for computing a sparse matrix-dense vector product. If the martix type is unsupported, you need adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: {
  //CHECK-NEXT: auto csrValA_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(csrValA_C);
  //CHECK-NEXT: auto csrRowPtrA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrRowPtrA);
  //CHECK-NEXT: auto csrColIndA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrColIndA);
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA.index, csrRowPtrA_buf_ct{{[0-9]+}}, csrColIndA_buf_ct{{[0-9]+}}, csrValA_C_buf_ct{{[0-9]+}});
  //CHECK-NEXT: dpct::sparse_csr_mv_wrapper(*handle, transA, dpct::get_value(&alpha_C, *handle), descrA, mat_handle_ct{{[0-9]+}}, x_C, dpct::get_value(&beta_C, *handle), y_C);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: }
  cusparseCcsrmv(handle, transA, m, n, nnz, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, &beta_C, y_C);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The oneMKL library does not support symmetric/Hermitian/triangular sparse matrix for computing a sparse matrix-dense matrix product. If the martix type is unsupported, you need adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: {
  //CHECK-NEXT: auto csrValA_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(csrValA);
  //CHECK-NEXT: auto csrRowPtrA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrRowPtrA);
  //CHECK-NEXT: auto csrColIndA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrColIndA);
  //CHECK-NEXT: auto x_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x);
  //CHECK-NEXT: auto y_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y);
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, k, descrA.index, csrRowPtrA_buf_ct{{[0-9]+}}, csrColIndA_buf_ct{{[0-9]+}}, csrValA_buf_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::gemm(*handle, transA, dpct::get_value(&alpha, *handle), mat_handle_ct{{[0-9]+}}, x_buf_ct{{[0-9]+}}, n, ldb, dpct::get_value(&beta, *handle), y_buf_ct{{[0-9]+}}, ldc);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: }
  cusparseScsrmm(handle, transA, m, n, k, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The oneMKL library does not support symmetric/Hermitian/triangular sparse matrix for computing a sparse matrix-dense matrix product. If the martix type is unsupported, you need adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: {
  //CHECK-NEXT: auto csrValA_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(csrValA_C);
  //CHECK-NEXT: auto csrRowPtrA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrRowPtrA);
  //CHECK-NEXT: auto csrColIndA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrColIndA);
  //CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_C);
  //CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_C);
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, k, descrA.index, csrRowPtrA_buf_ct{{[0-9]+}}, csrColIndA_buf_ct{{[0-9]+}}, csrValA_C_buf_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::gemm(*handle, transA, dpct::get_value(&alpha_C, *handle), mat_handle_ct{{[0-9]+}}, x_C_buf_ct{{[0-9]+}}, n, ldb, dpct::get_value(&beta_C, *handle), y_C_buf_ct{{[0-9]+}}, ldc);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: }
  cusparseCcsrmm(handle, transA, m, n, k, nnz, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, ldb, &beta_C, y_C, ldc);

  //CHECK:int status;
  cusparseStatus_t status;

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The oneMKL library does not support Hermitian sparse matrix for computing a sparse matrix-dense vector product. If the martix type is unsupported, you need adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: {
  //CHECK-NEXT: auto csrValA_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(csrValA);
  //CHECK-NEXT: auto csrRowPtrA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrRowPtrA);
  //CHECK-NEXT: auto csrColIndA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrColIndA);
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA.index, csrRowPtrA_buf_ct{{[0-9]+}}, csrColIndA_buf_ct{{[0-9]+}}, csrValA_buf_ct{{[0-9]+}});
  //CHECK-NEXT: dpct::sparse_csr_mv_wrapper(*handle, transA, dpct::get_value(&alpha, *handle), descrA, mat_handle_ct{{[0-9]+}}, x, dpct::get_value(&beta, *handle), y);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: }
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: if(status = 0){}
  if(status = cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The oneMKL library does not support Hermitian sparse matrix for computing a sparse matrix-dense vector product. If the martix type is unsupported, you need adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: {
  //CHECK-NEXT: auto csrValA_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(csrValA);
  //CHECK-NEXT: auto csrRowPtrA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrRowPtrA);
  //CHECK-NEXT: auto csrColIndA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrColIndA);
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA.index, csrRowPtrA_buf_ct{{[0-9]+}}, csrColIndA_buf_ct{{[0-9]+}}, csrValA_buf_ct{{[0-9]+}});
  //CHECK-NEXT: dpct::sparse_csr_mv_wrapper(*handle, transA, dpct::get_value(&alpha, *handle), descrA, mat_handle_ct{{[0-9]+}}, x, dpct::get_value(&beta, *handle), y);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: }
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a for statement. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: for(status = 0;;){}
  for(status = cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);;){}

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The oneMKL library does not support Hermitian sparse matrix for computing a sparse matrix-dense vector product. If the martix type is unsupported, you need adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: {
  //CHECK-NEXT: auto csrValA_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(csrValA);
  //CHECK-NEXT: auto csrRowPtrA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrRowPtrA);
  //CHECK-NEXT: auto csrColIndA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrColIndA);
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA.index, csrRowPtrA_buf_ct{{[0-9]+}}, csrColIndA_buf_ct{{[0-9]+}}, csrValA_buf_ct{{[0-9]+}});
  //CHECK-NEXT: dpct::sparse_csr_mv_wrapper(*handle, transA, dpct::get_value(&alpha, *handle), descrA, mat_handle_ct{{[0-9]+}}, x, dpct::get_value(&beta, *handle), y);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: }
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a switch statement. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: switch(status = 0){}
  switch(status = cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  //CHECK: int info;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseCreateSolveAnalysisInfo was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseScsrsv_analysis was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The oneMKL library does not support general/symmetric/Hermitian sparse matrix for solveing a system of linear equations for the sparse matrix. If the martix type is unsupported, you need adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1046:{{[0-9]+}}: The argument &alpha in original API call was removed, because mkl::sparse::trsv does not support specifing the scalar value. If the scalar value in the original code is not 1, you need adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: {
  //CHECK-NEXT: auto csrValA_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(csrValA);
  //CHECK-NEXT: auto csrRowPtrA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrRowPtrA);
  //CHECK-NEXT: auto csrColIndA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrColIndA);
  //CHECK-NEXT: auto x_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x);
  //CHECK-NEXT: auto y_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(y);
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, m, descrA.index, csrRowPtrA_buf_ct{{[0-9]+}}, csrColIndA_buf_ct{{[0-9]+}}, csrValA_buf_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::optimize_trsv(*handle, descrA.uplo, transA, descrA.diag, mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::trsv(*handle, descrA.uplo, transA, descrA.diag, mat_handle_ct{{[0-9]+}}, x_buf_ct{{[0-9]+}}, y_buf_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: }
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDestroySolveAnalysisInfo was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  cusparseSolveAnalysisInfo_t info;
  cusparseCreateSolveAnalysisInfo(&info);
  cusparseScsrsv_analysis(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info);
  cusparseScsrsv_solve(handle, transA, m, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, x, y);
  cusparseDestroySolveAnalysisInfo(info);

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseCcsrsv_analysis was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The oneMKL library does not support general/symmetric/Hermitian sparse matrix for solveing a system of linear equations for the sparse matrix. If the martix type is unsupported, you need adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1046:{{[0-9]+}}: The argument &alpha_C in original API call was removed, because mkl::sparse::trsv does not support specifing the scalar value. If the scalar value in the original code is not 1, you need adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: {
  //CHECK-NEXT: auto csrValA_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(csrValA_C);
  //CHECK-NEXT: auto csrRowPtrA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrRowPtrA);
  //CHECK-NEXT: auto csrColIndA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrColIndA);
  //CHECK-NEXT: auto x_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_C);
  //CHECK-NEXT: auto y_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(y_C);
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, m, descrA.index, csrRowPtrA_buf_ct{{[0-9]+}}, csrColIndA_buf_ct{{[0-9]+}}, csrValA_C_buf_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::optimize_trsv(*handle, descrA.uplo, transA, descrA.diag, mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::trsv(*handle, descrA.uplo, transA, descrA.diag, mat_handle_ct{{[0-9]+}}, x_C_buf_ct{{[0-9]+}}, y_C_buf_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: }
  cusparseCcsrsv_analysis(handle, transA, m, nnz, descrA, csrValA_C, csrRowPtrA, csrColIndA, info);
  cusparseCcsrsv_solve(handle, transA, m, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, info, x_C, y_C);

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDestroyMatDescr was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: handle = nullptr;
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
}

//CHECK: int foo(dpct::mat_prop descrB) try {
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1045:{{[0-9]+}}: The oneMKL library does not support Hermitian sparse matrix for computing a sparse matrix-dense vector product. If the martix type is unsupported, you need adjust the code.
//CHECK-NEXT: */
//CHECK-NEXT: auto csrValA_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(csrValA);
//CHECK-NEXT: auto csrRowPtrA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrRowPtrA);
//CHECK-NEXT: auto csrColIndA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrColIndA);
//CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
//CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
//CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrB.index, csrRowPtrA_buf_ct{{[0-9]+}}, csrColIndA_buf_ct{{[0-9]+}}, csrValA_buf_ct{{[0-9]+}});
//CHECK-NEXT: dpct::sparse_csr_mv_wrapper(*handle, transA, dpct::get_value(&alpha, *handle), descrB, mat_handle_ct{{[0-9]+}}, x, dpct::get_value(&beta, *handle), y);
//CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a return statement. You may need to rewrite this code.
//CHECK-NEXT: */
//CHECK-NEXT: return 0;
//CHECK-NEXT: }
int foo(cusparseMatDescr_t descrB){
  return cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}

//CHECK: void foo2(dpct::mat_prop descrB){
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1045:{{[0-9]+}}: The oneMKL library does not support Hermitian sparse matrix for computing a sparse matrix-dense vector product. If the martix type is unsupported, you need adjust the code.
//CHECK-NEXT: */
//CHECK-NEXT: auto csrValA_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(csrValA);
//CHECK-NEXT: auto csrRowPtrA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrRowPtrA);
//CHECK-NEXT: auto csrColIndA_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(csrColIndA);
//CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
//CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
//CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrB.index, csrRowPtrA_buf_ct{{[0-9]+}}, csrColIndA_buf_ct{{[0-9]+}}, csrValA_buf_ct{{[0-9]+}});
//CHECK-NEXT: dpct::sparse_csr_mv_wrapper(*handle, transA, dpct::get_value(&alpha, *handle), descrB, mat_handle_ct{{[0-9]+}}, x, dpct::get_value(&beta, *handle), y);
//CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
//CHECK-NEXT: }
void foo2(cusparseMatDescr_t descrB){
  cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}