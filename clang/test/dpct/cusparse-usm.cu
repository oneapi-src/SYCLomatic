// RUN: dpct --format-range=none --out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusparse-usm.dp.cpp --match-full-lines %s
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
//CHECK-NEXT: mkl::transpose transA = mkl::transpose::nontrans;
//CHECK-NEXT: mkl::index_base descrA;
cusparseHandle_t handle;
cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
cusparseMatDescr_t descrA;

int foo(int aaaaa){
  //CHECK: mkl::index_base descr1 , descr2 ;
  //CHECK-NEXT: mkl::index_base descr3 ;
  cusparseMatDescr_t descr1 = 0, descr2 = 0;
  cusparseMatDescr_t descr3 = 0;

  //CHECK: int mode = 1;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseGetPointerMode was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseSetPointerMode was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  cusparsePointerMode_t mode = CUSPARSE_POINTER_MODE_DEVICE;
  cusparseGetPointerMode(handle, &mode);
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

  //CHECK: mkl::diag diag0 = mkl::diag::nonunit;
  //CHECK-NEXT: mkl::uplo fill0 = mkl::uplo::lower;
  //CHECK-NEXT: mkl::index_base base0 = mkl::index_base::zero;
  //CHECK-NEXT: int type0 = 0;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseSetMatDiagType was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseSetMatFillMode was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: descrA = (mkl::index_base)aaaaa;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseSetMatType was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusparseGetMatDiagType was replaced with 0, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: diag0 = (mkl::diag)0;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusparseGetMatFillMode was replaced with 0, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: fill0 = (mkl::uplo)0;
  //CHECK-NEXT: base0 = descrA;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusparseGetMatType was replaced with 0, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: type0 = 0;
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

  //CHECK: handle = &dpct::get_default_queue();
  //CHECK-NEXT: descrA = mkl::index_base::zero;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseSetMatType was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: descrA = mkl::index_base::zero;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The Intel(R) DPC++ Compatibility Tool supports migration of only general sparse matrix type for this API currently. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA, const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
  //CHECK-NEXT: mkl::sparse::gemv(*handle, dpct::get_transpose(aaaaa), dpct::get_value(&alpha, *handle), mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), dpct::get_value(&beta, *handle), y);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, (cusparseMatrixType_t)aaaaa);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseDcsrmv(handle, (cusparseOperation_t)aaaaa, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);

  cuDoubleComplex alpha_Z, beta_Z, *csrValA_Z, *x_Z, *y_Z;

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The Intel(R) DPC++ Compatibility Tool supports migration of only general sparse matrix type for this API currently. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA, const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), (std::complex<double>*)csrValA_Z);
  //CHECK-NEXT: mkl::sparse::gemv(*handle, transA, dpct::get_value(&alpha_Z, *handle), mat_handle_ct{{[0-9]+}}, (std::complex<double>*)x_Z, dpct::get_value(&beta_Z, *handle), (std::complex<double>*)y_Z);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  cusparseZcsrmv(handle, transA, m, n, nnz, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, &beta_Z, y_Z);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The Intel(R) DPC++ Compatibility Tool supports migration of only general sparse matrix type for this API currently. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, k, descrA, const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
  //CHECK-NEXT: mkl::sparse::gemm(*handle, transA, dpct::get_value(&alpha, *handle), mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), n, ldb, dpct::get_value(&beta, *handle), y, ldc);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  cusparseDcsrmm(handle, transA, m, n, k, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The Intel(R) DPC++ Compatibility Tool supports migration of only general sparse matrix type for this API currently. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, k, descrA, const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), (std::complex<double>*)csrValA_Z);
  //CHECK-NEXT: mkl::sparse::gemm(*handle, transA, dpct::get_value(&alpha_Z, *handle), mat_handle_ct{{[0-9]+}}, (std::complex<double>*)x_Z, n, ldb, dpct::get_value(&beta_Z, *handle), (std::complex<double>*)y_Z, ldc);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  cusparseZcsrmm(handle, transA, m, n, k, nnz, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, ldb, &beta_Z, y_Z, ldc);

  //CHECK:int status;
  cusparseStatus_t status;

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The Intel(R) DPC++ Compatibility Tool supports migration of only general sparse matrix type for this API currently. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA, const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
  //CHECK-NEXT: mkl::sparse::gemv(*handle, transA, dpct::get_value(&alpha, *handle), mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), dpct::get_value(&beta, *handle), y);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: if(status = 0){}
  if(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The Intel(R) DPC++ Compatibility Tool supports migration of only general sparse matrix type for this API currently. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA, const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
  //CHECK-NEXT: mkl::sparse::gemv(*handle, transA, dpct::get_value(&alpha, *handle), mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), dpct::get_value(&beta, *handle), y);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a for statement. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: for(status = 0;;){}
  for(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);;){}

  //CHECK: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: The Intel(R) DPC++ Compatibility Tool supports migration of only general sparse matrix type for this API currently. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA, const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
  //CHECK-NEXT: mkl::sparse::gemv(*handle, transA, dpct::get_value(&alpha, *handle), mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), dpct::get_value(&beta, *handle), y);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a switch statement. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: switch(status = 0){}
  switch(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  //CHECK: int info;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseCreateSolveAnalysisInfo was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDcsrsv_analysis was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDestroySolveAnalysisInfo was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  cusparseSolveAnalysisInfo_t info;
  cusparseCreateSolveAnalysisInfo(&info);
  cusparseDcsrsv_analysis(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info);
  cusparseDestroySolveAnalysisInfo(info);

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseZcsrsv_analysis was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  cusparseZcsrsv_analysis(handle, transA, m, nnz, descrA, csrValA_Z, csrRowPtrA, csrColIndA, info);

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDestroyMatDescr was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: handle = nullptr;
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
}

//CHECK: int foo(mkl::index_base descrB) try {
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1045:{{[0-9]+}}: The Intel(R) DPC++ Compatibility Tool supports migration of only general sparse matrix type for this API currently. You may need to adjust the code.
//CHECK-NEXT: */
//CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
//CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
//CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrB, const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
//CHECK-NEXT: mkl::sparse::gemv(*handle, transA, dpct::get_value(&alpha, *handle), mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), dpct::get_value(&beta, *handle), y);
//CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a return statement. You may need to rewrite this code.
//CHECK-NEXT: */
//CHECK-NEXT: return 0;
//CHECK-NEXT: }
int foo(cusparseMatDescr_t descrB){
  return cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}