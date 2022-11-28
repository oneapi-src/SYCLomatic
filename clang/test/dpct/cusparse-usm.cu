// UNSUPPORTED: v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
// UNSUPPORTED: cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
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
//CHECK-NEXT: std::shared_ptr<dpct::sparse::sparse_matrix_info> descrA;
cusparseHandle_t handle;
cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
cusparseMatDescr_t descrA;

int foo(int aaaaa){
  //CHECK: std::shared_ptr<dpct::sparse::sparse_matrix_info> descr1 = 0, descr2 = 0;
  //CHECK-NEXT: std::shared_ptr<dpct::sparse::sparse_matrix_info> descr3 = 0;
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
  //CHECK-NEXT: dpct::sparse::sparse_matrix_info::matrix_type type0 = dpct::sparse::sparse_matrix_info::matrix_type::ge;
  //CHECK-NEXT: descrA->set((oneapi::mkl::diag)aaaaa);
  //CHECK-NEXT: descrA->set((oneapi::mkl::uplo)aaaaa);
  //CHECK-NEXT: descrA->set((oneapi::mkl::index_base)aaaaa);
  //CHECK-NEXT: descrA->set((dpct::sparse::sparse_matrix_info::matrix_type)aaaaa);
  //CHECK-NEXT: diag0 = descrA->get<oneapi::mkl::diag>();
  //CHECK-NEXT: fill0 = descrA->get<oneapi::mkl::uplo>();
  //CHECK-NEXT: base0 = descrA->get<oneapi::mkl::index_base>();
  //CHECK-NEXT: type0 = descrA->get<dpct::sparse::sparse_matrix_info::matrix_type>();
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
  //CHECK-NEXT: descrA = std::make_shared<dpct::sparse::sparse_matrix_info>();
  //CHECK-NEXT: descrA->set((dpct::sparse::sparse_matrix_info::matrix_type)aaaaa);
  //CHECK-NEXT: descrA->set(oneapi::mkl::index_base::zero);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: oneapi::mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: oneapi::mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: oneapi::mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA->get<oneapi::mkl::index_base>(), const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
  //CHECK-NEXT: oneapi::mkl::sparse::gemv(*handle, dpct::get_transpose(aaaaa), alpha, mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), beta, y);
  //CHECK-NEXT: oneapi::mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, (cusparseMatrixType_t)aaaaa);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseDcsrmv(handle, (cusparseOperation_t)aaaaa, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);

  cuDoubleComplex alpha_Z, beta_Z, *csrValA_Z, *x_Z, *y_Z;

  //CHECK: oneapi::mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: oneapi::mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: oneapi::mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA->get<oneapi::mkl::index_base>(), const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), (std::complex<double>*)csrValA_Z);
  //CHECK-NEXT: oneapi::mkl::sparse::gemv(*handle, transA, std::complex<double>(alpha_Z.x(), alpha_Z.y()), mat_handle_ct{{[0-9]+}}, (std::complex<double>*)x_Z, std::complex<double>(beta_Z.x(), beta_Z.y()), (std::complex<double>*)y_Z);
  //CHECK-NEXT: oneapi::mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  cusparseZcsrmv(handle, transA, m, n, nnz, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, &beta_Z, y_Z);

  //CHECK: oneapi::mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: oneapi::mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: oneapi::mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, k, descrA->get<oneapi::mkl::index_base>(), const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
  //CHECK-NEXT: oneapi::mkl::sparse::gemm(*handle, transA, alpha, mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), n, ldb, beta, y, ldc);
  //CHECK-NEXT: oneapi::mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  cusparseDcsrmm(handle, transA, m, n, k, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);

  //CHECK: oneapi::mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: oneapi::mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: oneapi::mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, k, descrA->get<oneapi::mkl::index_base>(), const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), (std::complex<double>*)csrValA_Z);
  //CHECK-NEXT: oneapi::mkl::sparse::gemm(*handle, transA, std::complex<double>(alpha_Z.x(), alpha_Z.y()), mat_handle_ct{{[0-9]+}}, (std::complex<double>*)x_Z, n, ldb, std::complex<double>(beta_Z.x(), beta_Z.y()), (std::complex<double>*)y_Z, ldc);
  //CHECK-NEXT: oneapi::mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  cusparseZcsrmm(handle, transA, m, n, k, nnz, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, ldb, &beta_Z, y_Z, ldc);

  //CHECK:int status;
  cusparseStatus_t status;

  //CHECK: oneapi::mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: oneapi::mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: oneapi::mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA->get<oneapi::mkl::index_base>(), const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
  //CHECK-NEXT: oneapi::mkl::sparse::gemv(*handle, transA, alpha, mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), beta, y);
  //CHECK-NEXT: oneapi::mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: if(status = 0){}
  if(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  //CHECK: oneapi::mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: oneapi::mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: oneapi::mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA->get<oneapi::mkl::index_base>(), const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
  //CHECK-NEXT: oneapi::mkl::sparse::gemv(*handle, transA, alpha, mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), beta, y);
  //CHECK-NEXT: oneapi::mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a for statement. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: for(status = 0;;){}
  for(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);;){}

  //CHECK: oneapi::mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: oneapi::mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: oneapi::mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA->get<oneapi::mkl::index_base>(), const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
  //CHECK-NEXT: oneapi::mkl::sparse::gemv(*handle, transA, alpha, mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), beta, y);
  //CHECK-NEXT: oneapi::mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a switch statement. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: switch(status = 0){}
  switch(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  //CHECK: int info;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseCreateSolveAnalysisInfo was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDcsrsv_analysis was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDestroySolveAnalysisInfo was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  cusparseSolveAnalysisInfo_t info;
  cusparseCreateSolveAnalysisInfo(&info);
  cusparseDcsrsv_analysis(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info);
  cusparseDestroySolveAnalysisInfo(info);

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseZcsrsv_analysis was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  cusparseZcsrsv_analysis(handle, transA, m, nnz, descrA, csrValA_Z, csrRowPtrA, csrColIndA, info);

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDestroyMatDescr was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: handle = nullptr;
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
}

//CHECK: int foo(std::shared_ptr<dpct::sparse::sparse_matrix_info> descrB) try {
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1045:{{[0-9]+}}: Migration is only supported for this API for the general sparse matrix type. You may need to adjust the code.
//CHECK-NEXT: */
//CHECK-NEXT: oneapi::mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
//CHECK-NEXT: oneapi::mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
//CHECK-NEXT: oneapi::mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrB->get<oneapi::mkl::index_base>(), const_cast<int*>(csrRowPtrA), const_cast<int*>(csrColIndA), const_cast<double*>(csrValA));
//CHECK-NEXT: oneapi::mkl::sparse::gemv(*handle, transA, alpha, mat_handle_ct{{[0-9]+}}, const_cast<double*>(x), beta, y);
//CHECK-NEXT: oneapi::mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a return statement. You may need to rewrite this code.
//CHECK-NEXT: */
//CHECK-NEXT: return 0;
//CHECK-NEXT: }
int foo(cusparseMatDescr_t descrB){
  return cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}

