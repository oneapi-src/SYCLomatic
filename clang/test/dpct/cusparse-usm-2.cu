// RUN: dpct --format-range=none --out-root %T/cusparse-usm-2 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusparse-usm-2/cusparse-usm-2.dp.cpp --match-full-lines %s
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
  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, (cusparseMatrixType_t)aaaaa);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

  cuDoubleComplex alpha_Z, beta_Z, *csrValA_Z, *x_Z, *y_Z;

  //CHECK:int status;
  cusparseStatus_t status;

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDestroyMatDescr was removed because this call is redundant in SYCL.
  //CHECK-NEXT: */
  //CHECK-NEXT: handle = nullptr;
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
}


