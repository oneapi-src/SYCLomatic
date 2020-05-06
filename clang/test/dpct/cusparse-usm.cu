// RUN: dpct --format-range=none --out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusparse-usm.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

int m, n, nnz;
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

int main(){
  //CHECK: mkl::index_base descr1 , descr2 ;
  //CHECK-NEXT: mkl::index_base descr3 ;
  cusparseMatDescr_t descr1 = 0, descr2 = 0;
  cusparseMatDescr_t descr3 = 0;

  //CHECK: handle = &dpct::get_default_queue();
  //CHECK-NEXT: descrA = mkl::index_base::zero;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseSetMatType was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: descrA = mkl::index_base::zero;
  //CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
  //CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrA, csrRowPtrA, csrColIndA, csrValA);
  //CHECK-NEXT: mkl::sparse::gemv(*handle, transA, alpha, mat_handle_ct{{[0-9]+}}, x, beta, y);
  //CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cusparseDestroyMatDescr was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: handle = nullptr;
  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
}

//CHECK: int foo(mkl::index_base descrB) try {
//CHECK-NEXT: mkl::sparse::matrix_handle_t mat_handle_ct{{[0-9]+}};
//CHECK-NEXT: mkl::sparse::init_matrix_handle(&mat_handle_ct{{[0-9]+}});
//CHECK-NEXT: mkl::sparse::set_csr_data(mat_handle_ct{{[0-9]+}}, m, n, descrB, csrRowPtrA, csrColIndA, csrValA);
//CHECK-NEXT: mkl::sparse::gemv(*handle, transA, alpha, mat_handle_ct{{[0-9]+}}, x, beta, y);
//CHECK-NEXT: mkl::sparse::release_matrix_handle(&mat_handle_ct{{[0-9]+}});
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. 0 is used in return statement. You need to rewrite this code.
//CHECK-NEXT: */
//CHECK-NEXT: return 0;
//CHECK-NEXT: }
int foo(cusparseMatDescr_t descrB){
  return cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}