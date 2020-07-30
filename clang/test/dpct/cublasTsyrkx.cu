// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasTsyrkx.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>

//CHECK: #define macro_a (oneapi::mkl::transpose)1
#define macro_a (cublasOperation_t)1

//CHECK: #define macro_b (oneapi::mkl::uplo)1
#define macro_b (cublasFillMode_t)1

cublasFillMode_t foo(){
  return CUBLAS_FILL_MODE_LOWER;
}

cublasOperation_t bar(){
  return CUBLAS_OP_T;
}

int main() {
  int n = 275;
  int k = 275;
  int lda = 1;
  int ldb = 1;
  int ldc = 1;

  float alpha_s = 1;
  float beta_s = 1;

  double alpha_d = 1;
  double beta_d = 1;

  cublasHandle_t handle;
  cublasStatus_t status;

  float* A_s=0;
  float* B_s=0;
  float* C_s=0;

  double* A_d=0;
  double* B_d=0;
  double* C_d=0;

  int trans0 = 0;
  int trans1 = 1;
  int fill0 = 0;
  int fill1 = 1;


  //CHECK: {
  //CHECK-NEXT: auto A_s_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_s);
  //CHECK-NEXT: auto B_s_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_s);
  //CHECK-NEXT: auto C_s_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_s);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: status = (oneapi::mkl::blas::gemmt(*handle, fill0==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans0==0 ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_s, *handle), A_s_buf_ct{{[0-9]+}}, lda, B_s_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_s, *handle), C_s_buf_ct{{[0-9]+}}, ldc), 0);
  //CHECK-NEXT: }
  //CHECK-NEXT: {
  //CHECK-NEXT: auto A_s_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(A_s);
  //CHECK-NEXT: auto B_s_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(B_s);
  //CHECK-NEXT: auto C_s_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(C_s);
  //CHECK-NEXT: oneapi::mkl::blas::gemmt(*handle, fill1==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, trans1==0 ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_s, *handle), A_s_buf_ct{{[0-9]+}}, lda, B_s_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_s, *handle), C_s_buf_ct{{[0-9]+}}, ldc);
  //CHECK-NEXT: }
  status = cublasSsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_s, A_s, lda, B_s, ldb, &beta_s, C_s, ldc);
  cublasSsyrkx(handle, (cublasFillMode_t)fill1, (cublasOperation_t)trans1, n, k, &alpha_s, A_s, lda, B_s, ldb, &beta_s, C_s, ldc);

  //CHECK: {
  //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_d);
  //CHECK-NEXT: auto B_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_d);
  //CHECK-NEXT: auto C_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_d);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: status = (oneapi::mkl::blas::gemmt(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, n, k, dpct::get_value(&alpha_d, *handle), A_d_buf_ct{{[0-9]+}}, lda, B_d_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_d, *handle), C_d_buf_ct{{[0-9]+}}, ldc), 0);
  //CHECK-NEXT: }
  //CHECK-NEXT: {
  //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_d);
  //CHECK-NEXT: auto B_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_d);
  //CHECK-NEXT: auto C_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_d);
  //CHECK-NEXT: oneapi::mkl::blas::gemmt(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_d, *handle), A_d_buf_ct{{[0-9]+}}, lda, B_d_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_d, *handle), C_d_buf_ct{{[0-9]+}}, ldc);
  //CHECK-NEXT: }
  status = cublasDsyrkx(handle, (cublasFillMode_t)0, (cublasOperation_t)0, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);
  cublasDsyrkx(handle, (cublasFillMode_t)1, (cublasOperation_t)1, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);


  //CHECK: {
  //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_d);
  //CHECK-NEXT: auto B_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_d);
  //CHECK-NEXT: auto C_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_d);
  //CHECK-NEXT: oneapi::mkl::blas::gemmt(*handle, foo(), (int)macro_a==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)macro_a, (int)macro_a==0 ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_d, *handle), A_d_buf_ct{{[0-9]+}}, lda, B_d_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_d, *handle), C_d_buf_ct{{[0-9]+}}, ldc);
  //CHECK-NEXT: }


  cublasDsyrkx(handle, foo(), macro_a, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);


  //CHECK: {
  //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_d);
  //CHECK-NEXT: auto B_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_d);
  //CHECK-NEXT: auto C_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_d);
  //CHECK-NEXT: auto bar_transpose_ct{{[0-9]+}} = bar();
  //CHECK-NEXT: oneapi::mkl::blas::gemmt(*handle, (int)macro_b==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, bar_transpose_ct{{[0-9]+}}, bar_transpose_ct{{[0-9]+}}==oneapi::mkl::transpose::nontrans ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_d, *handle), A_d_buf_ct{{[0-9]+}}, lda, B_d_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_d, *handle), C_d_buf_ct{{[0-9]+}}, ldc);
  //CHECK-NEXT: }
  cublasDsyrkx(handle, macro_b, bar(), n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);


  //CHECK: {
  //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(A_d);
  //CHECK-NEXT: auto B_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(B_d);
  //CHECK-NEXT: auto C_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(C_d);
  //CHECK-NEXT: oneapi::mkl::blas::gemmt(*handle, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, n, k, dpct::get_value(&alpha_d, *handle), A_d_buf_ct{{[0-9]+}}, lda, B_d_buf_ct{{[0-9]+}}, ldb, dpct::get_value(&beta_d, *handle), C_d_buf_ct{{[0-9]+}}, ldc);
  //CHECK-NEXT: }
  cublasDsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);

  return 0;
}
