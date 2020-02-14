// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasTtrmmLegacy.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>

int main(){
  cublasStatus_t status;
  cublasHandle_t handle;
  int n = 275;
  int m = 275;
  int lda = 275;
  int ldb = 275;
  const float *A_S = 0;
  float *B_S = 0;
  float alpha_S = 1.0f;
  const double *A_D = 0;
  double *B_D = 0;
  double alpha_D = 1.0;
  const cuComplex *A_C = 0;
  cuComplex *B_C = 0;
  cuComplex alpha_C = make_cuComplex(1.0f,0.0f);
  const cuDoubleComplex *A_Z = 0;
  cuDoubleComplex *B_Z = 0;
  cuDoubleComplex alpha_Z = make_cuDoubleComplex(1.0,0.0);


  //Legacy
  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'L';
  // CHECK-NEXT: auto fillmode_ct1 = 'U';
  // CHECK-NEXT: auto transpose_ct2 = 'N';
  // CHECK-NEXT: auto diagtype_ct3 = 'N';
  // CHECK-NEXT: auto A_S_buff_ct1 = dpct::get_buffer<float>(A_S);
  // CHECK-NEXT: auto B_S_buff_ct1 = dpct::get_buffer<float>(B_S);
  // CHECK-NEXT:mkl::blas::trmm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, alpha_S, A_S_buff_ct1, lda, B_S_buff_ct1, ldb);
  // CHECK-NEXT:}
  cublasStrmm('L', 'U', 'N', 'N', m, n, alpha_S, A_S, lda, B_S, ldb);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'L';
  // CHECK-NEXT: auto fillmode_ct1 = 'U';
  // CHECK-NEXT: auto transpose_ct2 = 'N';
  // CHECK-NEXT: auto diagtype_ct3 = 'N';
  // CHECK-NEXT: auto A_D_buff_ct1 = dpct::get_buffer<double>(A_D);
  // CHECK-NEXT: auto B_D_buff_ct1 = dpct::get_buffer<double>(B_D);
  // CHECK-NEXT:mkl::blas::trmm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, alpha_D, A_D_buff_ct1, lda, B_D_buff_ct1, ldb);
  // CHECK-NEXT:}
  cublasDtrmm('L', 'U', 'N', 'N', m, n, alpha_D, A_D, lda, B_D, ldb);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'L';
  // CHECK-NEXT: auto fillmode_ct1 = 'U';
  // CHECK-NEXT: auto transpose_ct2 = 'N';
  // CHECK-NEXT: auto diagtype_ct3 = 'N';
  // CHECK-NEXT: auto A_C_buff_ct1 = dpct::get_buffer<std::complex<float>>(A_C);
  // CHECK-NEXT: auto B_C_buff_ct1 = dpct::get_buffer<std::complex<float>>(B_C);
  // CHECK-NEXT:mkl::blas::trmm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_buff_ct1, lda, B_C_buff_ct1, ldb);
  // CHECK-NEXT:}
  cublasCtrmm('L', 'U', 'N', 'N', m, n, alpha_C, A_C, lda, B_C, ldb);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct0 = 'L';
  // CHECK-NEXT: auto fillmode_ct1 = 'U';
  // CHECK-NEXT: auto transpose_ct2 = 'N';
  // CHECK-NEXT: auto diagtype_ct3 = 'N';
  // CHECK-NEXT: auto A_Z_buff_ct1 = dpct::get_buffer<std::complex<double>>(A_Z);
  // CHECK-NEXT: auto B_Z_buff_ct1 = dpct::get_buffer<std::complex<double>>(B_Z);
  // CHECK-NEXT:mkl::blas::trmm(dpct::get_default_queue(), (((sidemode_ct0)=='L'||(sidemode_ct0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct1)=='L'||(fillmode_ct1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct2)=='N'||(transpose_ct2)=='n')?(mkl::transpose::nontrans):(((transpose_ct2)=='T'||(transpose_ct2)=='t')?(mkl::transpose::trans):(mkl::transpose::conjtrans))), (((diagtype_ct3)=='N'||(diagtype_ct3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_buff_ct1, lda, B_Z_buff_ct1, ldb);
  // CHECK-NEXT:}
  cublasZtrmm('L', 'U', 'N', 'N', m, n, alpha_Z, A_Z, lda, B_Z, ldb);
}
