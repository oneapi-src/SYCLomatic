// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cublasIsamax_etc.sycl.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
int main() {
  cublasStatus_t status;
  cublasHandle_t handle;
  int n = 275;
  int m = 275;
  int k = 275;
  int lda = 275;
  int ldb = 275;
  int ldc = 275;
  const float *A_S = 0;
  const float *B_S = 0;
  float *C_S = 0;
  float alpha_S = 1.0f;
  float beta_S = 0.0f;
  const double *A_D = 0;
  const double *B_D = 0;
  double *C_D = 0;
  double alpha_D = 1.0;
  double beta_D = 0.0;

  const float *x_S = 0;
  const double *x_D = 0;
  const float *y_S = 0;
  const double *y_D = 0;
  int incx = 1;
  int incy = 1;
  int *result = 0;
  float *result_S = 0;
  double *result_D = 0;
  //level1
  //cublasI<t>amax
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::isamax(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::isamax(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_BUFFER_{{[0-9,a-z]+}});
  status = cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::idamax(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::idamax(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_BUFFER_{{[0-9,a-z]+}});
  status = cublasIdamax(handle, n, x_D, incx, result);
  cublasIdamax(handle, n, x_D, incx, result);

  //cublasI<t>amin
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::isamin(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::isamin(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_BUFFER_{{[0-9,a-z]+}});
  status = cublasIsamin(handle, n, x_S, incx, result);
  cublasIsamin(handle, n, x_S, incx, result);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::idamin(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::idamin(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_BUFFER_{{[0-9,a-z]+}});
  status = cublasIdamin(handle, n, x_D, incx, result);
  cublasIdamin(handle, n, x_D, incx, result);

  //cublas<t>asum
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::sasum(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::sasum(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}});
  status = cublasSasum(handle, n, x_S, incx, result_S);
  cublasSasum(handle, n, x_S, incx, result_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dasum(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::dasum(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}});
  status = cublasDasum(handle, n, x_D, incx, result_D);
  cublasDasum(handle, n, x_D, incx, result_D);

  //cublas<t>axpy
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::saxpy(handle, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::saxpy(handle, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasSaxpy(handle, n, &alpha_S, x_S, incx, result_S, incy);
  cublasSaxpy(handle, n, &alpha_S, x_S, incx, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::daxpy(handle, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::daxpy(handle, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDaxpy(handle, n, &alpha_D, x_D, incx, result_D, incy);
  cublasDaxpy(handle, n, &alpha_D, x_D, incx, result_D, incy);

  //cublas<t>copy
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::scopy(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::scopy(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasScopy(handle, n, x_S, incx, result_S, incy);
  cublasScopy(handle, n, x_S, incx, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dcopy(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dcopy(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDcopy(handle, n, x_D, incx, result_D, incy);
  cublasDcopy(handle, n, x_D, incx, result_D, incy);

  //cublas<t>dot
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::sdot(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, y_S_BUFFER_{{[0-9,a-z]+}}, incy, result_S_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::sdot(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, y_S_BUFFER_{{[0-9,a-z]+}}, incy, result_S_BUFFER_{{[0-9,a-z]+}});
  status = cublasSdot(handle, n, x_S, incx, y_S, incy, result_S);
  cublasSdot(handle, n, x_S, incx, y_S, incy, result_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::ddot(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, y_D_BUFFER_{{[0-9,a-z]+}}, incy, result_D_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::ddot(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, y_D_BUFFER_{{[0-9,a-z]+}}, incy, result_D_BUFFER_{{[0-9,a-z]+}});
  status = cublasDdot(handle, n, x_D, incx, y_D, incy, result_D);
  cublasDdot(handle, n, x_D, incx, y_D, incy, result_D);

  //cublas<t>nrm2
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::snrm2(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::snrm2(handle, n, x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}});
  status = cublasSnrm2(handle, n, x_S, incx, result_S);
  cublasSnrm2(handle, n, x_S, incx, result_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dnrm2(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::dnrm2(handle, n, x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}});
  status = cublasDnrm2(handle, n, x_D, incx, result_D);
  cublasDnrm2(handle, n, x_D, incx, result_D);

  float *x_f = 0;
  float *y_f = 0;
  double *x_d = 0;
  double *y_d = 0;
  //cublas<t>rot
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::srot(handle, n, x_f_BUFFER_{{[0-9,a-z]+}}, incx, y_f_BUFFER_{{[0-9,a-z]+}}, incy, *(x_S), *(y_S)), 0);
  // CHECK: mkl::srot(handle, n, x_f_BUFFER_{{[0-9,a-z]+}}, incx, y_f_BUFFER_{{[0-9,a-z]+}}, incy, *(x_S), *(y_S));
  status = cublasSrot(handle, n, x_f, incx, y_f, incy, x_S, y_S);
  cublasSrot(handle, n, x_f, incx, y_f, incy, x_S, y_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::drot(handle, n, x_d_BUFFER_{{[0-9,a-z]+}}, incx, y_d_BUFFER_{{[0-9,a-z]+}}, incy, *(x_D), *(y_D)), 0);
  // CHECK: mkl::drot(handle, n, x_d_BUFFER_{{[0-9,a-z]+}}, incx, y_d_BUFFER_{{[0-9,a-z]+}}, incy, *(x_D), *(y_D));
  status = cublasDrot(handle, n, x_d, incx, y_d, incy, x_D, y_D);
  cublasDrot(handle, n, x_d, incx, y_d, incy, x_D, y_D);

  //cublas<t>rotg
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::srotg(handle, x_f_BUFFER_{{[0-9,a-z]+}}, y_f_BUFFER_{{[0-9,a-z]+}}, x_f_BUFFER_{{[0-9,a-z]+}}, y_f_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::srotg(handle, x_f_BUFFER_{{[0-9,a-z]+}}, y_f_BUFFER_{{[0-9,a-z]+}}, x_f_BUFFER_{{[0-9,a-z]+}}, y_f_BUFFER_{{[0-9,a-z]+}});
  status = cublasSrotg(handle, x_f, y_f, x_f, y_f);
  cublasSrotg(handle, x_f, y_f, x_f, y_f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::drotg(handle, x_d_BUFFER_{{[0-9,a-z]+}}, y_d_BUFFER_{{[0-9,a-z]+}}, x_d_BUFFER_{{[0-9,a-z]+}}, y_d_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::drotg(handle, x_d_BUFFER_{{[0-9,a-z]+}}, y_d_BUFFER_{{[0-9,a-z]+}}, x_d_BUFFER_{{[0-9,a-z]+}}, y_d_BUFFER_{{[0-9,a-z]+}});
  status = cublasDrotg(handle, x_d, y_d, x_d, y_d);
  cublasDrotg(handle, x_d, y_d, x_d, y_d);

  //cublas<t>rotm
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::srotm(handle, n, x_f_BUFFER_{{[0-9,a-z]+}}, incx, y_f_BUFFER_{{[0-9,a-z]+}}, incy, x_S_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::srotm(handle, n, x_f_BUFFER_{{[0-9,a-z]+}}, incx, y_f_BUFFER_{{[0-9,a-z]+}}, incy, x_S_BUFFER_{{[0-9,a-z]+}});
  status = cublasSrotm(handle, n, x_f, incx, y_f, incy, x_S);
  cublasSrotm(handle, n, x_f, incx, y_f, incy, x_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::drotm(handle, n, x_d_BUFFER_{{[0-9,a-z]+}}, incx, y_d_BUFFER_{{[0-9,a-z]+}}, incy, x_D_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::drotm(handle, n, x_d_BUFFER_{{[0-9,a-z]+}}, incx, y_d_BUFFER_{{[0-9,a-z]+}}, incy, x_D_BUFFER_{{[0-9,a-z]+}});
  status = cublasDrotm(handle, n, x_d, incx, y_d, incy, x_D);
  cublasDrotm(handle, n, x_d, incx, y_d, incy, x_D);

  //cublas<t>rotmg
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::srotmg(handle, x_f_BUFFER_{{[0-9,a-z]+}}, y_f_BUFFER_{{[0-9,a-z]+}}, y_f_BUFFER_{{[0-9,a-z]+}}, x_S, y_f_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::srotmg(handle, x_f_BUFFER_{{[0-9,a-z]+}}, y_f_BUFFER_{{[0-9,a-z]+}}, y_f_BUFFER_{{[0-9,a-z]+}}, x_S, y_f_BUFFER_{{[0-9,a-z]+}});
  status = cublasSrotmg(handle, x_f, y_f, y_f, x_S, y_f);
  cublasSrotmg(handle, x_f, y_f, y_f, x_S, y_f);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::drotmg(handle, x_d_BUFFER_{{[0-9,a-z]+}}, y_d_BUFFER_{{[0-9,a-z]+}}, y_d_BUFFER_{{[0-9,a-z]+}}, x_D, y_d_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::drotmg(handle, x_d_BUFFER_{{[0-9,a-z]+}}, y_d_BUFFER_{{[0-9,a-z]+}}, y_d_BUFFER_{{[0-9,a-z]+}}, x_D, y_d_BUFFER_{{[0-9,a-z]+}});
  status = cublasDrotmg(handle, x_d, y_d, y_d, x_D, y_d);
  cublasDrotmg(handle, x_d, y_d, y_d, x_D, y_d);

  //cublas<t>scal
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::sscal(handle, n, *(&alpha_S), x_f_BUFFER_{{[0-9,a-z]+}}, incx), 0);
  // CHECK: mkl::sscal(handle, n, *(&alpha_S), x_f_BUFFER_{{[0-9,a-z]+}}, incx);
  status = cublasSscal(handle, n, &alpha_S, x_f, incx);
  cublasSscal(handle, n, &alpha_S, x_f, incx);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dscal(handle, n, *(&alpha_D), x_d_BUFFER_{{[0-9,a-z]+}}, incx), 0);
  // CHECK: mkl::dscal(handle, n, *(&alpha_D), x_d_BUFFER_{{[0-9,a-z]+}}, incx);
  status = cublasDscal(handle, n, &alpha_D, x_d, incx);
  cublasDscal(handle, n, &alpha_D, x_d, incx);

  //cublas<t>swap
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::sswap(handle, n, x_f_BUFFER_{{[0-9,a-z]+}}, incx, y_f_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::sswap(handle, n, x_f_BUFFER_{{[0-9,a-z]+}}, incx, y_f_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasSswap(handle, n, x_f, incx, y_f, incy);
  cublasSswap(handle, n, x_f, incx, y_f, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dswap(handle, n, x_d_BUFFER_{{[0-9,a-z]+}}, incx, y_d_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dswap(handle, n, x_d_BUFFER_{{[0-9,a-z]+}}, incx, y_d_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDswap(handle, n, x_d, incx, y_d, incy);
  cublasDswap(handle, n, x_d, incx, y_d, incy);

  //level2
  //cublas<t>gbmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::sgbmv(handle, mkl::transpose::nontrans, m, n, m, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, lda, y_S_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_S), result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::sgbmv(handle, mkl::transpose::nontrans, m, n, m, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, lda, y_S_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_S), result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasSgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dgbmv(handle, mkl::transpose::nontrans, m, n, m, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, lda, y_D_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_D), result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dgbmv(handle, mkl::transpose::nontrans, m, n, m, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, lda, y_D_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_D), result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>gemv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::sgemv(handle, mkl::transpose::nontrans, m, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, lda, y_S_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_S), result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::sgemv(handle, mkl::transpose::nontrans, m, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, lda, y_S_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_S), result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dgemv(handle, mkl::transpose::nontrans, m, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, lda, y_D_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_D), result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dgemv(handle, mkl::transpose::nontrans, m, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, lda, y_D_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_D), result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>ger
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::sger(handle, m, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, y_S_BUFFER_{{[0-9,a-z]+}}, incy, result_S_BUFFER_{{[0-9,a-z]+}}, lda), 0);
  // CHECK: mkl::sger(handle, m, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, y_S_BUFFER_{{[0-9,a-z]+}}, incy, result_S_BUFFER_{{[0-9,a-z]+}}, lda);
  status = cublasSger(handle, m, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasSger(handle, m, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dger(handle, m, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, y_D_BUFFER_{{[0-9,a-z]+}}, incy, result_D_BUFFER_{{[0-9,a-z]+}}, lda), 0);
  // CHECK: mkl::dger(handle, m, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, y_D_BUFFER_{{[0-9,a-z]+}}, incy, result_D_BUFFER_{{[0-9,a-z]+}}, lda);
  status = cublasDger(handle, m, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  cublasDger(handle, m, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>sbmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::ssbmv(handle, mkl::uplo::upper, m, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, lda, y_S_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_S), result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::ssbmv(handle, mkl::uplo::upper, m, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, lda, y_S_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_S), result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasSsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dsbmv(handle, mkl::uplo::upper, m, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, lda, y_D_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_D), result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dsbmv(handle, mkl::uplo::upper, m, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, lda, y_D_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_D), result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>spmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::sspmv(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, y_S_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_S), result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::sspmv(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, y_S_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_S), result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasSspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, y_S, incx, &beta_S, result_S, incy);
  cublasSspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, y_S, incx, &beta_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dspmv(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, y_D_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_D), result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dspmv(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, y_D_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_D), result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, y_D, incx, &beta_D, result_D, incy);
  cublasDspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>spr
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::sspr(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::sspr(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}});
  status = cublasSspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S);
  cublasSspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dspr(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::dspr(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}});
  status = cublasDspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D);
  cublasDspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D);

  //cublas<t>spr2
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::sspr2(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, y_S_BUFFER_{{[0-9,a-z]+}}, incy, result_S_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::sspr2(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, y_S_BUFFER_{{[0-9,a-z]+}}, incy, result_S_BUFFER_{{[0-9,a-z]+}});
  status = cublasSspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S);
  cublasSspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dspr2(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, y_D_BUFFER_{{[0-9,a-z]+}}, incy, result_D_BUFFER_{{[0-9,a-z]+}}), 0);
  // CHECK: mkl::dspr2(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, y_D_BUFFER_{{[0-9,a-z]+}}, incy, result_D_BUFFER_{{[0-9,a-z]+}});
  status = cublasDspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D);
  cublasDspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D);

  //cublas<t>symv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::ssymv(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, lda, y_S_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_S), result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::ssymv(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, lda, y_S_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_S), result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dsymv(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, lda, y_D_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_D), result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dsymv(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, lda, y_D_BUFFER_{{[0-9,a-z]+}}, incx, *(&beta_D), result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>syr
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::ssyr(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}}, lda), 0);
  // CHECK: mkl::ssyr(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, result_S_BUFFER_{{[0-9,a-z]+}}, lda);
  status = cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S, lda);
  cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dsyr(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}}, lda), 0);
  // CHECK: mkl::dsyr(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, result_D_BUFFER_{{[0-9,a-z]+}}, lda);
  status = cublasDsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D, lda);
  cublasDsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D, lda);

  //cublas<t>syr2
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::ssyr2(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, y_S_BUFFER_{{[0-9,a-z]+}}, incy, result_S_BUFFER_{{[0-9,a-z]+}}, lda), 0);
  // CHECK: mkl::ssyr2(handle, mkl::uplo::upper, n, *(&alpha_S), x_S_BUFFER_{{[0-9,a-z]+}}, incx, y_S_BUFFER_{{[0-9,a-z]+}}, incy, result_S_BUFFER_{{[0-9,a-z]+}}, lda);
  status = cublasSsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasSsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dsyr2(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, y_D_BUFFER_{{[0-9,a-z]+}}, incy, result_D_BUFFER_{{[0-9,a-z]+}}, lda), 0);
  // CHECK: mkl::dsyr2(handle, mkl::uplo::upper, n, *(&alpha_D), x_D_BUFFER_{{[0-9,a-z]+}}, incx, y_D_BUFFER_{{[0-9,a-z]+}}, incy, result_D_BUFFER_{{[0-9,a-z]+}}, lda);
  status = cublasDsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  cublasDsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>tbmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::stbmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_BUFFER_{{[0-9,a-z]+}}, lda, result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::stbmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_BUFFER_{{[0-9,a-z]+}}, lda, result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasStbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);
  cublasStbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dtbmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_BUFFER_{{[0-9,a-z]+}}, lda, result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dtbmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_BUFFER_{{[0-9,a-z]+}}, lda, result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDtbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);
  cublasDtbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);

  //cublas<t>tbsv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::stbsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_BUFFER_{{[0-9,a-z]+}}, lda, result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::stbsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_S_BUFFER_{{[0-9,a-z]+}}, lda, result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasStbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);
  cublasStbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dtbsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_BUFFER_{{[0-9,a-z]+}}, lda, result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dtbsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, n, x_D_BUFFER_{{[0-9,a-z]+}}, lda, result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDtbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);
  cublasDtbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);

  //cublas<t>tpmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::stpmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_BUFFER_{{[0-9,a-z]+}}, result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::stpmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_BUFFER_{{[0-9,a-z]+}}, result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasStpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);
  cublasStpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dtpmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_BUFFER_{{[0-9,a-z]+}}, result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dtpmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_BUFFER_{{[0-9,a-z]+}}, result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDtpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);
  cublasDtpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);

  //cublas<t>tpsv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::stpsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_BUFFER_{{[0-9,a-z]+}}, result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::stpsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_BUFFER_{{[0-9,a-z]+}}, result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasStpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);
  cublasStpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dtpsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_BUFFER_{{[0-9,a-z]+}}, result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dtpsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_BUFFER_{{[0-9,a-z]+}}, result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDtpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);
  cublasDtpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);

  //cublas<t>trmv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::strmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_BUFFER_{{[0-9,a-z]+}}, lda, result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::strmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_BUFFER_{{[0-9,a-z]+}}, lda, result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);
  cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dtrmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_BUFFER_{{[0-9,a-z]+}}, lda, result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dtrmv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_BUFFER_{{[0-9,a-z]+}}, lda, result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDtrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);
  cublasDtrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);

  //cublas<t>trsv
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::strsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_BUFFER_{{[0-9,a-z]+}}, lda, result_S_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::strsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_S_BUFFER_{{[0-9,a-z]+}}, lda, result_S_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasStrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);
  cublasStrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dtrsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_BUFFER_{{[0-9,a-z]+}}, lda, result_D_BUFFER_{{[0-9,a-z]+}}, incy), 0);
  // CHECK: mkl::dtrsv(handle, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, n, x_D_BUFFER_{{[0-9,a-z]+}}, lda, result_D_BUFFER_{{[0-9,a-z]+}}, incy);
  status = cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);
  cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);

  //level3

  // cublas<T>symm
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::ssymm(handle, mkl::side::left, mkl::uplo::upper,  m, n, *(&alpha_S), A_S_BUFFER_{{[0-9,a-z]+}}, lda, B_S_BUFFER_{{[0-9,a-z]+}},  ldb, *(&beta_S), C_S_BUFFER_{{[0-9,a-z]+}}, ldc), 0);
  // CHECK: mkl::ssymm(handle, mkl::side::right, mkl::uplo::lower,  m, n, *(&alpha_S), A_S_BUFFER_{{[0-9,a-z]+}}, lda, B_S_BUFFER_{{[0-9,a-z]+}}, ldb, *(&beta_S), C_S_BUFFER_{{[0-9,a-z]+}}, ldc);
  status = cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);
  cublasSsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dsymm(handle, mkl::side::left, mkl::uplo::upper,  m, n, *(&alpha_D), A_D_BUFFER_{{[0-9,a-z]+}}, lda, B_D_BUFFER_{{[0-9,a-z]+}}, ldb, *(&beta_D), C_D_BUFFER_{{[0-9,a-z]+}}, ldc), 0);
  // CHECK: mkl::dsymm(handle, mkl::side::right, mkl::uplo::lower,  m, n, *(&alpha_D), A_D_BUFFER_{{[0-9,a-z]+}}, lda, B_D_BUFFER_{{[0-9,a-z]+}}, ldb, *(&beta_D), C_D_BUFFER_{{[0-9,a-z]+}}, ldc);
  status = cublasDsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);
  cublasDsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);

  // cublas<T>syrk
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::ssyrk(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_S), A_S_BUFFER_{{[0-9,a-z]+}}, lda, *(&beta_S), C_S_BUFFER_{{[0-9,a-z]+}}, ldc), 0);
  // CHECK: mkl::ssyrk(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_S), A_S_BUFFER_{{[0-9,a-z]+}}, lda, *(&beta_S), C_S_BUFFER_{{[0-9,a-z]+}}, ldc);
  status = cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, &beta_S, C_S, ldc);
  cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, &beta_S, C_S, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dsyrk(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_D), A_D_BUFFER_{{[0-9,a-z]+}}, lda, *(&beta_D), C_D_BUFFER_{{[0-9,a-z]+}}, ldc), 0);
  // CHECK: mkl::dsyrk(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_D), A_D_BUFFER_{{[0-9,a-z]+}}, lda, *(&beta_D), C_D_BUFFER_{{[0-9,a-z]+}}, ldc);
  status = cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, &beta_D, C_D, ldc);
  cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, &beta_D, C_D, ldc);

  // cublas<T>syr2k
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::ssyr2k(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_S), A_S_BUFFER_{{[0-9,a-z]+}}, lda, B_S_BUFFER_{{[0-9,a-z]+}}, ldb, *(&beta_S), C_S_BUFFER_{{[0-9,a-z]+}}, ldc), 0);
  // CHECK: mkl::ssyr2k(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_S), A_S_BUFFER_{{[0-9,a-z]+}}, lda, B_S_BUFFER_{{[0-9,a-z]+}}, ldb, *(&beta_S), C_S_BUFFER_{{[0-9,a-z]+}}, ldc);
  status = cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);
  cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dsyr2k(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_D), A_D_BUFFER_{{[0-9,a-z]+}}, lda, B_D_BUFFER_{{[0-9,a-z]+}}, ldb, *(&beta_D), C_D_BUFFER_{{[0-9,a-z]+}}, ldc), 0);
  // CHECK: mkl::dsyr2k(handle, mkl::uplo::upper, mkl::transpose::nontrans, n, k, *(&alpha_D), A_D_BUFFER_{{[0-9,a-z]+}}, lda, B_D_BUFFER_{{[0-9,a-z]+}}, ldb, *(&beta_D), C_D_BUFFER_{{[0-9,a-z]+}}, ldc);
  status = cublasDsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);
  cublasDsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);

  // cublas<T>trsm
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::strsm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, m, n, *(&alpha_S), A_S_BUFFER_{{[0-9,a-z]+}}, lda, C_S_BUFFER_{{[0-9,a-z]+}}, ldc), 0);
  // CHECK: mkl::strsm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, *(&alpha_S), A_S_BUFFER_{{[0-9,a-z]+}}, lda, C_S_BUFFER_{{[0-9,a-z]+}}, ldc);
  status = cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, n, &alpha_S, A_S, lda, C_S, ldc);
  cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_S, A_S, lda, C_S, ldc);

  // CHECK: /*
  // CHECK-NEXT: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  // CHECK-NEXT: */
  // CHECK: status = (mkl::dtrsm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::unit, m, n, *(&alpha_D), A_D_BUFFER_{{[0-9,a-z]+}}, lda, C_D_BUFFER_{{[0-9,a-z]+}}, ldc), 0);
  // CHECK: mkl::dtrsm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m, n, *(&alpha_D), A_D_BUFFER_{{[0-9,a-z]+}}, lda, C_D_BUFFER_{{[0-9,a-z]+}}, ldc);
  status = cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, n, &alpha_D, A_D, lda, C_D, ldc);
  cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_D, A_D, lda, C_D, ldc);

  return 0;
}
