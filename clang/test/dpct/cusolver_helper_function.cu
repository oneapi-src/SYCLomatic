// RUN: dpct --format-range=none -out-root %T/cusolver_helper_function %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolver_helper_function/cusolver_helper_function.dp.cpp --match-full-lines %s

//CHECK:#include <sycl/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include <oneapi/mkl.hpp>
//CHECK-NEXT:#include <dpct/lapack_utils.hpp>
#include "cusolverDn.h"

int foo1() {
  cusolverDnHandle_t handle;
  float *a_s, *b_s, *w_s, *work_s;
  double *a_d, *b_d, *w_d, *work_d;
  int lwork_s, lwork_d;
  int *devInfo;
  //CHECK:status = (lwork_s = oneapi::mkl::lapack::sygvd_scratchpad_size<float>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3), 0);
  cusolverStatus_t status;
  status = cusolverDnSsygvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s, 3, b_s, 3, w_s, &lwork_s);
  //CHECK:status = dpct::lapack::sygvd(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_s, 3, b_s, 3, w_s, work_s, lwork_s, devInfo);
  status = cusolverDnSsygvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s, 3, b_s, 3, w_s, work_s, lwork_s, devInfo);

  //CHECK:status = (lwork_d = oneapi::mkl::lapack::sygvd_scratchpad_size<double>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3), 0);
  status = cusolverDnDsygvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d, 3, b_d, 3, w_d, &lwork_d);
  //CHECK:status = dpct::lapack::sygvd(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_d, 3, b_d, 3, w_d, work_d, lwork_d, devInfo);
  status = cusolverDnDsygvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d, 3, b_d, 3, w_d, work_d, lwork_d, devInfo);

  return 0;
}

int foo2() {
  cusolverDnHandle_t handle;
  float2 *a_c, *b_c, *work_c;
  double2 *a_z, *b_z, *work_z;
  float *w_c;
  double *w_z;
  int lwork_c, lwork_z;
  int *devInfo;
  cusolverStatus_t status;

  //CHECK:status = (lwork_c = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<float>>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3), 0);
  status = cusolverDnChegvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_c, 3, b_c, 3, w_c, &lwork_c);
  //CHECK:status = dpct::lapack::hegvd(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_c, 3, b_c, 3, w_c, work_c, lwork_c, devInfo);
  status = cusolverDnChegvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_c, 3, b_c, 3, w_c, work_c, lwork_c, devInfo);

  //CHECK:status = (lwork_z = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<double>>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3), 0);
  status = cusolverDnZhegvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_z, 3, b_z, 3, w_z, &lwork_z);
  //CHECK:status = dpct::lapack::hegvd(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_z, 3, b_z, 3, w_z, work_z, lwork_z, devInfo);
  status = cusolverDnZhegvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_z, 3, b_z, 3, w_z, work_z, lwork_z, devInfo);

  return 0;
}

int foo3() {
  cusolverDnHandle_t handle;
  float ** a_s_ptrs;
  double ** a_d_ptrs;
  float2 ** a_c_ptrs;
  double2 ** a_z_ptrs;
  int *infoArray;
  cusolverStatus_t status;

  //CHECK:status = dpct::lapack::potrf_batch(*handle, oneapi::mkl::uplo::upper, 3, a_s_ptrs, 3, infoArray, 2);
  //CHECK:status = dpct::lapack::potrf_batch(*handle, oneapi::mkl::uplo::upper, 3, a_d_ptrs, 3, infoArray, 2);
  //CHECK:status = dpct::lapack::potrf_batch(*handle, oneapi::mkl::uplo::upper, 3, a_c_ptrs, 3, infoArray, 2);
  //CHECK:status = dpct::lapack::potrf_batch(*handle, oneapi::mkl::uplo::upper, 3, a_z_ptrs, 3, infoArray, 2);
  status = cusolverDnSpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, a_s_ptrs, 3, infoArray, 2);
  status = cusolverDnDpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, a_d_ptrs, 3, infoArray, 2);
  status = cusolverDnCpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, a_c_ptrs, 3, infoArray, 2);
  status = cusolverDnZpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, a_z_ptrs, 3, infoArray, 2);

  return 0;
}

int foo4() {
  cusolverDnHandle_t handle;
  float ** a_s_ptrs, ** b_s_ptrs;
  double ** a_d_ptrs, ** b_d_ptrs;
  float2 ** a_c_ptrs, ** b_c_ptrs;
  double2 ** a_z_ptrs, ** b_z_ptrs;
  int *infoArray;
  cusolverStatus_t status;

  //CHECK:status = dpct::lapack::potrs_batch(*handle, oneapi::mkl::uplo::upper, 3, 1, a_s_ptrs, 3, b_s_ptrs, 3, infoArray, 2);
  //CHECK:status = dpct::lapack::potrs_batch(*handle, oneapi::mkl::uplo::upper, 3, 1, a_d_ptrs, 3, b_d_ptrs, 3, infoArray, 2);
  //CHECK:status = dpct::lapack::potrs_batch(*handle, oneapi::mkl::uplo::upper, 3, 1, a_c_ptrs, 3, b_c_ptrs, 3, infoArray, 2);
  //CHECK:status = dpct::lapack::potrs_batch(*handle, oneapi::mkl::uplo::upper, 3, 1, a_z_ptrs, 3, b_z_ptrs, 3, infoArray, 2);
  status = cusolverDnSpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, a_s_ptrs, 3, b_s_ptrs, 3, infoArray, 2);
  status = cusolverDnDpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, a_d_ptrs, 3, b_d_ptrs, 3, infoArray, 2);
  status = cusolverDnCpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, a_c_ptrs, 3, b_c_ptrs, 3, infoArray, 2);
  status = cusolverDnZpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, a_z_ptrs, 3, b_z_ptrs, 3, infoArray, 2);

  return 0;
}
