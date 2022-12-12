// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/cusolver_helper_function_2 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolver_helper_function_2/cusolver_helper_function_2.dp.cpp --match-full-lines %s

//CHECK:#include <sycl/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include <oneapi/mkl.hpp>
//CHECK-NEXT:#include <dpct/lapack_utils.hpp>
#include "cusolverDn.h"

int foo1() {
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

int foo2() {
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
