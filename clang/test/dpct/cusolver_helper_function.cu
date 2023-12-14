// RUN: dpct --format-range=none -out-root %T/cusolver_helper_function %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolver_helper_function/cusolver_helper_function.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cusolver_helper_function/cusolver_helper_function.dp.cpp -o %T/cusolver_helper_function/cusolver_helper_function.dp.o %}

//CHECK:#include <sycl/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include <dpct/lapack_utils.hpp>
#include "cusolverDn.h"

int foo1() {
  cusolverDnHandle_t handle;
  float *a_s, *b_s, *w_s, *work_s;
  double *a_d, *b_d, *w_d, *work_d;
  int lwork_s, lwork_d;
  int *devInfo;
  //CHECK:status = DPCT_CHECK_ERROR(lwork_s = oneapi::mkl::lapack::sygvd_scratchpad_size<float>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3));
  cusolverStatus_t status;
  status = cusolverDnSsygvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s, 3, b_s, 3, w_s, &lwork_s);
  //CHECK:status = dpct::lapack::sygvd(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_s, 3, b_s, 3, w_s, work_s, lwork_s, devInfo);
  status = cusolverDnSsygvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s, 3, b_s, 3, w_s, work_s, lwork_s, devInfo);

  //CHECK:status = DPCT_CHECK_ERROR(lwork_d = oneapi::mkl::lapack::sygvd_scratchpad_size<double>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3));
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

  //CHECK:status = DPCT_CHECK_ERROR(lwork_c = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<float>>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3));
  status = cusolverDnChegvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_c, 3, b_c, 3, w_c, &lwork_c);
  //CHECK:status = dpct::lapack::hegvd(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_c, 3, b_c, 3, w_c, work_c, lwork_c, devInfo);
  status = cusolverDnChegvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_c, 3, b_c, 3, w_c, work_c, lwork_c, devInfo);

  //CHECK:status = DPCT_CHECK_ERROR(lwork_z = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<double>>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3));
  status = cusolverDnZhegvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_z, 3, b_z, 3, w_z, &lwork_z);
  //CHECK:status = dpct::lapack::hegvd(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_z, 3, b_z, 3, w_z, work_z, lwork_z, devInfo);
  status = cusolverDnZhegvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_z, 3, b_z, 3, w_z, work_z, lwork_z, devInfo);

  return 0;
}

void foo3() {
  cusolverDnHandle_t handle;
  float *a_s, *work_s;
  double *a_d, *work_d;
  float *w_s;
  double *w_d;
  int lwork_s, lwork_d;
  int *devInfo;
  cusolverStatus_t status;

  //CHECK:status = dpct::lapack::syheevd_scratchpad_size<float>(*handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, &lwork_s);
  status = cusolverDnSsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s, 3, w_s, &lwork_s);
  //CHECK:status = dpct::lapack::syheevd<float, float>(*handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_s, 3, w_s, work_s, lwork_s, devInfo);
  status = cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s, 3, w_s, work_s, lwork_s, devInfo);

  //CHECK:status = dpct::lapack::syheevd_scratchpad_size<double>(*handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, &lwork_d);
  status = cusolverDnDsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d, 3, w_d, &lwork_d);
  //CHECK:status = dpct::lapack::syheevd<double, double>(*handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_d, 3, w_d, work_d, lwork_d, devInfo);
  status = cusolverDnDsyevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d, 3, w_d, work_d, lwork_d, devInfo);
}

void foo4() {
  cusolverDnHandle_t handle;
  float2 *a_c, *b_c, *work_c;
  double2 *a_z, *b_z, *work_z;
  float *w_c;
  double *w_z;
  int lwork_c, lwork_z;
  int *devInfo;
  cusolverStatus_t status;

  //CHECK:status = dpct::lapack::syheevd_scratchpad_size<std::complex<float>>(*handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, &lwork_c);
  status = cusolverDnCheevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_c, 3, w_c, &lwork_c);
  //CHECK:status = dpct::lapack::syheevd<sycl::float2, float>(*handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_c, 3, w_c, work_c, lwork_c, devInfo);
  status = cusolverDnCheevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_c, 3, w_c, work_c, lwork_c, devInfo);

  //CHECK:status = dpct::lapack::syheevd_scratchpad_size<std::complex<double>>(*handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, &lwork_z);
  status = cusolverDnZheevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_z, 3, w_z, &lwork_z);
  //CHECK:status = dpct::lapack::syheevd<sycl::double2, double>(*handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_z, 3, w_z, work_z, lwork_z, devInfo);
  status = cusolverDnZheevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_z, 3, w_z, work_z, lwork_z, devInfo);
}
