// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3
// RUN: dpct --format-range=none -out-root %T/cusolver_helper_function_5 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolver_helper_function_5/cusolver_helper_function_5.dp.cpp --match-full-lines %s

//CHECK:#include <sycl/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include <dpct/lapack_utils.hpp>
#include "cusolverDn.h"

int foo1() {
  float *a_s;
  double *a_d;
  float2 *a_c;
  double2 *a_z;

  cusolverDnHandle_t handle;

  size_t lwork_s;
  size_t lwork_d;
  size_t lwork_c;
  size_t lwork_z;
  size_t lwork_host_s;
  size_t lwork_host_d;
  size_t lwork_host_c;
  size_t lwork_host_z;

  //CHECK:dpct::lapack::trtri_scratchpad_size(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2, dpct::library_data_t::real_float, 2, &lwork_s, &lwork_host_s);
  //CHECK-NEXT:dpct::lapack::trtri_scratchpad_size(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2, dpct::library_data_t::real_double, 2, &lwork_d, &lwork_host_d);
  //CHECK-NEXT:dpct::lapack::trtri_scratchpad_size(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2, dpct::library_data_t::complex_float, 2, &lwork_c, &lwork_host_c);
  //CHECK-NEXT:dpct::lapack::trtri_scratchpad_size(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2, dpct::library_data_t::complex_double, 2, &lwork_z, &lwork_host_z);
  cusolverDnXtrtri_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_R_32F, a_s, 2, &lwork_s, &lwork_host_s);
  cusolverDnXtrtri_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_R_64F, a_d, 2, &lwork_d, &lwork_host_d);
  cusolverDnXtrtri_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_C_32F, a_c, 2, &lwork_c, &lwork_host_c);
  cusolverDnXtrtri_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_C_64F, a_z, 2, &lwork_z, &lwork_host_z);

  void *device_ws_s;
  void *device_ws_d;
  void *device_ws_c;
  void *device_ws_z;
  void *host_ws_s;
  void *host_ws_d;
  void *host_ws_c;
  void *host_ws_z;

  int *info;

  //CHECK:dpct::lapack::trtri(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2, dpct::library_data_t::real_float, a_s, 2, device_ws_s, lwork_s, info);
  //CHECK-NEXT:dpct::lapack::trtri(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2, dpct::library_data_t::real_double, a_d, 2, device_ws_d, lwork_d, info);
  //CHECK-NEXT:dpct::lapack::trtri(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2, dpct::library_data_t::complex_float, a_c, 2, device_ws_c, lwork_c, info);
  //CHECK-NEXT:dpct::lapack::trtri(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2, dpct::library_data_t::complex_double, a_z, 2, device_ws_z, lwork_z, info);
  cusolverDnXtrtri(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_R_32F, a_s, 2, device_ws_s, lwork_s, host_ws_s, lwork_host_s, info);
  cusolverDnXtrtri(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_R_64F, a_d, 2, device_ws_d, lwork_d, host_ws_d, lwork_host_d, info);
  cusolverDnXtrtri(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_C_32F, a_c, 2, device_ws_c, lwork_c, host_ws_c, lwork_host_c, info);
  cusolverDnXtrtri(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_C_64F, a_z, 2, device_ws_z, lwork_z, host_ws_z, lwork_host_z, info);

  return 0;
}
