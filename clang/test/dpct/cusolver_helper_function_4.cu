// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T/cusolver_helper_function_4 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolver_helper_function_4/cusolver_helper_function_4.dp.cpp --match-full-lines %s

//CHECK:#include <sycl/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include <dpct/lapack_utils.hpp>
#include "cusolverDn.h"

int foo1() {
  float* a_s;
  double* a_d;
  float2* a_c;
  double2* a_z;
  float* w_s;
  double* w_d;
  float* w_c;
  double* w_z;

  cusolverDnHandle_t handle;
  cusolverDnParams_t params;

  int lwork_s;
  int lwork_d;
  int lwork_c;
  int lwork_z;

  int h_meig_s;
  int h_meig_d;
  int h_meig_c;
  int h_meig_z;

  //CHECK:dpct::lapack::syheevx_scratchpad_size<float, float>(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, 2, 0, 0, 0, 0, &lwork_s);
  //CHECK-NEXT:dpct::lapack::syheevx_scratchpad_size<double, double>(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, 2, 0, 0, 0, 0, &lwork_d);
  //CHECK-NEXT:dpct::lapack::syheevx_scratchpad_size<sycl::float2, float>(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, 2, 0, 0, 0, 0, &lwork_c);
  //CHECK-NEXT:dpct::lapack::syheevx_scratchpad_size<sycl::double2, double>(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, 2, 0, 0, 0, 0, &lwork_z);
  cusolverDnSsyevdx_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_s, 2, 0, 0, 0, 0, &h_meig_s, w_s, &lwork_s);
  cusolverDnDsyevdx_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_d, 2, 0, 0, 0, 0, &h_meig_d, w_d, &lwork_d);
  cusolverDnCheevdx_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_c, 2, 0, 0, 0, 0, &h_meig_c, w_c, &lwork_c);
  cusolverDnZheevdx_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_z, 2, 0, 0, 0, 0, &h_meig_z, w_z, &lwork_z);

  float* device_ws_s;
  double* device_ws_d;
  float2* device_ws_c;
  double2* device_ws_z;

  int *info;

  //CHECK:dpct::lapack::syheevx<float, float>(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, a_s, 2, 0, 0, 0, 0, &h_meig_s, w_s, device_ws_s, lwork_s, info);
  //CHECK-NEXT:dpct::lapack::syheevx<double, double>(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, a_d, 2, 0, 0, 0, 0, &h_meig_d, w_d, device_ws_d, lwork_d, info);
  //CHECK-NEXT:dpct::lapack::syheevx<sycl::float2, float>(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, a_c, 2, 0, 0, 0, 0, &h_meig_c, w_c, device_ws_c, lwork_c, info);
  //CHECK-NEXT:dpct::lapack::syheevx<sycl::double2, double>(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, a_z, 2, 0, 0, 0, 0, &h_meig_z, w_z, device_ws_z, lwork_z, info);
  cusolverDnSsyevdx(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_s, 2, 0, 0, 0, 0, &h_meig_s, w_s, device_ws_s, lwork_s, info);
  cusolverDnDsyevdx(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_d, 2, 0, 0, 0, 0, &h_meig_d, w_d, device_ws_d, lwork_d, info);
  cusolverDnCheevdx(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_c, 2, 0, 0, 0, 0, &h_meig_c, w_c, device_ws_c, lwork_c, info);
  cusolverDnZheevdx(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_z, 2, 0, 0, 0, 0, &h_meig_z, w_z, device_ws_z, lwork_z, info);

  return 0;
}

int foo2() {
  float* a_s;
  double* a_d;
  float2* a_c;
  double2* a_z;
  float* b_s;
  double* b_d;
  float2* b_c;
  double2* b_z;
  float* w_s;
  double* w_d;
  float* w_c;
  double* w_z;

  cusolverDnHandle_t handle;
  cusolverDnParams_t params;

  int lwork_s;
  int lwork_d;
  int lwork_c;
  int lwork_z;

  int h_meig_s;
  int h_meig_d;
  int h_meig_c;
  int h_meig_z;

  //CHECK:dpct::lapack::syhegvx_scratchpad_size<float, float>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, 2, 2, 0, 0, 0, 0, &lwork_s);
  //CHECK-NEXT:dpct::lapack::syhegvx_scratchpad_size<double, double>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, 2, 2, 0, 0, 0, 0, &lwork_d);
  //CHECK-NEXT:dpct::lapack::syhegvx_scratchpad_size<sycl::float2, float>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, 2, 2, 0, 0, 0, 0, &lwork_c);
  //CHECK-NEXT:dpct::lapack::syhegvx_scratchpad_size<sycl::double2, double>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, 2, 2, 0, 0, 0, 0, &lwork_z);
  cusolverDnSsygvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_s, 2, b_s, 2, 0, 0, 0, 0, &h_meig_s, w_s, &lwork_s);
  cusolverDnDsygvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_d, 2, b_d, 2, 0, 0, 0, 0, &h_meig_d, w_d, &lwork_d);
  cusolverDnChegvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_c, 2, b_c, 2, 0, 0, 0, 0, &h_meig_c, w_c, &lwork_c);
  cusolverDnZhegvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_z, 2, b_z, 2, 0, 0, 0, 0, &h_meig_z, w_z, &lwork_z);

  float* device_ws_s;
  double* device_ws_d;
  float2* device_ws_c;
  double2* device_ws_z;

  int *info;

  //CHECK:dpct::lapack::syhegvx<float, float>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, a_s, 2, b_s, 2, 0, 0, 0, 0, &h_meig_s, w_s, device_ws_s, lwork_s, info);
  //CHECK-NEXT:dpct::lapack::syhegvx<double, double>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, a_d, 2, b_d, 2, 0, 0, 0, 0, &h_meig_d, w_d, device_ws_d, lwork_d, info);
  //CHECK-NEXT:dpct::lapack::syhegvx<sycl::float2, float>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, a_c, 2, b_c, 2, 0, 0, 0, 0, &h_meig_c, w_c, device_ws_c, lwork_c, info);
  //CHECK-NEXT:dpct::lapack::syhegvx<sycl::double2, double>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, a_z, 2, b_z, 2, 0, 0, 0, 0, &h_meig_z, w_z, device_ws_z, lwork_z, info);
  cusolverDnSsygvdx(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_s, 2, b_s, 2, 0, 0, 0, 0, &h_meig_s, w_s, device_ws_s, lwork_s, info);
  cusolverDnDsygvdx(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_d, 2, b_d, 2, 0, 0, 0, 0, &h_meig_d, w_d, device_ws_d, lwork_d, info);
  cusolverDnChegvdx(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_c, 2, b_c, 2, 0, 0, 0, 0, &h_meig_c, w_c, device_ws_c, lwork_c, info);
  cusolverDnZhegvdx(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_z, 2, b_z, 2, 0, 0, 0, 0, &h_meig_z, w_z, device_ws_z, lwork_z, info);

  return 0;
}

int foo3() {
  float* a_s;
  double* a_d;
  float2* a_c;
  double2* a_z;
  float* b_s;
  double* b_d;
  float2* b_c;
  double2* b_z;
  float* w_s;
  double* w_d;
  float* w_c;
  double* w_z;

  cusolverDnHandle_t handle;
  syevjInfo_t params;

  int lwork_s;
  int lwork_d;
  int lwork_c;
  int lwork_z;

  //CHECK:dpct::lapack::syhegvd_scratchpad_size<float>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, 2, &lwork_s);
  //CHECK-NEXT:dpct::lapack::syhegvd_scratchpad_size<float>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, 2, &lwork_d);
  //CHECK-NEXT:dpct::lapack::syhegvd_scratchpad_size<sycl::float2>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, 2, &lwork_c);
  //CHECK-NEXT:dpct::lapack::syhegvd_scratchpad_size<sycl::double2>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, 2, &lwork_z);
  cusolverDnSsygvj_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_s, 2, b_s, 2, w_s, &lwork_s, params);
  cusolverDnDsygvj_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_d, 2, b_d, 2, w_d, &lwork_d, params);
  cusolverDnChegvj_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_c, 2, b_c, 2, w_c, &lwork_c, params);
  cusolverDnZhegvj_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_z, 2, b_z, 2, w_z, &lwork_z, params);

  float* device_ws_s;
  double* device_ws_d;
  float2* device_ws_c;
  double2* device_ws_z;

  int *info;

  //CHECK:dpct::lapack::syhegvd<float, float>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, a_s, 2, b_s, 2, w_s, device_ws_s, lwork_s, info);
  //CHECK-NEXT:dpct::lapack::syhegvd<double, double>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, a_d, 2, b_d, 2, w_d, device_ws_d, lwork_d, info);
  //CHECK-NEXT:dpct::lapack::syhegvd<sycl::float2, float>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, a_c, 2, b_c, 2, w_c, device_ws_c, lwork_c, info);
  //CHECK-NEXT:dpct::lapack::syhegvd<sycl::double2, double>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, a_z, 2, b_z, 2, w_z, device_ws_z, lwork_z, info);
  cusolverDnSsygvj(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_s, 2, b_s, 2, w_s, device_ws_s, lwork_s, info, params);
  cusolverDnDsygvj(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_d, 2, b_d, 2, w_d, device_ws_d, lwork_d, info, params);
  cusolverDnChegvj(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_c, 2, b_c, 2, w_c, device_ws_c, lwork_c, info, params);
  cusolverDnZhegvj(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_z, 2, b_z, 2, w_z, device_ws_z, lwork_z, info, params);

  return 0;
}
