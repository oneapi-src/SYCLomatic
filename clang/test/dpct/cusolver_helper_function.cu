// RUN: dpct --format-range=none -out-root %T/cusolver_helper_function %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolver_helper_function/cusolver_helper_function.dp.cpp --match-full-lines %s

//CHECK:#include <sycl/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include <dpct/lapack_utils.hpp>
#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  float *a_s, *b_s, *w_s, *work_s;
  int lwork_s;
  int *devInfo;
  //CHECK:status = (lwork_s = oneapi::mkl::lapack::sygvd_scratchpad_size<float>(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3), 0);
  cusolverStatus_t status;
  status = cusolverDnSsygvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s, 3, b_s, 3, w_s, &lwork_s);
  //CHECK:status = dpct::lapack::sygvd(*handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, a_s, 3, b_s, 3, w_s, work_s, lwork_s, devInfo);
  status = cusolverDnSsygvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s, 3, b_s, 3, w_s, work_s, lwork_s, devInfo);
  return 0;
}
