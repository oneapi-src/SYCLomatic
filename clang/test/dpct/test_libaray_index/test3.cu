// RUN: dpct --format-range=none -out-root %T/output %S/test1.cu %S/test2.cu %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/output/test3.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/output/test3.dp.cpp -o %T/output/test3.dp.o %}

#include "cublas_v2.h"

int foo () {
  cublasStatus_t s;
  cublasHandle_t handle;
  int N = 275;
  float *x1;
  int *result;

  //CHECK:[&]() {
  //CHECK-NEXT:dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), result);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::iamax(handle->get_queue(), N, x1, N, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}();
  cublasIsamax(handle, N, x1, N, result);
}
