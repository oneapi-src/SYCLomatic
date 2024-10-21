// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --out-root %T/cusparse-type102 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusparse-type102/cusparse-type102.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cusparse-type102/cusparse-type102.dp.cpp -o %T/cusparse-type102/cusparse-type102.dp.o %}
#include <cstdio>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

// CUSPARSE_STATUS_NOT_SUPPORTED is available since v10.2.
int main() {
  //CHECK: int a6;
  //CHECK-NEXT: a6 = 10;
  cusparseStatus_t a6;
  a6 = CUSPARSE_STATUS_NOT_SUPPORTED;

  //CHECK:/*
  //CHECK-NEXT:DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
  //CHECK-NEXT:*/
  //CHECK-NEXT:printf("Error string: %s", dpct::get_error_string_dummy(a6));
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
  //CHECK-NEXT:*/
  //CHECK-NEXT:printf("Error name: %s", dpct::get_error_string_dummy(a6));
  printf("Error string: %s", cusparseGetErrorString(a6));
  printf("Error name: %s", cusparseGetErrorName(a6));

  //CHECK:dpct::library_data_t b1 = dpct::library_data_t::real_uint16;
  //CHECK-NEXT:b1 = dpct::library_data_t::real_int32;
  //CHECK-NEXT:b1 = dpct::library_data_t::real_int64;
  //CHECK-NEXT:oneapi::mkl::layout b2 = oneapi::mkl::layout::col_major;
  //CHECK-NEXT:b2 = oneapi::mkl::layout::row_major;
  cusparseIndexType_t b1 = CUSPARSE_INDEX_16U;
  b1 = CUSPARSE_INDEX_32I;
  b1 = CUSPARSE_INDEX_64I;
  cusparseOrder_t b2 = CUSPARSE_ORDER_COL;
  b2 = CUSPARSE_ORDER_ROW;

  return 0;
}

//CHECK:void foo(int err) {
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
//CHECK-NEXT:  */
//CHECK-NEXT:  dpct::get_error_string_dummy(err);
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
//CHECK-NEXT:  */
//CHECK-NEXT:  dpct::get_error_string_dummy({{[0-9]+}});
//CHECK-NEXT:}
void foo(cusparseStatus_t err) {
  cusparseGetErrorString(err);
  cusparseGetErrorString(CUSPARSE_STATUS_NOT_INITIALIZED);
}
