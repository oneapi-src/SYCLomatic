// RUN: dpct --format-range=none -out-root %T/cuda-get-error-string %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda-get-error-string/cuda-get-error-string.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cuda-get-error-string/cuda-get-error-string.dp.cpp -o %T/cuda-get-error-string/cuda-get-error-string.dp.o %}

int printf(const char *format, ...);

// CHECK: #define PRINT_ERROR_STR(X) printf("%s\n", dpct::get_error_dummy(X))
#define PRINT_ERROR_STR(X) printf("%s\n", cudaGetErrorString(X))

// CHECK: #define PRINT_ERROR_STR2(X)\
// CHECK-NEXT:  printf("%s\n", dpct::get_error_dummy(X))
#define PRINT_ERROR_STR2(X)\
  printf("%s\n", cudaGetErrorString(X))

// CHECK: #define PRINT_ERROR_STR3(X)\
// CHECK-NEXT:   printf("%s\
// CHECK-NEXT:          \n", dpct::get_error_dummy(X))
#define PRINT_ERROR_STR3(X)\
  printf("%s\
         \n", cudaGetErrorString(X))

// CHECK: #define PRINT_ERROR_NAME(X) printf("%s\n", dpct::get_error_dummy(X))
#define PRINT_ERROR_NAME(X) printf("%s\n", cudaGetErrorName(X))

// CHECK: #define PRINT_ERROR_NAME2(X)\
// CHECK-NEXT:   printf("%s\n", dpct::get_error_dummy(X))
#define PRINT_ERROR_NAME2(X)\
  printf("%s\n", cudaGetErrorName(X))

// CHECK: #define PRINT_ERROR_NAME3(X)\
// CHECK-NEXT:   printf("%s\
// CHECK-NEXT:          \n", dpct::get_error_dummy(X))
#define PRINT_ERROR_NAME3(X)\
  printf("%s\
         \n", cudaGetErrorName(X))

// CHECK: #define PRINT_ERROR_STR_NAME(X)\
// CHECK-NEXT:   printf("%s\
// CHECK-NEXT:          %s\
// CHECK-NEXT:          \n", dpct::get_error_dummy(X),\
// CHECK-NEXT:          dpct::get_error_dummy(X))
#define PRINT_ERROR_STR_NAME(X)\
  printf("%s\
         %s\
         \n", cudaGetErrorString(X),\
         cudaGetErrorName(X))

const char *test_function() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_STR(cudaGetLastError());
  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_NAME(cudaGetLastError());
  PRINT_ERROR_STR(cudaSuccess);
  PRINT_ERROR_NAME(cudaSuccess);

  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_STR2(cudaGetLastError());
  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_NAME2(cudaGetLastError());
  PRINT_ERROR_STR2(cudaSuccess);
  PRINT_ERROR_NAME2(cudaSuccess);

  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_STR3(cudaGetLastError());
  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_NAME3(cudaGetLastError());
  PRINT_ERROR_STR3(cudaSuccess);
  PRINT_ERROR_NAME3(cudaSuccess);

  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_STR_NAME(cudaGetLastError());
  PRINT_ERROR_STR_NAME(cudaSuccess);

//CHECK:/*
//CHECK-NEXT:DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:  printf("%s\n", dpct::get_error_dummy(0));
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));


//CHECK:  printf("%s\n", dpct::get_error_dummy(0));
  printf("%s\n", cudaGetErrorString(cudaSuccess));

//CHECK:printf("%s\n", dpct::get_error_dummy(0));
  printf("%s\n", cudaGetErrorName(cudaSuccess));
  CUresult e;
  const char *err_s;

//CHECK:  err_s = dpct::get_error_dummy(e);
  cuGetErrorString(e, &err_s);

//CHECK:  return dpct::get_error_dummy(0);
  return cudaGetErrorName(cudaSuccess);
}

