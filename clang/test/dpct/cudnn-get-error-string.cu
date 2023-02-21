// RUN: dpct --format-range=none -out-root %T/cudnn-get-error-string %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cudnn-get-error-string/cudnn-get-error-string.dp.cpp

#include <cudnn.h>

int printf(const char *format, ...);

// CHECK: /*
// CHECK-NEXT: DPCT1009:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
// CHECK-NEXT: */
// CHECK-NEXT: #define PRINT_ERROR_STR(X) printf("%s\n", "cudnnGetErrorString is not supported"/*cudnnGetErrorString(X)*/)
#define PRINT_ERROR_STR(X) printf("%s\n", cudnnGetErrorString(X))

// CHECK:  /*
// CHECK-NEXT:  DPCT1009:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
// CHECK-NEXT:  */
// CHECK-NEXT: #define PRINT_ERROR_STR2(X)\
// CHECK-NEXT:  printf("%s\n", "cudnnGetErrorString is not supported"/*cudnnGetErrorString(X)*/)
#define PRINT_ERROR_STR2(X)\
  printf("%s\n", cudnnGetErrorString(X))

// CHECK: /*
// CHECK-NEXT: DPCT1009:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
// CHECK-NEXT: */
// CHECK-NEXT: #define PRINT_ERROR_STR3(X)\
// CHECK-NEXT:   printf("%s\
// CHECK-NEXT:          \n", "cudnnGetErrorString is not supported"/*cudnnGetErrorString(X)*/)
#define PRINT_ERROR_STR3(X)\
  printf("%s\
         \n", cudnnGetErrorString(X))


const char *test_function(cudnnStatus_t status) {

//CHECK:/*
//CHECK-NEXT:DPCT1009:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:  printf("%s\n", "cudnnGetErrorString is not supported"/*cudnnGetErrorString(status)*/);
  printf("%s\n", cudnnGetErrorString(status));

//CHECK:/*
//CHECK-NEXT:DPCT1009:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:  printf("%s\n", "cudnnGetErrorString is not supported"/*cudnnGetErrorString(CUDNN_STATUS_SUCCESS)*/);
  printf("%s\n", cudnnGetErrorString(CUDNN_STATUS_SUCCESS));

  PRINT_ERROR_STR(status);
  PRINT_ERROR_STR2(status);
  PRINT_ERROR_STR3(status);  
}

