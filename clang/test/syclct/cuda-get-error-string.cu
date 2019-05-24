// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda-get-error-string.sycl.cpp

int printf(const char *format, ...);

// CHECK: /*
// CHECK-NEXT: SYCLCT1009:{{[0-9]+}}: SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, cudaGetErrorString is commented out and a warning string is inserted. You may need to rewrite this code.
// CHECK-NEXT: */
#define PRINT_ERROR_STR(X) printf("%s\n", cudaGetErrorString(X))
// CHECK: /*
// CHECK-NEXT: SYCLCT1009:{{[0-9]+}}: SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, cudaGetErrorName is commented out and a warning string is inserted. You may need to rewrite this code.
// CHECK-NEXT: */
#define PRINT_ERROR_NAME(X) printf("%s\n", cudaGetErrorName(X))

const char *test_function() {
  // CHECK: /*
  // CHECK-NEXT: SYCLCT1010:{{[0-9]+}}: SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, cudaGetLastError was replaced with 0. You may need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_STR(cudaGetLastError());
  PRINT_ERROR_STR(cudaSuccess);
  PRINT_ERROR_NAME(cudaSuccess);

//CHECK:/*
//CHECK-NEXT:SYCLCT1009:{{[0-9]+}}: SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, cudaGetErrorString is commented out and a warning string is inserted. You may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:/*
//CHECK-NEXT:SYCLCT1010:{{[0-9]+}}: SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, cudaGetLastError was replaced with 0. You may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:  printf("%s\n", "cudaGetErrorString not supported"/*cudaGetErrorString(0)*/);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));


//CHECK:/*
//CHECK-NEXT:SYCLCT1009:{{[0-9]+}}: SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, cudaGetErrorString is commented out and a warning string is inserted. You may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:  printf("%s\n", "cudaGetErrorString not supported"/*cudaGetErrorString(0)*/);
  printf("%s\n", cudaGetErrorString(cudaSuccess));

//CHECK:/*
//CHECK-NEXT:SYCLCT1009:{{[0-9]+}}: SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, cudaGetErrorName is commented out and a warning string is inserted. You may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:printf("%s\n", "cudaGetErrorName not supported"/*cudaGetErrorName(0)*/);
  printf("%s\n", cudaGetErrorName(cudaSuccess));

//CHECK:/*
//CHECK-NEXT:SYCLCT1009:{{[0-9]+}}: SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, cudaGetErrorName is commented out and a warning string is inserted. You may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:  return "cudaGetErrorName not supported"/*cudaGetErrorName(0)*/;
  return cudaGetErrorName(cudaSuccess);
}
