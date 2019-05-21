// RUN: syclct -out-root=%T/abd -in-root=%S %S/ab*.cu -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/abd/abd.sycl.cpp --match-full-lines %S/abd.cu
// RUN: FileCheck --input-file %T/abd/abc.sycl.cpp --match-full-lines %S/abc.cu

int printf(const char *format, ...);

const char *test_function() {

//CHECK:/*
//CHECK-NEXT:SYCLCT1009:{{[0-9]+}}: SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, cudaGetErrorString is commented out and a warning string is inserted. You may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:/*
//CHECK-NEXT:SYCLCT1010:{{[0-9]+}}: SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, cudaGetLastError was replaced with 0. You may need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:  printf("%s\n", "cudaGetErrorString not supported"/*cudaGetErrorString(0)*/);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
