// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda-get-error-string.sycl.cpp

int printf(const char *format, ...);

const char *test_function() {

//CHECK:/*
//CHECK-NEXT:SYCLCT1004:{{[0-9]+}}: cudaGetErrorString is not supported in Sycl
//CHECK-NEXT:*/
//CHECK-NEXT:  printf("%s\n", "cudaGetErrorString not supported"/*cudaGetErrorString(0)*/);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));


//CHECK:/*
//CHECK-NEXT:SYCLCT1004:{{[0-9]+}}: cudaGetErrorString is not supported in Sycl
//CHECK-NEXT:*/
//CHECK-NEXT:  printf("%s\n", "cudaGetErrorString not supported"/*cudaGetErrorString(0)*/);
  printf("%s\n", cudaGetErrorString(cudaSuccess));

//CHECK:/*
//CHECK-NEXT:SYCLCT1004:{{[0-9]+}}: cudaGetErrorString is not supported in Sycl
//CHECK-NEXT:*/
//CHECK-NEXT:  printf("%s\n", "cudaGetErrorString not supported"/*cudaGetErrorString(0)*/);
  printf("%s\n", cudaGetErrorString(cudaSuccess));

//CHECK:/*
//CHECK-NEXT:SYCLCT1004:{{[0-9]+}}: cudaGetErrorString is not supported in Sycl
//CHECK-NEXT:*/
//CHECK-NEXT:  return "cudaGetErrorString not supported"/*cudaGetErrorString(0)*/;
  return cudaGetErrorString(cudaSuccess);
}
