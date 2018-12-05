// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda-get-error-string.sycl.cpp

int printf(const char* format, ...);

void test_function() {
  // CHECK:printf("%s\n", "cudaGetErrorString not supported"/*cudaGetErrorString(0)*/);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
 
  // CHECK:printf("%s\n", "cudaGetErrorString not supported"/*cudaGetErrorString(0)*/);
  printf("%s\n", cudaGetErrorString(cudaSuccess));
}

