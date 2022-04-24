// RUN: dpct --format-range=none --usm-level=none -in-root %S -out-root %T/no_file_info_test %S/test.cu %s -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++11 -x cuda --cuda-host-only
// RUN: FileCheck %S/test.cu --match-full-lines --input-file %T/no_file_info_test/test.dp.cpp
// RUN: rm -rf %T/no_file_info_test/*
// CHECK: #include "test.h"
#include "test.h"

template<typename T>
__host__ __device__ int test(T a, T b){
#ifdef __CUDA_ARCH__
  return threadIdx.x > 10 ? a : b;
#else
  cudaDeviceSynchronize();
  return a;
#endif
}

int main(){
  test<int>(1, 1);
  return 0;
}
