// RUN: dpct --format-range=none -out-root %T/auto_deref %s --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/auto_deref/auto_deref.dp.cpp
// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>

template <typename DataType>
void test() {
  DataType *tensor_mem_[3];
  size_t maxSize = 64;
  for (auto &mem : tensor_mem_) {
    // CHECK: mem = (typename std::remove_reference<decltype(mem)>::type)sycl::malloc_device(maxSize, q_ct1);
    cudaMalloc(&mem, maxSize);
    cudaMemset(mem, 0, maxSize);
  }
}

void test1() {
  int *tensor_mem_[3];
  size_t maxSize = 64;
  for (auto &mem : tensor_mem_) {
    // CHECK: mem = (int *)sycl::malloc_device(maxSize, q_ct1);
    cudaMalloc(&mem, maxSize);
    cudaMemset(mem, 0, maxSize);
  }
}

int main() {
  test<int>();
  test<float>();
  test1();
  return 1;
}