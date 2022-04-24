// RUN: dpct --format-range=none -usm-level=none -out-root %T/memcpy_from_symbol %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memcpy_from_symbol/memcpy_from_symbol.dp.cpp %s

#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      std::cout << "Failed" << std::endl;                                      \
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::abort();                                                            \
    }                                                                          \
  } while (false);

static const size_t k_num_elements = 16;

// CHECK: dpct::global_memory<int, 0> a;
__device__ int a;
__global__ void kernel1() { a = 10; }

// CHECK: dpct::global_memory<int, 1> b(k_num_elements);
__device__ int b[k_num_elements];
__global__ void kernel2() { b[threadIdx.x] = threadIdx.x; }

int main() {
  int h_a = 0;
  kernel1<<<1, 1>>>();
  // CHECK: dpct::dpct_memcpy(&h_a, a.get_ptr(), sizeof(h_a));
  cudaMemcpyFromSymbol(&h_a, a, sizeof(h_a));
  CHECK(h_a == 10);

  int h_b[k_num_elements] = {0};
  kernel2<<<1, k_num_elements>>>();
  // CHECK: dpct::dpct_memcpy(&h_b, b.get_ptr(), sizeof(h_b));
  cudaMemcpyFromSymbol(&h_b, b, sizeof(h_b));
  for (size_t i = 0; i < k_num_elements; ++i)
    CHECK(h_b[i] == i);

  // TODO: Add test case for device copy to device after get symbol address API
  //       migrated

  std::cout << "Passed" << std::endl;
  return 0;
}

