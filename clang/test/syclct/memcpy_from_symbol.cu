// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --match-full-lines --input-file %T/memcpy_from_symbol.sycl.cpp %s

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

// CHECK: syclct::device_memory<int, 0> a(syclct::syclct_range<0>());
__device__ int a;
__global__ void kernel1() { a = 10; }

// CHECK: syclct::device_memory<int, 1> b(syclct::syclct_range<1>(16));
__device__ int b[k_num_elements];
__global__ void kernel2() { b[threadIdx.x] = threadIdx.x; }

int main() {
  int h_a = 0;
  kernel1<<<1, 1>>>();
  // CHECK: syclct::sycl_memcpy_from_symbol((void*)(&h_a), a.get_ptr(), sizeof (h_a));
  cudaMemcpyFromSymbol(&h_a, a, sizeof(h_a));
  CHECK(h_a == 10);

  int h_b[k_num_elements] = {0};
  kernel2<<<1, k_num_elements>>>();
  // CHECK: syclct::sycl_memcpy_from_symbol((void*)(&h_b), b.get_ptr(), sizeof (h_b));
  cudaMemcpyFromSymbol(&h_b, b, sizeof(h_b));
  for (size_t i = 0; i < k_num_elements; ++i)
    CHECK(h_b[i] == i);

  // TODO: Add test case for device copy to device after get symbol address API
  //       translated

  std::cout << "Passed" << std::endl;
  return 0;
}
