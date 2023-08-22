#ifndef LLVM_CLANG_TEST_DPCT_CUDA_ARCH_TEST
#define LLVM_CLANG_TEST_DPCT_CUDA_ARCH_TEST
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK:inline static int Env_cuda_thread_in_threadblock(int axis,
// CHECK-NEXT: const sycl::nd_item<3> &item_ct1);
// CHECK-NEXT:inline static int Env_cuda_thread_in_threadblock_host_ct{{[0-9]+}}(int axis);
__host__ __device__ static int Env_cuda_thread_in_threadblock(int axis);

// CHECK: template<typename T>
// CHECK-NEXT:inline int test(T a, T b, const sycl::nd_item<3> &item_ct1);
// CHECK-NEXT: template<typename T>
// CHECK-NEXT:inline int test_host_ct{{[0-9]+}}(T a, T b);
template<typename T>
__host__ __device__ int test(T a, T b);
#endif //LLVM_CLANG_TEST_DPCT_CUDA_ARCH_TEST