#ifndef LLVM_CLANG_TEST_C2S_CUDA_ARCH_TEST
#define LLVM_CLANG_TEST_C2S_CUDA_ARCH_TEST
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: static int Env_cuda_thread_in_threadblock(int axis, sycl::nd_item<3> item_ct1);
// CHECK-NEXT: static int Env_cuda_thread_in_threadblock_host_ct{{[0-9]+}}(int axis);
__host__ __device__ static int Env_cuda_thread_in_threadblock(int axis);

// CHECK: template<typename T>
// CHECK-NEXT: int test(T a, T b, sycl::nd_item<3> item_ct1);
// CHECK-NEXT: template<typename T>
// CHECK-NEXT: int test_host_ct{{[0-9]+}}(T a, T b);
template<typename T>
__host__ __device__ int test(T a, T b);
#endif //LLVM_CLANG_TEST_C2S_CUDA_ARCH_TEST