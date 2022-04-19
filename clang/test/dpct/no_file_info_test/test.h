#ifndef LLVM_CLANG_TEST_C2S_NO_FILE_INFO_TEST
#define LLVM_CLANG_TEST_C2S_NO_FILE_INFO_TEST
#include <cuda_runtime.h>
template<typename T>
__host__ __device__ int test(T a, T b);
#endif //LLVM_CLANG_TEST_C2S_NO_FILE_INFO_TEST