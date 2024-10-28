// RUN: dpct --format-range=none -out-root %T/pack_expansion_in_arg_list %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --input-file %T/pack_expansion_in_arg_list/pack_expansion_in_arg_list.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/pack_expansion_in_arg_list/pack_expansion_in_arg_list.dp.cpp -o %T/pack_expansion_in_arg_list/pack_expansion_in_arg_list.dp.o %}

#ifndef NO_BUILD_TEST
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

namespace detail {
template <typename Func, typename... Args>
void check_cuda_error(const char *File, unsigned Line, Func &&Fn,
                      Args &&...Arg) {
  static_assert(std::is_same<decltype(Fn(Arg...)), cudaError_t>::value,
                "Fn must return cudaError_t");
  decltype(Fn(Arg...)) Ret = Fn(std::forward<Args>(Arg)...);
  if (Ret != cudaSuccess) {
    abort();
  }
}
} // namespace detail

#define AS_LAMBDA(func)                                                        \
  [&](auto &&...args) -> decltype(func(                                        \
                          std::forward<decltype(args)>(args)...)) {            \
    return func(std::forward<decltype(args)>(args)...);                        \
  }

#define CHECK_CUDA_RET(FN, ...)                                                \
  ::detail::check_cuda_error(__FILE__, __LINE__, AS_LAMBDA(FN), __VA_ARGS__)

int *foo(size_t N, int Low, int High) {
  int *Buffer = nullptr;
  std::vector<int> Vec;

  // 1. Crash in `clang::dpct::getSizeForMalloc`
  // CHECK: CHECK_CUDA_RET(cudaMalloc<int>, &Buffer, sizeof(int) * N);
  CHECK_CUDA_RET(cudaMalloc<int>, &Buffer, sizeof(int) * N);

  // 2. Crash in `clang::dpct::MemoryMigrationRule::memcpyMigration`
  // CHECK: CHECK_CUDA_RET(cudaMemcpy, Buffer, Vec.data(), Vec.size() * sizeof(int), dpct::host_to_device);
  CHECK_CUDA_RET(cudaMemcpy, Buffer, Vec.data(), Vec.size() * sizeof(int), cudaMemcpyHostToDevice);
  return Buffer;
}
#endif
