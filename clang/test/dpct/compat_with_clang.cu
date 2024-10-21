// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1
// RUN: dpct --format-range=none -out-root %T/compat_with_clang %s --cuda-include-path="%cuda-path/include" --stop-on-parse-err
// RUN: FileCheck %s --match-full-lines --input-file %T/compat_with_clang/compat_with_clang.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/compat_with_clang/compat_with_clang.dp.cpp -o %T/compat_with_clang/compat_with_clang.dp.o %}

#include "cuda_fp16.h"
#include <cstdint>

// CHECK: inline void foo1(sycl::half2 *array, sycl::half a) {
// CHECK-NEXT:   array[dpct::reverse_bits<unsigned int>(123)] = {a, sycl::vec<float, 1>(2.3f).convert<sycl::half, sycl::rounding_mode::automatic>()[0]};
// CHECK-NEXT: }
__device__ inline void foo1(__half2 *array, __half a) {
  array[__brev(123)] = {a, __float2half(2.3f)};
}

// CHECK: void foo2(int a, int b) {
// CHECK-NEXT:   dpct::dim3 block{dpct::min(512, uint32_t(a * b))};
// CHECK-NEXT: }
void foo2(int a, int b) {
  dim3 block{min(512, uint32_t(a * b))};
}

template <class T1, class T2> struct AAAAA {
  template <class T3> void foo(T3 x);
};

// CHECK: template <typename T4, typename T5>
// CHECK-NEXT: template <typename T6>
// CHECK-NEXT: void AAAAA<T4, T5>::foo(T6 x) {}
template <typename T4, typename T5>
template <typename T6>
void AAAAA<T4, T5>::foo<T6>(T6 x) {}

class test1 {
  public:
    __host__ __device__ test1();
  private:
    int state;
};

inline test1::test1() : state(0) {
  printf("hello1\n");
}

__host__ __device__ int test2();

int test2() {
  printf("hello2\n");
  return 0;
}

__global__ void kernel() {
  test1 t1;
  test2();
}

void foo3() {
  kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}
