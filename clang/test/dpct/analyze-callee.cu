// RUN: dpct --format-range=none -out-root %T/analyze-callee %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/analyze-callee/analyze-callee.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/analyze-callee/analyze-callee.dp.cpp -o %T/analyze-callee/analyze-callee.dp.o %}

#include <cuda_fp16.h>

template <typename T> struct Math {
  static inline __device__ T zero() { return (T)0; }
};

template <> struct Math<int> {
  static inline __device__ half2 zero() {
    // CHECK: return sycl::half2(Math<sycl::half>::zero());
    return __half2half2(Math<half>::zero());
  }
};

template <typename T>
struct Math1 {
    static inline __device__ T add(T a, T b) {
      // CHECK: return a + b;
        return a + b;
    }
};

template <>
struct Math1<half2> {
    static inline __device__ half2 add(half2 a, half2 b) {
  // CHECK: return a + b;
        return __hadd2(a, b);
    }
};

template<class T>
__global__ void kernel() {
 half2 a, b;
// CHECK: a = Math1<sycl::half2>::add(a, b);
 a = Math1<half2>::add(a, b);
}

int main() {
  kernel<int><<<1, 1>>>();
  return 0;
}

