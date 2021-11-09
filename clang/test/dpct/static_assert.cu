// RUN: dpct --format-range=none -out-root %T/static_assert %s --stop-on-parse-err --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/static_assert/static_assert.dp.cpp
#include <complex>

namespace user_namespace {
template <typename T>
struct complex {
  T real_;
  T imag_;

  constexpr complex(const T& re, const T& im)
      : real_(re), imag_(im) {}

  constexpr T real() const {
    return real_;
  }
  constexpr T imag() const {
    return imag_;
  }
};
}

namespace std {
template <typename T>
constexpr T norm(const user_namespace::complex<T>& z) {
  return z.real() * z.real() + z.imag() * z.imag();
}
}

__host__ __device__ void test() {
  // CHECK:static_assert(std::norm(user_namespace::complex<int>(3, 4)) == 25,"");
  static_assert(std::norm(user_namespace::complex<int>(3, 4)) == 25,"");
}

__global__ void k() {
  test();
}