// RUN: dpct --format-range=none -out-root %T/overload_unary_operator %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/overload_unary_operator/overload_unary_operator.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/overload_unary_operator/overload_unary_operator.dp.cpp -o %T/overload_unary_operator/overload_unary_operator.dp.o %}

#include <cuda_runtime.h>

// CHECK: namespace dpct_operator_overloading {
// CHECK: static sycl::uint2 operator~(sycl::uint2 test) {
// CHECK-NEXT:    return sycl::uint2(~test.x(), ~test.y());
// CHECK-NEXT: }
static __device__ uint2 operator~(uint2 test) {
    return make_uint2(~test.x, ~test.y);
}

// CHECK: namespace dpct_operator_overloading {
// CHECK: static sycl::uint2 operator&(const sycl::uint2 a, const sycl::uint2 b) {
// CHECK-NEXT:   return sycl::uint2(a.x() & b.x(), b.y() & a.y());
// CHECK-NEXT: }
static __device__ uint2 operator&(const uint2 a, const uint2 b) {
    return make_uint2(a.x & b.x, b.y & a.y);
}
// CHECK: namespace dpct_operator_overloading {
// CHECK: static sycl::uint2 operator^(const sycl::uint2 a, const sycl::uint2 b) {
// CHECK-NEXT:    return sycl::uint2(a.x() ^ b.x(), a.y() ^ b.y());
// CHECK-NEXT: }
static __device__ uint2 operator^(const uint2 a, const uint2 b) {
    return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

// CHECK: sycl::uint2 chi(const sycl::uint2 a, const sycl::uint2 b, const sycl::uint2 c) {
// CHECK: return dpct_operator_overloading::operator^(a , dpct_operator_overloading::operator&((dpct_operator_overloading::operator~(b)) , c));

__device__ uint2 chi(const uint2 a, const uint2 b, const uint2 c) {
    return a ^ (~b) & c;
};

// The code piece below is used to test the assert that `Name.isIdentifier() &&
// "Name is not a simple identifier"' when assert is enabled to build
// SYCLomatic.
template <typename T, int m> struct array {
  __host__ __device__ bool operator==(const array &other) const { return true; }
  __host__ __device__ bool operator!=(const array &other) const {
    return !operator==(other);
  }
};
