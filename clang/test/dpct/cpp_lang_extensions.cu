// RUN: dpct --format-range=none -out-root %T/cpp_lang_extensions %s --cuda-include-path="%cuda-path/include" -extra-arg="-I%S" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cpp_lang_extensions/cpp_lang_extensions.dp.cpp --match-full-lines %s

#include "cpp_lang_extensions.cuh"

__device__ float df(float f) {
  float a[23];
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to __ldg was removed because there is no correspoinding API in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: f;
  __ldg(&f);
  int *pi;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to __ldg was removed because there is no correspoinding API in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: *pi;
  __ldg(pi);
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to __ldg was removed because there is no correspoinding API in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: *(pi + 2);
  __ldg(pi + 2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to __ldg was removed because there is no correspoinding API in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: return 45 * a[23] * f * 23;
  return 45 * __ldg(&a[23]) * f * 23;
}

#define DEV_INLINE __device__ __forceinline__
__device__ __constant__ uint2 const keccak_round_constants[4] = {
    { 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 }, { 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};


// CHECK:namespace dpct_operator_overloading {
// CHECK:inline sycl::uint2 &operator^=(sycl::uint2 &v, const sycl::uint2 &v2) {
// CHECK:  return v;
// CHECK:}
// CHECK:}  // namespace dpct_operator_overloading
__host__ __device__ inline uint2 &operator^=(uint2 &v, const uint2 &v2) {
  return v;
}


DEV_INLINE void SHA3_512(uint2* s) {
    int i;
  // CHECK:    dpct_operator_overloading::operator^=(s[0] , LDG(keccak_round_constants[i]));
  // CHECK-NEXT:    LDG(keccak_round_constants[23]);
    s[0] ^= LDG(keccak_round_constants[i]);
    LDG(keccak_round_constants[23]);
}
