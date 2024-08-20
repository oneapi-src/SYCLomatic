// RUN: dpct --format-range=none --out-root %T/user_defined_half_raw %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/user_defined_half_raw/user_defined_half_raw.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/user_defined_half_raw/user_defined_half_raw.dp.cpp -o %T/user_defined_half_raw/user_defined_half_raw.dp.o %}

struct __half_raw {
  __device__ __half_raw() : _raw(0) {}
  explicit __device__ __half_raw(unsigned short raw) : _raw(raw) {}
  unsigned short _raw;
};

//      CHECK: __half_raw foo(unsigned short x) {
// CHECK-NEXT:   __half_raw h;
// CHECK-NEXT:   h._raw = x;
// CHECK-NEXT:   return h;
// CHECK-NEXT: }
__device__ __half_raw foo(unsigned short x) {
  __half_raw h;
  h._raw = x;
  return h;
}
