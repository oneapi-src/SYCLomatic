// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/half_raw %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/half_raw/half_raw.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/half_raw/half_raw.dp.cpp -o %T/half_raw/half_raw.dp.o %}

#include <cuda_fp16.h>
#include <cuda_runtime.h>

void foo1() {
  // CHECK: uint16_t one_h{0x3C00};
  __half_raw one_h{0x3C00};
  // CHECK: uint16_t zero_h{0};
  __half_raw zero_h{0};
  // CHECK: uint16_t *ptr = new uint16_t{0};
  __half_raw *ptr = new __half_raw{0};
  // clang-format off
  // CHECK: *ptr = 0x3C00;
  ptr->x = 0x3C00;
  // CHECK: *ptr = 0x3C00;
  ptr ->x = 0x3C00;
  // CHECK: *ptr = 0x3C00;
  ptr-> x = 0x3C00;
  // CHECK: *ptr = 0x3C00;
  ptr -> x = 0x3C00;
  // CHECK: zero_h = 0x3C00;
  zero_h.x = 0x3C00;
  // CHECK: zero_h = 0x3C00;
  zero_h .x = 0x3C00;
  // CHECK: zero_h = 0x3C00;
  zero_h. x = 0x3C00;
  // CHECK: zero_h = 0x3C00;
  zero_h . x = 0x3C00;
  // clang-format on
  // CHECK: sycl::half alpha = sycl::bit_cast<sycl::half>(one_h);
  half alpha = one_h;
  // CHECK: alpha = sycl::bit_cast<sycl::half>(one_h);
  alpha = one_h;
  // CHECK: uint16_t as = zero_h;
  uint16_t as = zero_h.x;
  // CHECK: uint16_t *ptr1 = &one_h;
  __half_raw *ptr1 = &one_h;
}

void foo2() {
  // CHECK: sycl::ushort2 one_h{0x3C00};
  __half2_raw one_h{0x3C00};
  // CHECK: sycl::ushort2 zero_h{0};
  __half2_raw zero_h{0};
  // CHECK: sycl::ushort2 *ptr = new sycl::ushort2{0};
  __half2_raw *ptr = new __half2_raw{0};
  // clang-format off
  // CHECK: ptr->x() = 0x3C00;
  ptr->x = 0x3C00;
  // CHECK: ptr ->x() = 0x3C00;
  ptr ->x = 0x3C00;
  // CHECK: ptr-> x() = 0x3C00;
  ptr-> x = 0x3C00;
  // CHECK: ptr -> x() = 0x3C00;
  ptr -> x = 0x3C00;
  // CHECK: zero_h.x() = 0x3C00;
  zero_h.x = 0x3C00;
  // CHECK: zero_h .x() = 0x3C00;
  zero_h .x = 0x3C00;
  // CHECK: zero_h. x() = 0x3C00;
  zero_h. x = 0x3C00;
  // CHECK: zero_h . x() = 0x3C00;
  zero_h . x = 0x3C00;
  // CHECK: ptr->y() = 0x3C00;
  ptr->y = 0x3C00;
  // CHECK: ptr ->y() = 0x3C00;
  ptr ->y = 0x3C00;
  // CHECK: ptr-> y() = 0x3C00;
  ptr-> y = 0x3C00;
  // CHECK: ptr -> y() = 0x3C00;
  ptr -> y = 0x3C00;
  // CHECK: zero_h.y() = 0x3C00;
  zero_h.y = 0x3C00;
  // CHECK: zero_h .y() = 0x3C00;
  zero_h .y = 0x3C00;
  // CHECK: zero_h. y() = 0x3C00;
  zero_h. y = 0x3C00;
  // CHECK: zero_h . y() = 0x3C00;
  zero_h . y = 0x3C00;
  // clang-format on
  // CHECK: sycl::half2 alpha = one_h.as<sycl::half2>();
  half2 alpha = one_h;
  // CHECK: alpha = one_h.as<sycl::half2>();
  alpha = one_h;
  // CHECK: uint16_t as = zero_h.x();
  uint16_t as = zero_h.x;
  // CHECK: sycl::ushort2 *ptr1 = &one_h;
  __half2_raw *ptr1 = &one_h;
}
