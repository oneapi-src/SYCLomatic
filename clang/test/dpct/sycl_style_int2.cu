// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/sycl_style_int2.dp.cpp --match-full-lines %s
#ifdef _WIN32
#include <cstdint>
#endif
// CHECK: void func3(sycl::int2 a, sycl::int2 b, sycl::int2 c) {
void func3(int2 a, int2 b, int2 c) {
}
// CHECK: void fun(sycl::int2 a) {}
void fun(int2 a) {}

// CHECK: void kernel(sycl::int2* data) {}
__global__ void kernel(int2* data) {}

int main() {
  // range default constructor does the right thing.
  // CHECK: sycl::int2 deflt;
  int2 deflt;

  // CHECK: sycl::int2 copyctor1 = sycl::int2(1, 2);
  int2 copyctor1 = make_int2(1, 2);

  // CHECK: sycl::int2 copyctor2 = sycl::int2(copyctor1);
  int2 copyctor2 = int2(copyctor1);

  // CHECK: sycl::int2 copyctor3(copyctor1);
  int2 copyctor3(copyctor1);

  // CHECK: func3(deflt, sycl::int2(deflt), (sycl::int2)deflt);
  func3(deflt, int2(deflt), (int2)deflt);

  // CHECK: sycl::int2 *i4;
  int2 *i4;
  // CHECK: sycl::int2 *i5;
  int2 *i5;
  // CHECK: sycl::int2 i6;
  int2 i6;
  // CHECK: sycl::int2 i7;
  int2 i7;
  // CHECK: int i = i6.x();
  int i = i6.x;
  // CHECK: i6.x() = i7.x();
  i6.x = i7.x;
  // CHECK: if (i6.x() == i7.x()) {
  if (i6.x == i7.x) {
  }
  // CHECK: sycl::int2 i2_array[10];
  int2 i2_array[10];
  // CHECK: sycl::int2 i2_array2[10];
  int2 i2_array2[10];
  // CHECK: if (i2_array[1].x() == i2_array2[1].x()) {
  if (i2_array[1].x == i2_array2[1].x) {
  }
  // CHECK: sycl::int2 x = sycl::int2(1, 2);
  int2 x = make_int2(1, 2);
  // CHECK: i4 = (sycl::int2 *)i2_array;
  i4 = (int2 *)i2_array;
  // CHECK: i7 = (sycl::int2)i6;
  i7 = (int2)i6;
  // CHECK: i7 = sycl::int2(i6);
  i7 = int2(i6);

  // CHECK: struct benchtype_s {
  // CHECK:   uint64_t u64;
  // CHECK:   sycl::uint2 u32;
  // CHECK:   sycl::uint2 *u32_p ;
  // CHECK:   sycl::uint2 array[10];
  // CHECK: };
  struct benchtype_s {
    uint64_t u64;
    uint2 u32;
    uint2 *u32_p ;
    uint2 array[10];
  };

  // CHECK: union benchtype_u {
  // CHECK:   uint64_t u64;
  // CHECK:   sycl::uint2 u32{};
  // CHECK:   sycl::uint2 *u32_p ;
  // CHECK:   sycl::uint2 array[10];
  // CHECK: };
  union benchtype_u {
    uint64_t u64;
    uint2 u32;
    uint2 *u32_p ;
    uint2 array[10];
  };

  // CHECK: sycl::int2* data;
  // CHECK-NEXT: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> data_buf_ct0 = dpct::get_buffer_and_offset(data);
  // CHECK-NEXT:   size_t data_offset_ct0 = data_buf_ct0.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto data_acc_ct0 = data_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           sycl::int2 *data_ct0 = (sycl::int2 *)(&data_acc_ct0[0] + data_offset_ct0);
  // CHECK-NEXT:           kernel(data_ct0);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  int2* data;
  kernel<<<1, 1>>>(data);

  // CHECK: volatile int aaa1;
  // CHECK-NEXT: aaa1 = 1;
  volatile int1 aaa1;
  aaa1.x = 1;
  // CHECK: /*
  // CHECK-NEXT: DPCT1052:{{[0-9]+}}: DPC++ does not support the member access for volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::int2 aaa2;
  // CHECK-NEXT: aaa2.x() = 1;
  volatile int2 aaa2;
  aaa2.x = 1;
  // CHECK: /*
  // CHECK-NEXT: DPCT1052:{{[0-9]+}}: DPC++ does not support the member access for volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: static sycl::int3 aaa3;
  // CHECK-NEXT: aaa3.x() = 1;
  static volatile int3 aaa3;
  aaa3.x = 1;
  // CHECK: /*
  // CHECK-NEXT: DPCT1052:{{[0-9]+}}: DPC++ does not support the member access for volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::int4 aaa4;
  volatile int4 aaa4;
}
