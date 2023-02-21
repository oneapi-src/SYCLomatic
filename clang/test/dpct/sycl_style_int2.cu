// RUN: dpct --format-range=none --usm-level=none -out-root %T/sycl_style_int2 %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/sycl_style_int2/sycl_style_int2.dp.cpp --match-full-lines %s
#ifdef _WIN32
#include <cstdint>
#endif
// CHECK: void func3(sycl::mint2 a, sycl::mint2 b, sycl::mint2 c) {
void func3(int2 a, int2 b, int2 c) {
}
// CHECK: void fun(sycl::mint2 a) {}
void fun(int2 a) {}

// CHECK: void kernel(sycl::mint2* data) {
__global__ void kernel(int2* data) {
}

int main() {
  // range default constructor does the right thing.
  // CHECK: sycl::mint2 deflt;
  int2 deflt;

  // CHECK: sycl::mint2 copyctor1 = sycl::mint2(1, 2);
  int2 copyctor1 = make_int2(1, 2);

  // CHECK: sycl::mint2 copyctor2 = sycl::mint2(copyctor1);
  int2 copyctor2 = int2(copyctor1);

  // CHECK: sycl::mint2 copyctor3(copyctor1);
  int2 copyctor3(copyctor1);

  // CHECK: func3(deflt, sycl::mint2(deflt), (sycl::mint2)deflt);
  func3(deflt, int2(deflt), (int2)deflt);

  // CHECK: sycl::mint2 *i4;
  int2 *i4;
  // CHECK: sycl::mint2 *i5;
  int2 *i5;
  // CHECK: sycl::mint2 i6;
  int2 i6;
  // CHECK: sycl::mint2 i7;
  int2 i7;
  // CHECK: int i = i6[0];
  int i = i6.x;
  // CHECK: i6[0] = i7[0];
  i6.x = i7.x;
  // CHECK: if (i6[0] == i7[0]) {
  if (i6.x == i7.x) {
  }
  // CHECK: sycl::mint2 i2_array[10];
  int2 i2_array[10];
  // CHECK: sycl::mint2 i2_array2[10];
  int2 i2_array2[10];
  // CHECK: if (i2_array[1][0] == i2_array2[1][0]) {
  if (i2_array[1].x == i2_array2[1].x) {
  }
  // CHECK: sycl::mint2 x = sycl::mint2(1, 2);
  int2 x = make_int2(1, 2);
  // CHECK: i4 = (sycl::mint2 *)i2_array;
  i4 = (int2 *)i2_array;
  // CHECK: i7 = (sycl::mint2)i6;
  i7 = (int2)i6;
  // CHECK: i7 = sycl::mint2(i6);
  i7 = int2(i6);

  // CHECK: struct benchtype_s {
  // CHECK:   uint64_t u64;
  // CHECK:   sycl::muint2 u32;
  // CHECK:   sycl::muint2 *u32_p ;
  // CHECK:   sycl::muint2 array[10];
  // CHECK: };
  struct benchtype_s {
    uint64_t u64;
    uint2 u32;
    uint2 *u32_p ;
    uint2 array[10];
  };

  // CHECK: union benchtype_u {
  // CHECK:   uint64_t u64;
  // CHECK:   sycl::muint2 u32{};
  // CHECK:   sycl::muint2 *u32_p ;
  // CHECK:   sycl::muint2 array[10];
  // CHECK: };
  union benchtype_u {
    uint64_t u64;
    uint2 u32;
    uint2 *u32_p ;
    uint2 array[10];
  };

  // CHECK: sycl::mint2* data;
  // CHECK-NEXT: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::mint2 *> data_acc_ct0(data, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel(data_acc_ct0.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  int2* data;
  kernel<<<1, 1>>>(data);

  // CHECK: volatile int aaa1;
  // CHECK-NEXT: aaa1 = 1;
  volatile int1 aaa1;
  aaa1.x = 1;
  // CHECK: /*
  // CHECK-NEXT: DPCT1052:{{[0-9]+}}: SYCL does not support the member access for a volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::mint2 aaa2;
  // CHECK-NEXT: aaa2[0] = 1;
  volatile int2 aaa2;
  aaa2.x = 1;
  // CHECK: /*
  // CHECK-NEXT: DPCT1052:{{[0-9]+}}: SYCL does not support the member access for a volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: static sycl::mint3 aaa3;
  // CHECK-NEXT: aaa3[0] = 1;
  static volatile int3 aaa3;
  aaa3.x = 1;
  // CHECK: /*
  // CHECK-NEXT: DPCT1052:{{[0-9]+}}: SYCL does not support the member access for a volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::mint4 aaa4;
  volatile int4 aaa4;

  // CHECK: volatile int *pv1;
  volatile int1 *pv1;

  // CHECK: /*
  // CHECK-NEXT: DPCT1052:{{[0-9]+}}: SYCL does not support the member access for a volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::int2 *pv2;
  volatile int2 *pv2;

    // CHECK: /*
  // CHECK-NEXT: DPCT1052:{{[0-9]+}}: SYCL does not support the member access for a volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::int3 *pv3;
  volatile int3 *pv3;

    // CHECK: /*
  // CHECK-NEXT: DPCT1052:{{[0-9]+}}: SYCL does not support the member access for a volatile qualified vector type. The volatile qualifier was removed. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::int4 *pv4;
  volatile int4 *pv4;
}

