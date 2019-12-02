// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/sycl_style_int2.dp.cpp --match-full-lines %s
#ifdef _WIN32
#include <cstdint>
#endif
// CHECK: void func3(cl::sycl::int2 a, cl::sycl::int2 b, cl::sycl::int2 c) {
void func3(int2 a, int2 b, int2 c) {
}
// CHECK: void fun(cl::sycl::int2 a) {}
void fun(int2 a) {}

// CHECK: void kernel(cl::sycl::int2* data) {}
__global__ void kernel(int2* data) {}

int main() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::int2 deflt;
  int2 deflt;

  // CHECK: cl::sycl::int2 copyctor1 = cl::sycl::int2(1, 2);
  int2 copyctor1 = make_int2(1, 2);

  // CHECK: cl::sycl::int2 copyctor2 = cl::sycl::int2(copyctor1);
  int2 copyctor2 = int2(copyctor1);

  // CHECK: cl::sycl::int2 copyctor3(copyctor1);
  int2 copyctor3(copyctor1);

  // CHECK: func3(deflt, cl::sycl::int2(deflt), (cl::sycl::int2)deflt);
  func3(deflt, int2(deflt), (int2)deflt);

  // CHECK: cl::sycl::int2 *i4;
  int2 *i4;
  // CHECK: cl::sycl::int2 *i5;
  int2 *i5;
  // CHECK: cl::sycl::int2 i6;
  int2 i6;
  // CHECK: cl::sycl::int2 i7;
  int2 i7;
  // CHECK: int i = static_cast<int>(i6.x());
  int i = i6.x;
  // CHECK: i6.x() = static_cast<int>(i7.x());
  i6.x = i7.x;
  // CHECK: if (static_cast<int>(i6.x()) == static_cast<int>(i7.x())) {
  if (i6.x == i7.x) {
  }
  // CHECK: cl::sycl::int2 i2_array[10];
  int2 i2_array[10];
  // CHECK: cl::sycl::int2 i2_array2[10];
  int2 i2_array2[10];
  // CHECK: if (static_cast<int>(i2_array[1].x()) == static_cast<int>(i2_array2[1].x())) {
  if (i2_array[1].x == i2_array2[1].x) {
  }
  // CHECK: cl::sycl::int2 x = cl::sycl::int2(1, 2);
  int2 x = make_int2(1, 2);
  // CHECK: i4 = (cl::sycl::int2 *)i2_array;
  i4 = (int2 *)i2_array;
  // CHECK: i7 = (cl::sycl::int2)i6;
  i7 = (int2)i6;
  // CHECK: i7 = cl::sycl::int2(i6);
  i7 = int2(i6);

  // CHECK: struct benchtype_s {
  // CHECK:   uint64_t u64;
  // CHECK:   cl::sycl::uint2 u32;
  // CHECK:   cl::sycl::uint2 *u32_p ;
  // CHECK:   cl::sycl::uint2 array[10];
  // CHECK: };
  struct benchtype_s {
    uint64_t u64;
    uint2 u32;
    uint2 *u32_p ;
    uint2 array[10];
  };

  // CHECK: union benchtype_u {
  // CHECK:   uint64_t u64;
  // CHECK:   cl::sycl::uint2 u3{};
  // CHECK:   cl::sycl::uint2 *u32_p ;
  // CHECK:   cl::sycl::uint2 array[10];
  // CHECK: };
  union benchtype_u {
    uint64_t u64;
    uint2 u32;
    uint2 *u32_p ;
    uint2 array[10];
  };

  // CHECK: cl::sycl::int2* data;
  // CHECK-NEXT: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> data_buf_ct0 = dpct::get_buffer_and_offset(data);
  // CHECK-NEXT:   size_t data_offset_ct0 = data_buf_ct0.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto data_acc_ct0 = data_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::int2 *data_ct0 = (cl::sycl::int2 *)(&data_acc_ct0[0] + data_offset_ct0);
  // CHECK-NEXT:           kernel(data_ct0);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  int2* data;
  kernel<<<1, 1>>>(data);
}
