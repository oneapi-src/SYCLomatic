// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/sycl_style_int2.sycl.cpp --match-full-lines %s

// CHECK: void func3(cl::sycl::int2 a, cl::sycl::int2 b, cl::sycl::int2 c) try {
void func3(int2 a, int2 b, int2 c) {
}
// CHECK: void fun(cl::sycl::int2 a) try {}
void fun(int2 a) {}

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
  // CHECK: int i = i6.x();
  int i = i6.x;
  // CHECK: i6.x() = i7.x();
  i6.x = i7.x;
  // CHECK: if (i6.x() == i7.x()) {
  if (i6.x == i7.x) {
  }
  // CHECK: cl::sycl::int2 i2_array[10];
  int2 i2_array[10];
  // CHECK: cl::sycl::int2 i2_array2[10];
  int2 i2_array2[10];
  // CHECK: if (i2_array[1].x() == i2_array2[1].x()) {
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
}
