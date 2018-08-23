// RUN: cu2sycl -out-root %T %s -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/replace-dim3.sycl.cpp --match-full-lines %s

// CHECK: void func(cl::sycl::range<3> a, cl::sycl::range<3> b, cl::sycl::range<3> c, cl::sycl::range<3> d) try {
void func(dim3 a, dim3 b, dim3 c, dim3 d) {
}

int main() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::range<3> deflt;
  dim3 deflt;

  // CHECK: cl::sycl::range<3> round1(1, 1, 1);
  dim3 round1(1);

  // CHECK: cl::sycl::range<3> round2(2, 1, 1);
  dim3 round2(2, 1);

  // CHECK: cl::sycl::range<3> assign = cl::sycl::range<3>(32, 1, 1);
  dim3 assign = 32;

  // CHECK: cl::sycl::range<3> castini = cl::sycl::range<3>(4, 1, 1);
  dim3 castini = (dim3)4;

  // CHECK: cl::sycl::range<3> castini2 = cl::sycl::range<3>(2, 2, 1);
  dim3 castini2 = dim3(2, 2);

  // CHECK: cl::sycl::range<3> castini3 = cl::sycl::range<3>(3, 1, 10);
  dim3 castini3 = dim3(3, 1, 10);

  // CHECK: deflt = cl::sycl::range<3>(3, 1, 1);
  deflt = dim3(3);

  // CHECK: cl::sycl::range<3> copyctor1 = cl::sycl::range<3>(cl::sycl::range<3>(33, 1, 1));
  dim3 copyctor1 = dim3((dim3)33);

  // CHECK: cl::sycl::range<3> copyctor2 = cl::sycl::range<3>(copyctor1);
  dim3 copyctor2 = dim3(copyctor1);

  // CHECK: cl::sycl::range<3> copyctor3(copyctor1);
  dim3 copyctor3(copyctor1);

  // CHECK: func(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(2, 1, 1), cl::sycl::range<3>(3, 2, 1));
  func((dim3)1, dim3(1), dim3(2, 1), dim3(3, 2, 1));
  // CHECK: func(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(2, 1, 1), cl::sycl::range<3>(3, 1, 1), cl::sycl::range<3>(4, 1, 1));
  func(1, 2, 3, 4);
  // CHECK: func(deflt, cl::sycl::range<3>(deflt), cl::sycl::range<3>(deflt), cl::sycl::range<3>(2+3*3, 1, 1));
  func(deflt, dim3(deflt), (dim3)deflt, 2+3*3);
}
