// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/replace-dim3.sycl.cpp --match-full-lines %s

// CHECK: void func(cl::sycl::range<3> a, cl::sycl::range<3> b, cl::sycl::range<3> c, cl::sycl::range<3> d) try {
void func(dim3 a, dim3 b, dim3 c, dim3 d) {
}

// CHECK: void test(const cl::sycl::range<3> & a, const cl::sycl::range<3> & b) try {
void test(const dim3& a, const dim3& b) {
}

// CHECK: void test(cl::sycl::range<3> && a, cl::sycl::range<3> && b) try {
void test(dim3&& a, dim3&& b) {
}

// CHECK: void test(const cl::sycl::range<3> * a, const cl::sycl::range<3> * b) try {
void test(const dim3* a, const dim3* b) {
}

// CHECK: void test(const cl::sycl::range<3> ** a, const cl::sycl::range<3> ** b) try {
void test(const dim3** a, const dim3** b) {
}

__global__ void kernel(int dim) {}

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
  // CHECK: func(deflt, cl::sycl::range<3>(deflt), cl::sycl::range<3>(deflt), cl::sycl::range<3>(2 + 3 * 3, 1, 1));
  func(deflt, dim3(deflt), (dim3)deflt, 2 + 3 * 3);

  // CHECK: cl::sycl::range<3> test(1, 2, 3);
  dim3 test(1, 2, 3);

  // CHECK: int b = test[0] + test[1] + test [2];
  int b = test.x + test. y + test .z;
  // CHECK: cl::sycl::range<3> *p = &test;
  dim3 *p = &test;
  // CHECK: cl::sycl::range<3> **pp = &p;
  dim3 **pp = &p;

  // CHECK: int a = p->operator[](0) + p->operator[](1) + p->operator[](2);
  int a = p->x + p->y + p->z;
  // CHECK: int aa = (*pp)->operator[](0) + (*pp)->operator[](1) + (*pp)->operator[](2);
  int aa = (*pp)->x + (*pp)->y + (*pp)->z;

  struct  container
  {
    unsigned int x, y, z;
    // CHECK: cl::sycl::range<3> w;
    dim3 w;
    // CHECK: cl::sycl::range<3> *pw;
    dim3 *pw;
    // CHECK: cl::sycl::range<3> **ppw;
    dim3 **ppw;
  };
  typedef  struct container container;

  container t;

  // CHECK: int c = t.w[0] + t.w[1] + t.w[2];
  int c = t.w.x + t.w.y + t.w.z;
  // Note: The checks below should work.  Fix is needed (CTST-301 filed)
  // NO-CHECK: int c2 = t.pw->operator[](0) + t.pw->operator[](1) + t.pw->operator[](2);
  int c2 = t.pw->x + t.pw->y + t.pw->z;
  // NO-CHECK: int c3 = (*t.ppw)->operator[](0) + (*t.ppw)->operator[](1) + (*t.ppw)->operator[](2);
  int c3 = (*t.ppw)->x + (*t.ppw)->y + (*t.ppw)->z;

  // CHECK: cl::sycl::range<3> d3_1(test[0], 1, 1);
  dim3 d3_1(test.x);
  // CHECK: cl::sycl::range<3> d3_2(test[0] + 1, 1, 1);
  dim3 d3_2(test.x + 1);
  // CHECK: cl::sycl::range<3> d3_3(2 + test[0] + 1, 1, 1);
  dim3 d3_3(2 + test.x + 1);
  // CHECK: cl::sycl::range<3> d3_4(test[0], test[1], 1);
  dim3 d3_4(test.x, test.y);
  // CHECK: cl::sycl::range<3> d3_5(test[0], test[1], test[2]);
  dim3 d3_5(test.x, test.y, test.z);
  // CHECK: cl::sycl::range<3> d3_6 = cl::sycl::range<3>(test[0] + 1, 2 + test[1], 3 + test[2] + 4);
  dim3 d3_6 = dim3(test.x + 1, 2 + test.y, 3 + test.z + 4);

  // CHECK: {
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
  // CHECK:           kernel(d3_6[0]);
  // CHECK:         });
  // CHECK:     });
  // CHECK: };
  kernel<<<1, 1>>>(d3_6.x);
}
