// FIXME
// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda --format-range=none --enable-ctad -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/ctad.dp.cpp --match-full-lines %s

#include <cstdio>
#include <algorithm>

#define NUM 23

// CHECK: void func(cl::sycl::range<3> a, cl::sycl::range<3> b, cl::sycl::range<3> c, cl::sycl::range<3> d) {
void func(dim3 a, dim3 b, dim3 c, dim3 d) {
}

__global__ void kernel(int dim) {
  __shared__ int k[32];
}

int main() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::range deflt(1, 1, 1);
  dim3 deflt;

  // CHECK: cl::sycl::range round1_1(NUM, 1, 1);
  dim3 round1_1(NUM);

  // CHECK: cl::sycl::range castini = cl::sycl::range(4, 1, 1);
  dim3 castini = (dim3)4;

  // CHECK: cl::sycl::range copyctor1 = cl::sycl::range(cl::sycl::range(33, 1, 1));
  dim3 copyctor1 = dim3((dim3)33);

  // CHECK: cl::sycl::range copyctor2 = cl::sycl::range(copyctor1);
  dim3 copyctor2 = dim3(copyctor1);

  // CHECK: cl::sycl::range copyctor3(copyctor1);
  dim3 copyctor3(copyctor1);

  // CHECK: func(cl::sycl::range(1, 1, 1), cl::sycl::range(1, 1, 1), cl::sycl::range(2, 1, 1), cl::sycl::range(3, 2, 1));
  func((dim3)1, dim3(1), dim3(2, 1), dim3(3, 2, 1));
  // CHECK: func(deflt, cl::sycl::range(deflt), cl::sycl::range(deflt), cl::sycl::range(2 + 3 * 3, 1, 1));
  func(deflt, dim3(deflt), (dim3)deflt, 2 + 3 * 3);


  // CHECK: cl::sycl::range<3> *p = &deflt;
  dim3 *p = &deflt;
  // CHECK: cl::sycl::range<3> **pp = &p;
  dim3 **pp = &p;

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

  // CHECK: cl::sycl::range gpu_blocks(1 / (castini[0] * 200), 1, 1);
  dim3 gpu_blocks(1 / (castini.x * 200));
  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> k_acc_ct1(cl::sycl::range(32/*32*/), cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range(cl::sycl::range(1, 1, 1) * cl::sycl::range(1, 1, 1), cl::sycl::range(1, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernel(1, k_acc_ct1.get_pointer());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernel<<<1, 1>>>(1);
  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> k_acc_ct1(cl::sycl::range(32/*32*/), cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range(cl::sycl::range(1, 1, NUM) * cl::sycl::range(1, 1, NUM), cl::sycl::range(1, 1, NUM)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernel(1, k_acc_ct1.get_pointer());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernel<<<NUM, NUM>>>(1);
}

