// FIXME
// RUN: c2s --usm-level=none -out-root %T/ctad %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda --format-range=none --enable-ctad -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/ctad/ctad.dp.cpp --match-full-lines %s

#include <cstdio>
#include <algorithm>

#define NUM 23

// CHECK: void func(sycl::range<3> a, sycl::range<3> b, sycl::range<3> c, sycl::range<3> d) {
void func(dim3 a, dim3 b, dim3 c, dim3 d) {
}

__global__ void kernel(int dim) {
  __shared__ int k[32];
}

int main() {
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // range default constructor does the right thing.
  // CHECK: sycl::range deflt(1, 1, 1);
  dim3 deflt;

  // CHECK:  sycl::range deflt_1;
  // CHECK-NEXT: sycl::id deflt_2;
  cudaExtent deflt_1;
  cudaPos deflt_2;

  // CHECK: sycl::range round1_1(1, 1, NUM);
  dim3 round1_1(NUM);

  cudaExtent exten = make_cudaExtent(1,1,1);;

  // CHECK: sycl::range castini = sycl::range(1, 1, 4);
  dim3 castini = (dim3)4;

  // CHECK:   sycl::range castini_1 = exten;
  // CHECK-NEXT:sycl::id castini_2 = deflt_2;
  cudaExtent castini_1 = exten;
  cudaPos castini_2 = deflt_2;

  // CHECK: sycl::range copyctor1 = sycl::range(sycl::range(1, 1, 33));
  dim3 copyctor1 = dim3((dim3)33);


  // CHECK: sycl::range copyctor2 = sycl::range(copyctor1);
  dim3 copyctor2 = dim3(copyctor1);

  // CHECK: sycl::range copyctor3(copyctor1);
  dim3 copyctor3(copyctor1);

  // CHECK: sycl::range copyctor31(exten);
  // CHECK-NEXT: sycl::id copyctor32(deflt_2);
  cudaExtent copyctor31(exten);
  cudaPos copyctor32(deflt_2);

  // CHECK: func(sycl::range(1, 1, 1), sycl::range(1, 1, 1), sycl::range(1, 1, 2), sycl::range(1, 2, 3));
  func((dim3)1, dim3(1), dim3(2, 1), dim3(3, 2, 1));
  // CHECK: func(deflt, sycl::range(deflt), sycl::range(deflt), sycl::range(1, 1, 2 + 3 * 3));
  func(deflt, dim3(deflt), (dim3)deflt, 2 + 3 * 3);


  // CHECK: sycl::range<3> *p = &deflt;
  dim3 *p = &deflt;
  // CHECK: sycl::range<3> **pp = &p;
  dim3 **pp = &p;

  // CHECK: sycl::range<3> *p_1 = &deflt_1;
  // CHECK-NEXT: sycl::id<3> *p_2 = &deflt_2;
  cudaExtent *p_1 = &deflt_1;
  cudaPos *p_2 = &deflt_2;

  struct  container
  {
    unsigned int x, y, z;
    // CHECK: sycl::range<3> w;
    dim3 w;
    // CHECK: sycl::range<3> *pw;
    dim3 *pw;
    // CHECK: sycl::range<3> **ppw;
    dim3 **ppw;
  };

  // CHECK: sycl::range gpu_blocks(1, 1, 1 / (castini[2] * 200));
  dim3 gpu_blocks(1 / (castini.x * 200));
  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> k_acc_ct1(sycl::range(32), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<c2s_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernel(1, k_acc_ct1.get_pointer());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernel<<<1, 1>>>(1);
  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> k_acc_ct1(sycl::range(32), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<c2s_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range(sycl::range(1, 1, NUM) * sycl::range(1, 1, NUM), sycl::range(1, 1, NUM)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernel(1, k_acc_ct1.get_pointer());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernel<<<NUM, NUM>>>(1);
}


