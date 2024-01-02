// RUN: dpct --format-range=none --out-root %T %s --cuda-include-path="%cuda-path/include" --in-root %S --extra-arg="-I %S/../header2 ,-I %S/../header "
// RUN: FileCheck --input-file %T/test.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST  %T/test.dp.cpp -o %T/test.dp.o %}

#ifndef BUILD_TEST
#include "../header/in_header.h"
#include "../header2/out_header.h"
int main() {
  // CHECK: sycl::device dev_ct1;
  // CHECK-NEXT: sycl::queue q_ct1(dev_ct1, sycl::property_list{sycl::property::queue::in_order()});

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT: sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT: [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:   helloFromGPU();
  // CHECK-NEXT: });
  helloFromGPU<<<1, 1>>>();
  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT: sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT: [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:   helloFromGPU2();
  // CHECK-NEXT: });
  helloFromGPU2<<<1, 1>>>();

  return 0;
}
#endif
