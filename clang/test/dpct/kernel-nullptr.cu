// RUN: dpct --format-range=none --usm-level=none --out-root %T/kernel-nullptr %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14

// RUN: FileCheck --input-file %T/kernel-nullptr/kernel-nullptr.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/kernel-nullptr/kernel-nullptr.dp.cpp -o %T/kernel-nullptr/kernel-nullptr.dp.o %}

#ifndef NO_BUILD_TEST
__global__ void kernel(int *a) {}

int main() {
    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel(0);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>(0);

    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel(nullptr);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>(nullptr);

    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel(NULL);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>(NULL);

    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel((int *)0);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>((int *)0);

    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel((int *)nullptr);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>((int *)nullptr);

    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel((int *)NULL);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>((int *)NULL);
}
#endif
