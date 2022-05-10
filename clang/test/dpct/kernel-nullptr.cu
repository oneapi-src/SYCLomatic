// RUN: dpct --format-range=none --no-cl-namespace-inline --usm-level=none -out-root %T/kernel-nullptr %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14

// RUN: FileCheck --input-file %T/kernel-nullptr/kernel-nullptr.dp.cpp --match-full-lines %s


__global__ void kernel(int *a) {}

int main() {
    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel(0);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>(0);

    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel(nullptr);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>(nullptr);

    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel(NULL);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>(NULL);

    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel((int *)0);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>((int *)0);

    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel((int *)nullptr);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>((int *)nullptr);

    // CHECK: q_ct1.parallel_for(
    // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           kernel((int *)NULL);
    // CHECK-NEXT:         });
    kernel<<<1,1>>>((int *)NULL);
}

