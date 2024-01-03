// RUN: echo 0
// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK: void add(){
__global__ void add(){
    return;
}

int main(){
    // CHECK: dpct::get_in_order_queue().parallel_for(
    // CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:                add();
    // CHECK-NEXT:        });
    add<<<1, 1>>>();
    return 0;
}