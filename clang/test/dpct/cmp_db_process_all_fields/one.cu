// ------ prepare test directory
// RUN: cd %T
// RUN: cp %s one.cu
// RUN: dpct -p=. --out-root=dpct_output/ --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/out/dpct_output/one.dp.cpp %T/one.cu

// CHECK: void add() {
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