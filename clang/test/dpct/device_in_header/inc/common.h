

template <int X>
//CHECK: inline int foo(sycl::nd_item<3> item_ct1) {
__device__ inline int foo() {
  return threadIdx.x + X;
}