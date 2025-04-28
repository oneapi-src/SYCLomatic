

template <int X>
//CHECK: inline int foo() {
__device__ inline int foo() {
  return threadIdx.x + X;
}