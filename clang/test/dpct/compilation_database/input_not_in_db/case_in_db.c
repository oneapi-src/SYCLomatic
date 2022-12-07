// RUN: echo 0
// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>


// CHECK: dpct::constant_memory<float, 1> const_angle(360);
// CHECK-NEXT: void simple_kernel(float *d_array, float *const_angle) {
// CHECK-NEXT:  d_array[0] = const_angle[0];
// CHECK-NEXT:  return;
// CHECK-NEXT: }
#ifdef TEST
__constant__ float const_angle[360];
__global__ void simple_kernel(float *d_array) {
  d_array[0] = const_angle[0];
  return;
}
#else
__constant__ float const_angle[230];
#endif
