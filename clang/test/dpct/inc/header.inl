// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <math.h>
#include <math.h>

// CHECK: #ifdef CUDA
// CHECK-NEXT: void bar3() {
// CHECK-NEXT: }
#ifdef CUDA
#include <cuda.h>
__global__ void bar3() {
}
#endif
