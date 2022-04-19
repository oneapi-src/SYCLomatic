// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <c2s/c2s.hpp>
// CHECK-NEXT: #include <math.h>
#include <math.h>

// CHECK: #ifdef CUDA
// CHECK-NEXT: void bar5() {
// CHECK-NEXT: }
#ifdef CUDA
#include <cuda.h>
__global__ void bar5() {
}
#endif
