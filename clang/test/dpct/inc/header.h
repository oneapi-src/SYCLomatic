// RUN: FileCheck --match-full-lines --input-file %T/header.h %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <math.h>
#include <math.h>

// CHECK: #ifdef CUDA
// CHECK-NEXT: void bar() {
// CHECK-NEXT: }
#ifdef CUDA
#include <cuda.h>
__global__ void bar() {
}
#elif defined(OPENMP)
void bar() {
}
#else
void bar() {
}
#endif
