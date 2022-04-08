// RUN: echo "empty command"

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <c2s/c2s.hpp>
// CHECK-NEXT: #include <math.h>
#include <math.h>

__global__ void header3() {
}
