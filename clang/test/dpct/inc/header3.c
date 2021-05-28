// RUN: echo "empty command"

// CHECK: #define DPCT_NAMED_LAMBDA
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <math.h>
#include <math.h>

__global__ void header3() {
}
