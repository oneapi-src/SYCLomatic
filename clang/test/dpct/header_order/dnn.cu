// RUN: echo
// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include "dnn.h"
// CHECK-NEXT: #include <dpct/dnnl_utils.hpp>
#include "dnn.h"

int test(cudnnHandle_t handle) {
  cudnnCreate(&handle);
  return 0;
}
