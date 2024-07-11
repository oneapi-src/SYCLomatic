// RUN: echo "empty command"
// CHECK: #define MACRO_B
// CHECK-NOT: #include <sycl/sycl.hpp>
// CHECK-NOT: #include <dpct/dpct.hpp>
#define MACRO_B
#include "test.h"

void test() {
// CHECK: int *host_a = (int *)malloc(100 * sizeof(int));
  int *host_a = malloc(100 * sizeof(int));
}
