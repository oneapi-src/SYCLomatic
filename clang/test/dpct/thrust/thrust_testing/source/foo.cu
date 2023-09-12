// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/thrust/thrust_testing %s --cuda-include-path="%cuda-path/include" -extra-arg-before="-I%S" -- -x cuda --cuda-host-only -std=c++17
// RUN: FileCheck --input-file %T//thrust/thrust_testing/foo.dp.cpp --match-full-lines %s

#include <algorithm>
#include <thrust/complex.h> // here complex.h is user defined head file
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "../foo.h"

int main() {

  // CHECK: sycl::range<3> t(1, 1, 1);
  dim3 t;
  return 0;
}
