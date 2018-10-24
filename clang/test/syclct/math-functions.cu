// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/math-functions.sycl.cpp --match-full-lines %s

#include <algorithm>

int main() {
  float a = 1.0f, b = 2.0f, c = 0;
  
  //CHECK: c = cl::sycl::max(a, b);
  c = max(a, b);
  
  //CHECK: c = std::max(a, b);
  c = std::max(a, b);
}
