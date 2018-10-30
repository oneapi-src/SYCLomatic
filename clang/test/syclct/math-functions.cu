// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/math-functions.sycl.cpp --match-full-lines %s

#include <algorithm>

int main() {
  float f_a = 1.0f, f_b = 2.0f, f_c = 0;
  double d_a = 1.0, d_b = 2.0, d_c = 0;
  unsigned u_a = 1, u_b = 2, u_c = 0;
  int i_a = 1, i_b = 2, i_c = 0;
  
  //CHECK: f_c = cl::sycl::max(f_a, f_b);
  f_c = max(f_a, f_b);

  //CHECK: d_c = cl::sycl::max(d_a, d_b);
  d_c = max(d_a, d_b);

  //CHECK: u_c = cl::sycl::max(u_a, u_b);
  u_c = max(u_a, u_b);

  //CHECK: i_c = cl::sycl::max(i_a, i_b);
  i_c = max(i_a, i_b);
  
  //CHECK: i_c = std::max(i_a, i_b);
  i_c = std::max(i_a, i_b);
}
