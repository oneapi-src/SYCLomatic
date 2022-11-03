// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/iterator/constant_iterator %S/constant_iterator.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/iterator/constant_iterator/constant_iterator.dp.cpp %s

// CHECK: #include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>
#include <stdio.h>

// CHECK: dpct::constant_iterator<double> itr(5.0);
int main() {
  cub::ConstantInputIterator<double> itr(5.0);
  for (int i = 0; i< 100; ++i) 
    printf("%.2lf\n", itr[i]);
  return 0;
}
