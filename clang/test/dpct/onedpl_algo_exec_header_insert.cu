// RUN: dpct --format-range=none -out-root %T/onedpl_algo_exec_header_insert %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: FileCheck --input-file %T/onedpl_algo_exec_header_insert/onedpl_algo_exec_header_insert.dp.cpp %s
// RUN: FileCheck --input-file %T/onedpl_algo_exec_header_insert/onedpl_algo_exec_header_insert.h %S/onedpl_algo_exec_header_insert.h

// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include "onedpl_algo_exec_header_insert.h"
#include "onedpl_algo_exec_header_insert.h"

int main() {
  int *p = NULL;
  cudaMalloc((void **)&p, sizeof(int));
  return 0;
}
