// RUN: cp %S/* .
// RUN: dpct -p=%S --format-range=none -out-root %T/c_source_file_test --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %S/test.h --match-full-lines --input-file %T/c_source_file_test/test.h
// RUN: FileCheck %S/test1.c --match-full-lines --input-file %T/c_source_file_test/test1.c
// RUN: FileCheck %S/main.cu --match-full-lines --input-file %T/c_source_file_test/main.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/c_source_file_test/main.dp.cpp -o %T/c_source_file_test/main.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
// CHECK: #include "test.h"
// CHECK: #include "test2.c"
// CHECK: #include "test3.c.dp.cpp"

#include "test.h"
#include <cuda_runtime.h>

#include "test2.c"
#include "test3.c"

int main() {
  int *dev_a;
// CHECK:   dev_a = sycl::malloc_device<int>(100, dpct::get_in_order_queue());
  cudaMalloc(&dev_a, 100 * sizeof(int));
  return 0;
}
