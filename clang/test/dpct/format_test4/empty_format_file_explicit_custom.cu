// RUN: cd %T
// RUN: cat %s > %T/empty_format_file_explicit_custom.cu
// RUN: echo "" > %T/.clang-format
// RUN: dpct empty_format_file_explicit_custom.cu --out-root=%T --cuda-include-path="%cuda-path/include" -- --cuda-host-only
// RUN: FileCheck -strict-whitespace %s --match-full-lines --input-file %T/empty_format_file_explicit_custom.dp.cpp
#include "cuda.h"

void bar();
#define SIZE 100

size_t size = 1234567 * sizeof(float);
float *h_A = (float *)malloc(size);
float *d_A = NULL;

     //CHECK:void foo1() try {
//CHECK-NEXT:  for(;;)
//CHECK-NEXT:    int a = DPCT_CHECK_ERROR(dpct::get_in_order_queue()
//CHECK-NEXT:                                 .memcpy(d_A, h_A, sizeof(double) * SIZE * SIZE)
//CHECK-NEXT:                                 .wait());
//CHECK-NEXT:}
//CHECK-NEXT:catch (sycl::exception const &exc) {
//CHECK-NEXT:  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
//CHECK-NEXT:            << ", line:" << __LINE__ << std::endl;
//CHECK-NEXT:  std::exit(1);
//CHECK-NEXT:}
void foo1() {
  for(;;)
    int a = cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
}