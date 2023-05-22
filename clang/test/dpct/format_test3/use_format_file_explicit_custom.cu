// RUN: cd %T
// RUN: cat %s > %T/use_format_file_explicit_custom.cu
// RUN: echo "BasedOnStyle: LLVM" > %T/.clang-format
// RUN: echo "ColumnLimit: 50" > %T/.clang-format
// RUN: dpct use_format_file_explicit_custom.cu --out-root=%T --cuda-include-path="%cuda-path/include" --format-style=custom -- --cuda-host-only
// RUN: FileCheck -strict-whitespace %s --match-full-lines --input-file %T/use_format_file_explicit_custom.dp.cpp
#include "cuda.h"

void bar();
#define SIZE 100

size_t size = 1234567 * sizeof(float);
float *h_A = (float *)malloc(size);
float *d_A = NULL;

     //CHECK:void foo1() try {
//CHECK-NEXT:  for(;;)
//CHECK-NEXT:    int a = CHECK_SYCL_ERROR(
//CHECK-NEXT:        dpct::get_default_queue()
//CHECK-NEXT:            .memcpy(d_A, h_A,
//CHECK-NEXT:                    sizeof(double) * SIZE * SIZE)
//CHECK-NEXT:            .wait());
//CHECK-NEXT:}
//CHECK-NEXT:catch (sycl::exception const &exc) {
//CHECK-NEXT:  std::cerr << exc.what()
//CHECK-NEXT:            << "Exception caught at file:"
//CHECK-NEXT:            << __FILE__ << ", line:" << __LINE__
//CHECK-NEXT:            << std::endl;
//CHECK-NEXT:  std::exit(1);
//CHECK-NEXT:}
void foo1() {
  for(;;)
    int a = cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
}