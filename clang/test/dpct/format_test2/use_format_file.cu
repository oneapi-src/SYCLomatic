// RUN: cd %T
// RUN: cat %s > %T/use_format_file.cu
// RUN: echo "ColumnLimit: 50" > %T/.clang-format
// RUN: echo "TabWidth: 4" >> %T/.clang-format
// RUN: dpct use_format_file.cu --out-root=%T --cuda-include-path="%cuda-path/include" -- --cuda-host-only
// RUN: FileCheck -strict-whitespace %s --match-full-lines --input-file %T/use_format_file.dp.cpp
#include "cuda.h"

void bar();
#define SIZE 100

size_t size = 1234567 * sizeof(float);
float *h_A = (float *)malloc(size);
float *d_A = NULL;

     //CHECK:void foo1() try {
//CHECK-NEXT:  for(;;)
//CHECK-NEXT:    /*
//CHECK-NEXT:    DPCT1003:0: Migrated API does not return error
//CHECK-NEXT:    code. (*, 0) is inserted. You may need to
//CHECK-NEXT:    rewrite this code.
//CHECK-NEXT:    */
//CHECK-NEXT:    int a =
//CHECK-NEXT:        (dpct::get_default_queue()
//CHECK-NEXT:             .memcpy(d_A, h_A,
//CHECK-NEXT:                     sizeof(double) * SIZE * SIZE)
//CHECK-NEXT:             .wait(),
//CHECK-NEXT:         0);
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

     //CHECK:void foo2() {
//CHECK-NEXT:        sycl::int4 n;
//CHECK-NEXT:        sycl::int4 m;
//CHECK-NEXT:        n.x() = m.w();
//CHECK-NEXT:        n.y() = m.z();
//CHECK-NEXT:        n.z() = m.y();
//CHECK-NEXT:        n.w() = m.x();
//CHECK-NEXT:}
__global__ void foo2() {
		int4 n;
		int4 m;
		n.x = m.w;
		n.y = m.z;
		n.z = m.y;
		n.w = m.x;
}