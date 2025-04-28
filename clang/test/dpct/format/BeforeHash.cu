// RUN: cd %S
// RUN: dpct --out-root %T %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/BeforeHash.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/BeforeHash.dp.cpp -o %T/BeforeHash.dp.o %}

#ifndef ABCD
#define ABCD 0
#endif

// CHECK: #define DPCT_PROFILING_ENABLED
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <cstdio>
#include <cstdio>

// CHECK: void foo() {
// CHECK-NEXT:   dpct::event_ptr start;
// CHECK-NEXT:   dpct::sync_barrier(start);
// CHECK-NEXT: }
void foo() {
  cudaEvent_t start;
  cudaEventRecord(start);
}
