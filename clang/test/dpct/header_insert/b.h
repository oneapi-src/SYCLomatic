
// CHECK: #ifndef CU_FILE
// CHECK-NEXT: #include <cstdio>
// CHECK-NEXT: #else
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <iostream>
// CHECK-NEXT: #endif
// CHECK-EMPTY:
// CHECK-NEXT: typedef struct dpct_type_{{[0-9]+}} {
// CHECK-NEXT:   unsigned i;
// CHECK-NEXT: } T1;
// CHECK-EMPTY:
// CHECK-NEXT: #ifdef CU_FILE
// CHECK-NEXT: sycl::float2 ff;
// CHECK-NEXT: #endif

// This test targets to make sure SYCL header files are inserted into
// right location (in the code section: CU_FILE is defined)
// to avoid build fail for code path without CU_FILE.

#ifndef CU_FILE
#include <cstdio>
#else
#include <iostream>
#endif

typedef struct {
  unsigned i;
} T1;

#ifdef CU_FILE
float2 ff;
#endif
