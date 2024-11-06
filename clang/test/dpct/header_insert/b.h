
// CHECK: #ifndef CU_FILE
// CHECK-NEXT: #include <cstdio>
// CHECK-NEXT: #else
// CHECK-NEXT: #include <iostream>
// CHECK-NEXT: #endif
// CHECK-EMPTY:
// CHECK-NEXT: typedef class dpct_type_798840 {
// CHECK-NEXT:     unsigned i;
// CHECK-NEXT: } T1;
// CHECK-EMPTY:
// CHECK-NEXT: #ifdef CU_FILE
// CHECK-NEXT: sycl::float2 ff;
// CHECK-NEXT: #endif

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
