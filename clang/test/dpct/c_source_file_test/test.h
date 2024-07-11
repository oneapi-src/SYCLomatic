// CHECK: #define MACRO_A
// CHECK-NOT: #include <sycl/sycl.hpp>
// CHECK-NOT: #include <dpct/dpct.hpp>
#define MACRO_A
#include <stdlib.h>
#include <stdio.h>

// CHECK: #if defined(__ARM_NEON) && defined(SYCL_LANGUAGE_VERSION)
// CHECK: typedef uint16_t fp16_t;
// CHECK: #endif
#if defined(__ARM_NEON) && defined(__CUDACC__)
typedef uint16_t fp16_t;
#endif

// CHECK: typedef struct dpct_type_{{[0-9a-z]+}} {
// CHECK:   char *type_name;
// CHECK:   size_t type_size;
// CHECK: } type_traits_t;
typedef struct {
  char *type_name;
  size_t type_size;
} type_traits_t;
