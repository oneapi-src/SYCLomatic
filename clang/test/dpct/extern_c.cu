// RUN: dpct --format-range=none --usm-level=none -out-root %T/extern_c %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/extern_c/extern_c.dp.cpp --match-full-lines %s

//     CHECK:#ifdef __cplusplus
//CHECK-NEXT:#define DPCT_USM_LEVEL_NONE
//CHECK-NEXT:#include <sycl/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:extern "C" {
//CHECK-NEXT:#endif
//CHECK-NEXT:#include "extern_c.h"
//CHECK-NEXT:#ifdef __cplusplus
//CHECK-NEXT:}
//CHECK-NEXT:#endif


#ifdef __cplusplus
extern "C" {
#endif
#include "extern_c.h"
#ifdef __cplusplus
}
#endif

int main() {
  float2 f2;
  return 0;
}

