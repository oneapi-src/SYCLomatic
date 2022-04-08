// RUN: c2s -out-root %T/types005 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck %s --match-full-lines --input-file %T/types005/types005.dp.cpp

//CHECK:#include <CL/sycl.hpp>
//CHECK-NEXT:#include <c2s/c2s.hpp>
//CHECK-NEXT:#include <oneapi/mkl.hpp>
//CHECK-NEXT:#include <c2s/lib_common_utils.hpp>
#include "cufft.h"

void foo() {
  //CHECK:auto ver = c2s::version_field::major;
  auto ver = MAJOR_VERSION;
}
