// RUN: dpct -out-root %T/types005 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck %s --match-full-lines --input-file %T/types005/types005.dp.cpp

//CHECK:#include <sycl/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include <dpct/fft_utils.hpp>
//CHECK-NEXT:#include <dpct/lib_common_utils.hpp>
#include "cufft.h"

void foo() {
  //CHECK:auto ver = dpct::version_field::major;
  auto ver = MAJOR_VERSION;
}
