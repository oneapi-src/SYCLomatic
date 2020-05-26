// RUN: dpct --out-root=%T %s --cuda-include-path="%cuda-path/include" --format-range=all -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace %s --match-full-lines --input-file %T/not-format-headers.dp.cpp

//CHECK:#include <CL/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include "test-header.h"
#include "cuda.h"
#include "test-header.h"

int main(){
  return 0;
}
