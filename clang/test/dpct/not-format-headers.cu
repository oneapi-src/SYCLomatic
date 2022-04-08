// RUN: c2s --out-root=%T/not-format-headers %s --cuda-include-path="%cuda-path/include" --format-range=all -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace %s --match-full-lines --input-file %T/not-format-headers/not-format-headers.dp.cpp

//CHECK:#include <CL/sycl.hpp>
//CHECK-NEXT:#include <c2s/c2s.hpp>
//CHECK-NEXT:#include "test-header.h"
#include "cuda.h"
#include "test-header.h"

int main(){
  return 0;
}
