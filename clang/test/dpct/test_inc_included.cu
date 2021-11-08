// RUN: dpct --format-range=none --usm-level=none -out-root %T/test_inc_included %s --cuda-include-path="%cuda-path/include" -extra-arg="-I  %S/inc"   -- -x cuda --cuda-host-only
// RUN: dpct --format-range=none --usm-level=none -out-root %T/test_inc_included %s --cuda-include-path="%cuda-path/include" -extra-arg="-I %S/inc"   -- -x cuda --cuda-host-only
// RUN: dpct --format-range=none --usm-level=none -out-root %T/test_inc_included %s --cuda-include-path="%cuda-path/include" -extra-arg="-I%S/inc"   -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/test_inc_included/test_inc_included.dp.cpp

// CHECK:#include "foo.dp.hpp"
#include "foo.cuh"

// CHECK:#include "no_cuda_syntax.dp.hpp"
#include "no_cuda_syntax.cuh"

// CHECK:#include "no_cuda_syntax.dp.hpp"
#include <no_cuda_syntax.cuh>

void test(){
 foo<<<1,1>>>();
}
