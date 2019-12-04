// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -extra-arg="-I  %S/inc"   -- -x cuda --cuda-host-only
// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -extra-arg="-I %S/inc"   -- -x cuda --cuda-host-only
// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -extra-arg="-I%S/inc"   -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/test_inc_included.dp.cpp

// CHECK:#include "foo.dp.hpp"
#include "foo.cuh"

void test(){
 foo<<<1,1>>>();
}
