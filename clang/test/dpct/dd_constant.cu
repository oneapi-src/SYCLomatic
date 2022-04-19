// RUN: cd %T
// RUN: mkdir dd_constant
// RUN: cd dd_constant
// RUN: cat %s > dd_constant.cu
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: dpct dd_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: dpct dd_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/dd_constant/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./dd_constant

// CHECK: static dpct::constant_memory<const float, 0> aaa((float)(1ll << 40));
// CHECK-NEXT: static dpct::constant_memory<const float, 0> bbb((float)(1ll << 20));

#include "constant_header.h"
