// RUN: cd %T
// RUN: mkdir d_constant
// RUN: cd d_constant
// RUN: cat %s > d_constant.cu
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: dpct d_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/d_constant/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./d_constant

// CHECK: static dpct::constant_memory<const float, 0> aaa((float)(1ll << 40));
// CHECK-NEXT: static dpct::constant_memory<const float, 0> bbb((float)(1ll << 20));

#include "constant_header.h"
