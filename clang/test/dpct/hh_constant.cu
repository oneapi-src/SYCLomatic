// RUN: cd %T
// RUN: mkdir hh_constant
// RUN: cd hh_constant
// RUN: cat %s > hh_constant.cu
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: c2s hh_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x c --cuda-host-only
// RUN: c2s hh_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x c --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/hh_constant/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./hh_constant

// CHECK: /*
// CHECK-NEXT: DPCT1056:{{[0-9]+}}: The use of variable aaa in device code was not detected. If this
// CHECK-NEXT: variable is also used in device code, you need to rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: static const float aaa = (float)(1ll << 40);
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1056:{{[0-9]+}}: The use of variable bbb in device code was not detected. If this
// CHECK-NEXT: variable is also used in device code, you need to rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: static const float bbb = (float)(1ll << 20);

#include "constant_header.h"
