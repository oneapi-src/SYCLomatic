// RUN: cd %T
// RUN: mkdir hh_constant
// RUN: cd hh_constant
// RUN: cat %s > hh_constant.cu
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: dpct hh_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x c --cuda-host-only
// RUN: dpct hh_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x c --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/hh_constant/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./hh_constant

// CHECK: /*
// CHECK-NEXT: DPCT1056:{{[0-9]+}}: DPCT did not detect that variable aaa is used in device code. If
// CHECK-NEXT: this variable is also used in device code, you need rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: static const float aaa = (float)(1ll << 40);
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1056:{{[0-9]+}}: DPCT did not detect that variable bbb is used in device code. If
// CHECK-NEXT: this variable is also used in device code, you need rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: static const float bbb = (float)(1ll << 20);

#include "constant_header.h"
