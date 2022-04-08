// RUN: cd %T
// RUN: mkdir dh_constant
// RUN: cd dh_constant
// RUN: cat %s > dh_constant.cu
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: c2s dh_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: c2s dh_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x c --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/dh_constant/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./dh_constant

// CHECK: /*
// CHECK-NEXT: DPCT1057:{{[0-9]+}}: Variable aaa was used in host code and device code. aaa type was
// CHECK-NEXT: updated to be used in SYCL device code and new aaa_host_ct1 was generated to be
// CHECK-NEXT: used in host code. You need to update the host code manually to use the new
// CHECK-NEXT: aaa_host_ct1.
// CHECK-NEXT: */
// CHECK-NEXT: static const float aaa_host_ct1 = (float)(1ll << 40);
// CHECK-NEXT: static c2s::constant_memory<const float, 0> aaa((float)(1ll << 40));
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1057:{{[0-9]+}}: Variable bbb was used in host code and device code. bbb type was
// CHECK-NEXT: updated to be used in SYCL device code and new bbb_host_ct1 was generated to be
// CHECK-NEXT: used in host code. You need to update the host code manually to use the new
// CHECK-NEXT: bbb_host_ct1.
// CHECK-NEXT: */
// CHECK-NEXT: static const float bbb_host_ct1 = (float)(1ll << 20);
// CHECK-NEXT: static c2s::constant_memory<const float, 0> bbb((float)(1ll << 20));

#include "constant_header.h"
