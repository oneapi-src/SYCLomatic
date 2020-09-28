// RUN: cd %T
// RUN: mkdir dhd_constant
// RUN: cd dhd_constant
// RUN: cat %s > dhd_constant.cu
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: dpct dhd_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: dpct dhd_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x c --cuda-host-only
// RUN: dpct dhd_constant.cu --out-root=./out --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/dhd_constant/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./dhd_constant

// CHECK: /*
// CHECK-NEXT: DPCT1057:{{[0-9]+}}: Variable aaa was used in host code and device code. The Intel(R)
// CHECK-NEXT: DPC++ Compatibility Tool updated aaa type to be used in SYCL device code and
// CHECK-NEXT: generated new aaa_host_ct1 to be used in host code. You need to update the host
// CHECK-NEXT: code manually to use the new aaa_host_ct1.
// CHECK-NEXT: */
// CHECK-NEXT: static const float aaa_host_ct1 = (float)(1ll << 40);
// CHECK-NEXT: static dpct::constant_memory<const float, 0> aaa((float)(1ll << 40));
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1057:{{[0-9]+}}: Variable bbb was used in host code and device code. The Intel(R)
// CHECK-NEXT: DPC++ Compatibility Tool updated bbb type to be used in SYCL device code and
// CHECK-NEXT: generated new bbb_host_ct1 to be used in host code. You need to update the host
// CHECK-NEXT: code manually to use the new bbb_host_ct1.
// CHECK-NEXT: */
// CHECK-NEXT: static const float bbb_host_ct1 = (float)(1ll << 20);
// CHECK-NEXT: static dpct::constant_memory<const float, 0> bbb((float)(1ll << 20));

#include "constant_header.h"
