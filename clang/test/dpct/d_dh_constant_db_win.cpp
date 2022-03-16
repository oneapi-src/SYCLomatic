// UNSUPPORTED: -linux-
// RUN: cd %T
// RUN: mkdir d_dh_constant_db_win
// RUN: cd d_dh_constant_db_win
// RUN: cat %s > d_dh_constant_db_win.cpp
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: dpct d_dh_constant_db_win.cpp --out-root=./out --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"CudaCompile d_dh_constant_db_win.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/d_dh_constant_db_win\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/d_dh_constant_db_win/d_dh_constant_db_win.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"ClCompile d_dh_constant_db_win.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/d_dh_constant_db_win\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/d_dh_constant_db_win/d_dh_constant_db_win.cpp\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/d_dh_constant_db_win/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./d_dh_constant_db_win

// CHECK: /*
// CHECK-NEXT: DPCT1057:{{[0-9]+}}: Variable aaa was used in host code and device code. aaa type was
// CHECK-NEXT: updated to be used in SYCL device code and new aaa_host_ct1 was generated to be
// CHECK-NEXT: used in host code. You need to update the host code manually to use the new
// CHECK-NEXT: aaa_host_ct1.
// CHECK-NEXT: */
// CHECK-NEXT: static const float aaa_host_ct1 = (float)(1ll << 40);
// CHECK-NEXT: static dpct::constant_memory<const float, 0> aaa((float)(1ll << 40));
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1057:{{[0-9]+}}: Variable bbb was used in host code and device code. bbb type was
// CHECK-NEXT: updated to be used in SYCL device code and new bbb_host_ct1 was generated to be
// CHECK-NEXT: used in host code. You need to update the host code manually to use the new
// CHECK-NEXT: bbb_host_ct1.
// CHECK-NEXT: */
// CHECK-NEXT: static const float bbb_host_ct1 = (float)(1ll << 20);
// CHECK-NEXT: static dpct::constant_memory<const float, 0> bbb((float)(1ll << 20));


#include "constant_header.h"
