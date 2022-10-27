// UNSUPPORTED: -windows-
// RUN: cd %T
// RUN: mkdir dhd_constant_db
// RUN: cd dhd_constant_db
// RUN: cat %s > dhd_constant_db.cpp
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc dhd_constant_db.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/dhd_constant_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/dhd_constant_db/dhd_constant_db.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"c++ dhd_constant_db.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/dhd_constant_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/dhd_constant_db/dhd_constant_db.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc dhd_constant_db.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/dhd_constant_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/dhd_constant_db/dhd_constant_db.cpp\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/dhd_constant_db/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./dhd_constant_db

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
