// UNSUPPORTED: -windows-
// RUN: cd %T
// RUN: mkdir hdd_constant_db
// RUN: cd hdd_constant_db
// RUN: cat %s > hdd_constant_db.cpp
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"c++ hdd_constant_db.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/hdd_constant_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/hdd_constant_db/hdd_constant_db.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc hdd_constant_db.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/hdd_constant_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/hdd_constant_db/hdd_constant_db.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc hdd_constant_db.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/hdd_constant_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/hdd_constant_db/hdd_constant_db.cpp\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/hdd_constant_db/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./hdd_constant_db

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
