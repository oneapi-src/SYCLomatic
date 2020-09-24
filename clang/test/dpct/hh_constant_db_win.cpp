// UNSUPPORTED: -linux-
// RUN: cd %T
// RUN: mkdir hh_constant_db_win
// RUN: cd hh_constant_db_win
// RUN: cat %s > hh_constant_db_win.cpp
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"ClCompile hh_constant_db_win.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/hh_constant_db_win\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/hh_constant_db_win/hh_constant_db_win.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"ClCompile hh_constant_db_win.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/hh_constant_db_win\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/hh_constant_db_win/hh_constant_db_win.cpp\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/hh_constant_db_win/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./hh_constant_db_win

// CHECK: /*
// CHECK-NEXT: DPCT1056:{{[0-9]+}}: The Intel(R) DPC++ Compatibility Tool did not detect the variable
// CHECK-NEXT: aaa used in device code. If this variable is also used in device code, you need
// CHECK-NEXT: to rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: static const float aaa = (float)(1ll << 40);
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1056:{{[0-9]+}}: The Intel(R) DPC++ Compatibility Tool did not detect the variable
// CHECK-NEXT: bbb used in device code. If this variable is also used in device code, you need
// CHECK-NEXT: to rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: static const float bbb = (float)(1ll << 20);

#include "constant_header.h"
