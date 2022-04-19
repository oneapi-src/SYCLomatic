// UNSUPPORTED: -linux-
// RUN: cd %T
// RUN: mkdir dd_constant_db_win
// RUN: cd dd_constant_db_win
// RUN: cat %s > dd_constant_db_win.cpp
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"CudaCompile dd_constant_db_win.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/dd_constant_db_win\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/dd_constant_db_win/dd_constant_db_win.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"CudaCompile dd_constant_db_win.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/dd_constant_db_win\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/dd_constant_db_win/dd_constant_db_win.cpp\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: c2s -p=. --out-root=./out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/dd_constant_db_win/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./dd_constant_db_win

// CHECK: static c2s::constant_memory<const float, 0> aaa((float)(1ll << 40));
// CHECK-NEXT: static c2s::constant_memory<const float, 0> bbb((float)(1ll << 20));

#include "constant_header.h"
