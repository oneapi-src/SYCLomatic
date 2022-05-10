// UNSUPPORTED: -windows-
// RUN: cd %T
// RUN: mkdir dd_constant_db
// RUN: cd dd_constant_db
// RUN: cat %s > dd_constant_db.cpp
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc dd_constant_db.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/dd_constant_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/dd_constant_db/dd_constant_db.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc dd_constant_db.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/dd_constant_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/dd_constant_db/dd_constant_db.cpp\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/dd_constant_db/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./dd_constant_db

// CHECK: static dpct::constant_memory<const float, 0> aaa((float)(1ll << 40));
// CHECK-NEXT: static dpct::constant_memory<const float, 0> bbb((float)(1ll << 20));

#include "constant_header.h"
