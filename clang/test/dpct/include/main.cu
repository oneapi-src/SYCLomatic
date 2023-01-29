// UNSUPPORTED: -windows-
// RUN: echo "[" > %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/main.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/main.cu\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/test1.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/test1.cu\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/test3.cuh\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/test3.cuh\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/test5.h\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/test5.h\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/test7.h\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/test7.h\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/test9.cc\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/test9.cc\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/test11.cc\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/test11.cc\"" >> %T/compile_commands.json
// RUN: echo "    }" >> %T/compile_commands.json
// RUN: echo "]" >> %T/compile_commands.json

// RUN: dpct -in-root=%S -p=%T --out-root=%T --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/main.dp.cpp

// case 1: cu file in database.
// CHECK: #include "test1.dp.cpp"
#include "test1.cu"
// case 2: cu file not in database.
// CHECK: #include "test2.dp.cpp"
#include "test2.cu"
// case 3: cuh file in database.
// CHECK: #include "test3.dp.hpp"
#include "test3.cuh"
// case 4: cuh file not in database.
// CHECK: #include "test4.dp.hpp"
#include "test4.cuh"
// case 5: header file has CUDA syntax, in database.
// CHECK: #include "test5.h.dp.cpp"
#include "test5.h"
// case 6: header file has CUDA syntax, not in database.
// CHECK: #include "test6.h"
#include "test6.h"
// case 7: header file does not have CUDA syntax, in database.
// CHECK: #include "test7.h"
#include "test7.h"
// case 8: header file does not have CUDA syntax, not in database.
// CHECK: #include "test8.h"
#include "test8.h"
// case 9: source file has CUDA syntax, in database.
// CHECK: #include "test9.cc.dp.cpp"
#include "test9.cc"
// case 10: source file has CUDA syntax, not in database.
// CHECK: #include "test10.cc"
#include "test10.cc"
// case 11: source file does not have CUDA syntax, in database.
// CHECK: #include "test11.cc"
#include "test11.cc"
// case 12: source file does not have CUDA syntax, not in database.
// CHECK: #include "test12.cc"
#include "test12.cc"

__global__ void f() { }

int main() {
  return 0;
}
