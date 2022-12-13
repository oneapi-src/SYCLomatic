// UNSUPPORTED: -windows-
// RUN: echo "[" > %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/main.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/main.cu\"" >> %T/compile_commands.json
// RUN: echo "    }" >> %T/compile_commands.json
// RUN: echo "]" >> %T/compile_commands.json

// RUN: dpct -in-root=%S -p=%T --out-root=%T --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/main.dp.cpp

// CHECK: #include "test.dp.hpp"
#include "test.cuh"
// CHECK: #include "test.cc"
#include "test.cc"

__global__ void f() { y = x; }

int main() {
  return 0;
}
