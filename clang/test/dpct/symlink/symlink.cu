// UNSUPPORTED: system-windows
// RUN: rm -rf %T/%basename_t && mkdir -p %T/%basename_t && cd %T/%basename_t
// RUN: ln -s `which dpct` ./dpct
// RUN: cp %S/symlink.cu ./symlink.cu
// RUN: cp %S/lib.cuh ./lib.cuh
// RUN: ./dpct -out-root ./symlink ./symlink.cu --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file ./symlink/symlink.dp.cpp --match-full-lines %s
// RUN: rm -rf %T/%basename_t

// CHECK: #include "lib.dp.hpp"
#include "lib.cuh"

int main() { return add(4, -4); }
