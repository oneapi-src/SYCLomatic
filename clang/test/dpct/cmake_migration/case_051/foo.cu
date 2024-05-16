// RUN: dpct --format-range=none -out-root %T/foo %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/foo/foo.dp.cpp

// CHECK: void foo() {}
__global__ void foo() {}
