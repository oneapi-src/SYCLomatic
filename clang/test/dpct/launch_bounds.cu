// RUN: dpct --format-range=none --out-root %T/launch_bounds %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/launch_bounds/launch_bounds.dp.cpp

// CHECK: void foo() {}
__global__ void __launch_bounds__(32, 1, 1) foo() {}
