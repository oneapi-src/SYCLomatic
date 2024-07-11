// UNSUPPORTED: system-windows

// RUN: rm -f %T/temp
// RUN: mkdir -p %T/temp1
// RUN: ln -s %T/temp1 %T/temp
// RUN: TMPDIR=%T/temp dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMalloc | FileCheck %s
// CHECK-NOT: dpct exited with code: {{.*}}
