// RUN: dpct -in-root=%S --out-root=%T/test5 %S/test5.cu --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/test5/test5.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/test5/test5.dp.cpp -o %T/test5/main.dp.o %}
// CHECK: #include "test6.cpp"
#include "test6.cpp"
#include <cuda_runtime.h>

__global__ void kernel() {}
