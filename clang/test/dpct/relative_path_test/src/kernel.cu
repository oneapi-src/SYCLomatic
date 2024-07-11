// RUN: cd %S/../build
// RUN: dpct -in-root ../src -out-root=%T -p ./  --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/kernel.dp.cpp -o %T/kernel.dp.o %}

#include "cuda_runtime.h"
#include <stdio.h>

// CHECK: void kernel(){}
__global__ void kernel(){}
